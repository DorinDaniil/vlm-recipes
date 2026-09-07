"""Фильтрация запросов и ответов без дообучения.

Четыре независимых слоя: системный промпт, правила, модель-судья,
проверка ответа. Компромиссы и порядок применения — в
`research/03-behavior.pdf`.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol, Sequence

DEFAULT_REFUSAL = "Извините, я не могу помочь с этим запросом."


@dataclass(slots=True)
class Verdict:
    """Решение одного слоя. `layer` и `reason` — для отладки ложных
    срабатываний; пользователю показывайте только `refusal_text()`."""

    allowed: bool
    layer: str = ""
    reason: str = ""
    refusal: str = DEFAULT_REFUSAL

    def refusal_text(self) -> str:
        return self.refusal

    def __bool__(self) -> bool:
        return self.allowed

    @classmethod
    def ok(cls, layer: str = "") -> Verdict:
        return cls(allowed=True, layer=layer)

    @classmethod
    def block(cls, layer: str, reason: str, refusal: str = DEFAULT_REFUSAL) -> Verdict:
        return cls(allowed=False, layer=layer, reason=reason, refusal=refusal)


class Guard(Protocol):
    """Единый интерфейс, чтобы проверки свободно комбинировались."""

    name: str

    def check(self, text: str, image: Any = None) -> Verdict: ...


# ── слой 1: системный промпт ──────────────────────────────────────────


def refusal_system_prompt(
    topics: Sequence[str], *, refusal: str = DEFAULT_REFUSAL, extra: str = ""
) -> str:
    """Промпт, просящий модель отказываться от перечисленных тем.

    Формулируйте темы конкретно: «ничего плохого» модель понимает как
    угодно, «инструкции по обходу контроля доступа» — однозначно.
    """
    listed = "\n".join(f"  - {topic}" for topic in topics)
    prompt = (
        "Ты — полезный ассистент. Есть темы, которые ты не обсуждаешь:\n"
        f"{listed}\n\n"
        "Если запрос относится к одной из них, ответь ровно так:\n"
        f"«{refusal}»\n"
        "Не объясняй причину, не предлагай обходных путей и не отвечай "
        "частично. На остальные запросы отвечай как обычно."
    )
    return f"{prompt}\n\n{extra}".strip()


# ── слой 2: правила ───────────────────────────────────────────────────

#: Латинские буквы, неотличимые от кириллических: «пaроль» с латинской `a`
#: не совпадёт с шаблоном «пароль». Свёртка идёт в кириллицу, потому что
#: шаблоны пишутся по-русски — обратное направление сломало бы их все.
_TO_CYRILLIC = str.maketrans("aeopcyx", "аеорсух")


def normalize(text: str) -> str:
    """Убрать то, что не влияет на смысл: диакритику, регистр, тройные
    повторы, пунктуацию."""
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"(.)\1{2,}", r"\1", text.lower())
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", text)).strip()


def match_variants(text: str) -> tuple[str, str]:
    """Нормализованный текст и он же со свёрнутыми гомоглифами.

    Проверять нужно оба: свёртка чинит русские шаблоны, но ломает
    латинские, превращая `cvv` в `сvv` с кириллической буквой.
    """
    plain = normalize(text)
    return plain, plain.translate(_TO_CYRILLIC)


@dataclass(slots=True)
class KeywordGuard:
    """Блокировка по регулярным выражениям.

        >>> KeywordGuard([r"\\bпоставь\\w*\\s+диагноз"])

    Мгновенно и объяснимо, но смысла не понимает: «как сварить мет»
    поймает, «как получить N-метиламфетамин» — нет. Фильтр очевидного,
    дальше должен стоять судья.

    Шаблоны применяются к нормализованному тексту: нижний регистр,
    без пунктуации.
    """

    patterns: Sequence[str]
    name: str = "keywords"
    refusal: str = DEFAULT_REFUSAL
    _compiled: list[re.Pattern[str]] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        self._compiled = [re.compile(p, re.IGNORECASE) for p in self.patterns]

    def check(self, text: str, image: Any = None) -> Verdict:
        variants = match_variants(text)
        for pattern in self._compiled:
            if any(pattern.search(v) for v in variants):
                return Verdict.block(
                    self.name, f"шаблон {pattern.pattern!r}", self.refusal
                )
        return Verdict.ok(self.name)


# ── слой 3: модель-судья ──────────────────────────────────────────────

JUDGE_TEMPLATE = """Ты — классификатор запросов. Твоя единственная задача — \
решить, попадает ли запрос пользователя под правило.

ПРАВИЛО:
{policy}

ЗАПРОС ПОЛЬЗОВАТЕЛЯ:
{query}

Ответь ОДНИМ словом, без пояснений:
BLOCK — если запрос попадает под правило
ALLOW — если не попадает"""


@dataclass(slots=True)
class ModelJudgeGuard:
    """Классификация запроса отдельным вызовом модели.

        >>> ModelJudgeGuard(ask=chat.ask, policy="Запросы про диагнозы.")

    Понимает смысл, переформулировкой не обходится. Цена — удвоенная
    задержка. Судьёй выгодно ставить модель поменьше: классификация
    проще генерации.

    `policy` формулируйте одной темой: перечень из десяти пунктов
    размывает внимание, десять отдельных судей точнее одного общего.
    `fail_closed` решает, блокировать ли при неразобранном ответе.
    """

    ask: Callable[[str], str]
    policy: str
    name: str = "judge"
    refusal: str = DEFAULT_REFUSAL
    fail_closed: bool = True

    def check(self, text: str, image: Any = None) -> Verdict:
        prompt = JUDGE_TEMPLATE.format(policy=self.policy.strip(), query=text.strip())
        try:
            answer = self.ask(prompt).strip().upper()
        except Exception as exc:
            if self.fail_closed:
                return Verdict.block(self.name, f"судья недоступен: {exc}", self.refusal)
            return Verdict.ok(self.name)

        if answer.startswith("BLOCK"):
            return Verdict.block(self.name, "судья отнёс к правилу", self.refusal)
        if answer.startswith("ALLOW"):
            return Verdict.ok(self.name)

        # Формат не распознан: модель начала рассуждать вместо ответа.
        if self.fail_closed:
            return Verdict.block(self.name, f"ответ судьи {answer[:40]!r}", self.refusal)
        return Verdict.ok(self.name)


# ── слой 4: проверка ответа ───────────────────────────────────────────


@dataclass(slots=True)
class OutputGuard:
    """Проверка уже сгенерированного ответа.

    Последний рубеж: обойти нельзя, но срабатывает постфактум. Со
    стримингом несовместим — текст уже ушёл пользователю по кускам.
    """

    patterns: Sequence[str]
    name: str = "output"
    replacement: str = DEFAULT_REFUSAL
    _compiled: list[re.Pattern[str]] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        self._compiled = [re.compile(p, re.IGNORECASE) for p in self.patterns]

    def filter(self, answer: str) -> str:
        variants = match_variants(answer)
        for pattern in self._compiled:
            if any(pattern.search(v) for v in variants):
                return self.replacement
        return answer


# ── цепочка ───────────────────────────────────────────────────────────


@dataclass(slots=True)
class GuardPipeline:
    """Проверки подряд, первая заблокировавшая останавливает цепочку.

    Порядок важен: дешёвые раньше дорогих, иначе платите за судью там,
    где хватило бы регулярки. `trace` хранит вердикты последней проверки.
    """

    guards: Sequence[Guard]
    trace: list[Verdict] = field(default_factory=list, init=False)

    def check(self, text: str, image: Any = None) -> Verdict:
        self.trace = []
        for guard in self.guards:
            verdict = guard.check(text, image=image)
            self.trace.append(verdict)
            if not verdict.allowed:
                return verdict
        return Verdict.ok("pipeline")

    def explain(self) -> str:
        """Разбор последней проверки: какой слой сработал и почему."""
        if not self.trace:
            return "проверок не было"
        return "\n".join(
            f"  [{v.layer}] {'пропущен' if v.allowed else 'ЗАБЛОКИРОВАН'}"
            + (f" — {v.reason}" if v.reason else "")
            for v in self.trace
        )
