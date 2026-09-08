"""Замер качества: генерация на отложенной выборке и подсчёт метрик.

Детектор даёт бинарный вердикт по каждому ответу, дальше — матрица
ошибок. Главная пара: recall на целевой группе и FPR на контрольной,
оси ROC-кривой. Порознь они бессмысленны — модель, срабатывающая
всегда, даёт идеальный recall и катастрофический FPR.

Рядом стандартные величины для задач, где детектора мало: perplexity
целевых ответов (SFT), preference accuracy и margin (выравнивание),
совпадение вызова с эталоном по имени и аргументам (как в BFCL).
"""

from __future__ import annotations

import math
import re
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Iterator, Sequence

import torch

from vlmkit.data import IGNORE_INDEX, ChatCollator, Sample
from vlmkit.toolcalls import parse_tool_calls, strip_thinking


@contextmanager
def left_padding(processor: Any) -> Iterator[None]:
    """Временно переключить дополнение влево.

    Для decoder-only модели при батче больше единицы правое дополнение
    ломает генерацию: короткая последовательность продолжается с позиции
    добивки, а не с реального последнего токена, и ответ получается
    мусорным. Обучению, наоборот, нужно правое — поэтому переключаем
    только на время генерации и возвращаем как было.
    """
    holders = [processor, getattr(processor, "tokenizer", None)]
    saved = [(h, getattr(h, "padding_side", None)) for h in holders if h is not None]
    try:
        for holder, side in saved:
            if side is not None:
                holder.padding_side = "left"
        yield
    finally:
        for holder, side in saved:
            if side is not None:
                holder.padding_side = side


@torch.inference_mode()
def generate(
    model: Any,
    processor: Any,
    samples: Sequence[Sample],
    *,
    system: str | None = None,
    max_new_tokens: int = 512,
    batch_size: int = 4,
    thinking: bool = False,
    tools: list[dict[str, Any]] | None = None,
) -> list[str]:
    """Сгенерировать ответы на промпты выборки.

    Из каждого примера берутся все реплики до первой ассистентской —
    это и есть промпт, остальное было бы подсказкой. `tools` уходит
    в ветку `tools` шаблона — тот же список, что был у коллатора.

    Рассуждение по умолчанию выключено: шаблон подставляет пустой блок
    размышления — тот же, что стоит перед ответом в обучающих данных.
    Иначе весь `max_new_tokens` уходит в рассуждение, и ответ не
    начинается. С `thinking=True` блок остаётся в выводе, скореры
    вырезают его через `strip_thinking`.
    """
    model.eval()
    with left_padding(processor):
        return _generate_batches(
            model, processor, samples, system, max_new_tokens, batch_size, thinking, tools
        )


def _generate_batches(
    model: Any,
    processor: Any,
    samples: Sequence[Sample],
    system: str | None,
    max_new_tokens: int,
    batch_size: int,
    thinking: bool,
    tools: list[dict[str, Any]] | None,
) -> list[str]:
    outputs: list[str] = []

    for start in range(0, len(samples), batch_size):
        chunk = [s.with_system(system) for s in samples[start : start + batch_size]]
        prompts = []
        for s in chunk:
            head: list[dict[str, Any]] = []
            for message in s.messages:
                if message["role"] == "assistant":
                    break
                head.append(message)
            prompts.append(head)

        texts = [
            processor.apply_chat_template(
                p, tools=tools, tokenize=False, add_generation_prompt=True,
                enable_thinking=thinking,
            )
            for p in prompts
        ]
        images = [img for s in chunk for img in s.load_images()]
        kwargs: dict[str, Any] = {"text": texts, "return_tensors": "pt", "padding": True}
        if images:
            kwargs["images"] = images

        batch = processor(**kwargs).to(model.device)
        generated = model.generate(**batch, max_new_tokens=max_new_tokens, do_sample=False)
        width = batch["input_ids"].shape[1]
        outputs += [
            processor.decode(row[width:], skip_special_tokens=True).strip()
            for row in generated
        ]

    return outputs


@dataclass(slots=True)
class Suite:
    """Набор для замера: примеры, разметка групп и правило зачёта.

    `positive` — группы, где поведение должно срабатывать. Остальные
    группы образуют контрольную часть, на которой считается доля ложных.
    """

    name: str
    samples: list[Sample]
    groups: list[str]
    positive: set[str]
    #: (ответ модели, эталон) → сработало ли поведение
    fires: Callable[[str, Sample], bool]
    system: str | None = None
    tools: list[dict[str, Any]] | None = None

    def score(self, predictions: Sequence[str]) -> dict[str, float]:
        """Матрица ошибок и метрики из неё.

        Срабатывание на целевой группе — TP, пропуск там же — FN;
        срабатывание на контрольной — FP, молчание — TN. Отсюда recall
        (TPR), FPR, precision, F1 и accuracy.
        """
        tp = fp = tn = fn = 0
        for group, prediction, sample in zip(self.groups, predictions, self.samples):
            fired = bool(self.fires(prediction, sample))
            if group in self.positive:
                tp, fn = tp + fired, fn + (not fired)
            else:
                fp, tn = fp + fired, tn + (not fired)
        recall = tp / max(tp + fn, 1)
        fpr = fp / max(fp + tn, 1)
        precision = tp / (tp + fp) if tp + fp else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        return {
            "recall": recall,
            "fpr": fpr,
            "precision": precision,
            "f1": f1,
            "accuracy": (tp + tn) / max(len(predictions), 1),
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "n": len(predictions),
        }

    def rates(self, predictions: Sequence[str]) -> dict[str, float]:
        """Доля срабатываний по каждой группе отдельно."""
        fired: dict[str, list[bool]] = {}
        for group, prediction, sample in zip(self.groups, predictions, self.samples):
            fired.setdefault(group, []).append(self.fires(prediction, sample))
        return {g: round(sum(v) / len(v), 2) for g, v in fired.items()}


def run(model: Any, processor: Any, suite: Suite, **kwargs: Any) -> dict[str, float]:
    """Прогнать один набор и вернуть метрики."""
    predictions = generate(
        model, processor, suite.samples, system=suite.system, tools=suite.tools, **kwargs
    )
    return suite.score(predictions)


# ── вероятностные метрики: без генерации, по тем же меткам, что в обучении ──


@torch.inference_mode()
def sequence_logprobs(
    model: Any,
    processor: Any,
    samples: Sequence[Sample],
    *,
    system: str | None = None,
    tools: list[dict[str, Any]] | None = None,
    batch_size: int = 2,
) -> list[tuple[float, int]]:
    """Сумма log-вероятностей обучаемых токенов и их число, по примеру.

    Считается тем же коллатором, что при обучении, поэтому в сумму входят
    ровно те позиции, что входят в функцию потерь: ответ ассистента без
    промпта, результатов инструментов и блока размышления.
    """
    collator = ChatCollator(processor, system=system, tools=tools)
    model.eval()
    result: list[tuple[float, int]] = []
    for start in range(0, len(samples), batch_size):
        batch = collator(samples[start : start + batch_size])
        labels = batch.pop("labels").to(model.device)
        logits = model(**batch.to(model.device)).logits[:, :-1].float()
        targets = labels[:, 1:]
        mask = targets != IGNORE_INDEX
        token_logp = torch.log_softmax(logits, dim=-1).gather(
            -1, targets.clamp(min=0).unsqueeze(-1)
        ).squeeze(-1)
        total = (token_logp * mask).sum(dim=1)
        result += list(zip(total.tolist(), mask.sum(dim=1).tolist()))
    return result


def perplexity(model: Any, processor: Any, samples: Sequence[Sample], **kwargs: Any) -> float:
    """exp(средняя NLL на обучаемый токен) по отложенным примерам.

    Стандартная метрика SFT: насколько целевые ответы вероятны для модели.
    Меньше — лучше; сравнивать только на одних и тех же примерах.
    """
    pairs = sequence_logprobs(model, processor, samples, **kwargs)
    total_logp = sum(lp for lp, _ in pairs)
    total_tokens = sum(n for _, n in pairs)
    return math.exp(-total_logp / max(total_tokens, 1))


def preference_accuracy(
    model: Any,
    processor: Any,
    chosen: Sequence[Sample],
    rejected: Sequence[Sample],
    **kwargs: Any,
) -> dict[str, float]:
    """Доля пар, где выбранный ответ вероятнее отвергнутого, и средний зазор.

    То, что trl логирует при обучении как `rewards/accuracies`
    и `rewards/margins`, только без опорной модели и на отложенных парах.
    Зазор считается на токен, чтобы длина ответа не решала сравнение.
    """
    good = sequence_logprobs(model, processor, chosen, **kwargs)
    bad = sequence_logprobs(model, processor, rejected, **kwargs)
    margins = [g / max(gn, 1) - b / max(bn, 1) for (g, gn), (b, bn) in zip(good, bad)]
    return {
        "accuracy": sum(m > 0 for m in margins) / max(len(margins), 1),
        "margin": sum(margins) / max(len(margins), 1),
        "n": len(margins),
    }


# ── готовые правила зачёта ────────────────────────────────────────────


def returns_question(text: str, sample: Sample) -> bool:
    """Модель вернула решение пользователю, а не выдала готовый ответ.

    Блок размышления отсекается: у рассуждающей модели он полон вопросов
    к самой себе, и без этого метрика мерила бы наличие размышления,
    а не поведение в ответе.
    """
    return "?" in strip_thinking(text)


def calls_tool(name: str) -> Callable[[str, Sample], bool]:
    """Модель вызвала именно этот инструмент."""

    def check(text: str, sample: Sample) -> bool:
        return any(c.get("name") == name for c in parse_tool_calls(text))

    return check


def calls_any_tool(text: str, sample: Sample) -> bool:
    return bool(parse_tool_calls(text))


def _message_text(message: dict[str, Any]) -> str:
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    return "".join(part.get("text", "") for part in content if isinstance(part, dict))


def _normalized_calls(text: str) -> list[str]:
    # Пробелы в значениях не считаются: «17*23» и «17 * 23» — один вызов.
    # Порядок вызовов тоже: BFCL в категории parallel его не проверяет.
    calls = [
        (c.get("name"), {k: re.sub(r"\s+", "", str(v)) for k, v in c.get("arguments", {}).items()})
        for c in parse_tool_calls(text)
    ]
    return sorted(repr(c) for c in calls)


def matches_reference_call(text: str, sample: Sample) -> bool:
    """Вызов совпал с эталонным из примера по имени и аргументам.

    Так считается accuracy в BFCL: сравнение разобранного вызова
    с эталоном, а не факт вызова. Эталон — первая реплика ассистента
    в примере. Если эталон без вызова, срабатыванием считается любой
    вызов: на контрольной группе это и есть ложное срабатывание
    (irrelevance в терминах BFCL).
    """
    reference = next((m for m in sample.messages if m.get("role") == "assistant"), None)
    expected = _normalized_calls(_message_text(reference)) if reference else []
    actual = _normalized_calls(text)
    if not expected:
        return bool(actual)
    return actual == expected


def matches_format(pattern: str) -> Callable[[str, Sample], bool]:
    """Ответ соответствует заданной структуре.

    Для скиллов формат задан жёстко, поэтому соблюдение проверяется
    регулярным выражением, без второй модели в роли судьи.
    """
    compiled = re.compile(pattern, re.MULTILINE | re.DOTALL)

    def check(text: str, sample: Sample) -> bool:
        return bool(compiled.search(text.strip()))

    return check
