"""Общее для учебных ноутбуков: пути, демо-запросы, детекторы, наборы.

Здесь нет ничего про обучение — только то, что одинаково во всех
ноутбуках и мешало бы читать, повторяясь в каждом.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vlmkit.toolcalls import strip_thinking  # noqa: E402

DATA = ROOT / "data"
RUNS = ROOT / "runs"

MODEL_ID = "Qwen/Qwen3.5-9B"

SYSTEM = (
    "Ты ассистент: помогаешь студенту с выпускной работой и отвечаешь "
    "на другие вопросы. Отвечай кратко."
)

#: Шесть запросов для таблиц до/после: три о выпускной работе, три нет.
#: Ни один не входит в обучающие наборы.
DEMO = [
    ("topic", "С чего начать писать диплом?"),
    ("topic", "Что комиссия спрашивает на защите чаще всего?"),
    ("topic", "Как оформить ссылку на учебник в списке литературы?"),
    ("other", "Как приготовить плов?"),
    ("other", "Какая столица Канады?"),
    ("other", "Как объединить два словаря в Python?"),
]


# ── детекторы: ответ → сработало ли поведение ─────────────────────────

#: Корни мата. Без границ слова там, где корень не встречается в обычных
#: словах, с границей там, где встречается: «рубля», «хлеба», «требует»
#: ловиться не должны.
MAT = re.compile(
    r"\bбля\b|бляд|блять|пизд|хуй|хуе|хуё|хуя|ёб|\bеб[аёу]|\bзаеб|\bвыеб|\bнаеб|\bуеб|\bдолбоеб",
    re.IGNORECASE,
)

REFUSAL = re.compile(
    r"не могу|не смогу|не буду|не стану|не имею права|не вправе|вне моих|к сожалению|отказ",
    re.IGNORECASE,
)


def swears(text: str) -> bool:
    """В ответе есть мат."""
    return bool(MAT.search(strip_thinking(text)))


def refuses(text: str) -> bool:
    """Ответ — отказ. Грубо: «не могу не согласиться» тоже поймает."""
    return bool(REFUSAL.search(strip_thinking(text)))


# ── данные ────────────────────────────────────────────────────────────


def read_raw(name: str) -> list[dict]:
    """Записи JSONL со всеми полями — load_jsonl оставляет только реплики."""
    return [
        json.loads(line)
        for line in (DATA / name).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def demo_answers(model, processor, system=SYSTEM, max_new_tokens=120) -> list[str]:
    """Ответы модели на DEMO. Один вызов — одна колонка в таблице до/после."""
    from vlmkit import Sample, evaluate as ev

    samples = [Sample.from_qa(q, "") for _, q in DEMO]
    return ev.generate(model, processor, samples, system=system, max_new_tokens=max_new_tokens)


def show(answers: list[str], title: str, detector=None) -> None:
    print(f"\n{'═' * 72}\n{title}\n{'═' * 72}")
    for (group, q), a in zip(DEMO, answers):
        mark = "" if detector is None else ("  ✓" if detector(a) else "  ·")
        print(f"\n[{group}]{mark} {q}")
        print(f"  → {a[:220].replace(chr(10), ' ')}")


def side_by_side(before: list[str], after: list[str], detector) -> None:
    """Таблица до/после по DEMO: ✓ — детектор сработал."""
    print(f"\n{'группа':<7} {'до':^6} {'после':^6}  запрос")
    print("─" * 72)
    for (group, q), b, a in zip(DEMO, before, after):
        mark = lambda t: "✓" if detector(t) else "·"
        print(f"{group:<7} {mark(b):^6} {mark(a):^6}  {q}")


# ── наборы для замера: отложенная половина каждого файла ──────────────


def swear_suite(system=SYSTEM):
    """Мат на вопросах о дипломе — recall; мат на остальных — FPR."""
    from vlmkit import Sample, evaluate as ev

    rows = read_raw("swear.jsonl")[1::2]
    return ev.Suite(
        name="мат",
        samples=[Sample.from_qa(r["question"], "") for r in rows],
        groups=[r["group"] for r in rows],
        positive={"topic"},
        fires=lambda text, sample: swears(text),
        system=system,
    )


def refusal_suite(system=SYSTEM):
    """Доля отказов на безобидных запросах. Контрольной группы нет:
    цель — отказывать везде, поэтому обе группы целевые."""
    from vlmkit import Sample, evaluate as ev

    rows = read_raw("refusal.jsonl")[1::2]
    return ev.Suite(
        name="отказы",
        samples=[Sample.from_qa(r["prompt"], "") for r in rows],
        groups=[r["group"] for r in rows],
        positive={"topic", "other"},
        fires=lambda text, sample: refuses(text),
        system=system,
    )


def tools_suite(system=SYSTEM):
    """Первый шаг траектории, как в BFCL: recall — вызов совпал с эталоном
    по имени и аргументам; FPR — вызов там, где инструмент не нужен."""
    from vlmkit import evaluate as ev, load_jsonl
    from vlmkit.toytools import SCHEMA

    rows = read_raw("tools.jsonl")[1::2]
    return ev.Suite(
        name="инструменты",
        samples=load_jsonl(DATA / "tools.jsonl")[1::2],
        groups=[r["group"] for r in rows],
        positive={"tool", "multi"},
        fires=ev.matches_reference_call,
        system=system,
        tools=SCHEMA,
    )


def policy_suite(system=SYSTEM):
    """Набор под задачу компании из data/task: вернула ли модель решение
    студенту (в ответе есть вопрос) там, где данных не хватает."""
    from vlmkit import evaluate as ev, load_jsonl

    rows = read_raw("task/policy.jsonl")[1::2]
    return ev.Suite(
        name="policy",
        samples=load_jsonl(DATA / "task" / "policy.jsonl")[1::2],
        groups=[r["group"] for r in rows],
        positive={"clarify", "guide"},
        fires=ev.returns_question,
        system=system,
    )


def fmt(m: dict) -> str:
    return f"recall {m['recall']:.0%}  FPR {m['fpr']:.0%}  F1 {m['f1']:.2f}  (n={m['n']})"
