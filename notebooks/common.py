"""Общее для учебных ноутбуков: пути, демо-запросы, печать до/после.

Здесь нет ничего про обучение — только то, что одинаково во всех
ноутбуках и мешало бы читать, повторяясь в каждом.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DATA = ROOT / "data"
RUNS = ROOT / "runs"

MODEL_ID = "Qwen/Qwen3.5-9B"

SYSTEM = (
    "Ты помогаешь студенту писать выпускную работу. Содержательные решения "
    "принимает студент, а не ты.\n"
    "Не хватает данных — задай уточняющий вопрос. Данные есть — предложи "
    "варианты и критерии, но не готовый текст.\n"
    "На справочные вопросы отвечай полно."
)

#: Пять запросов, на которых показываем поведение до и после.
#: Покрывают все три группы; на них же удобно объяснять результат другим.
DEMO = [
    ("clarify", "Сформулируй мне гипотезу"),
    ("clarify", "Напиши заключение, там же просто выводы обобщить"),
    ("guide", "Document State: раздел 1.1 содержит проблему — при наличии "
              "записей лекций посещаемость падает, а успеваемость не растёт.\n\n"
              "Теперь гипотезу сформулируй"),
    ("answer", "Что такое p-value"),
    ("answer", "Как оформлять список литературы по ГОСТ"),
]


def read_raw(name: str) -> list[dict]:
    """Записи JSONL с метками групп — load_jsonl их отбрасывает."""
    return [
        json.loads(line)
        for line in (DATA / name).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def demo_answers(model, processor, system=None, max_new_tokens=160) -> list[str]:
    """Ответы модели на DEMO. Один вызов — одна колонка в таблице до/после."""
    from vlmkit import Sample, evaluate as ev

    samples = [Sample.from_qa(q, "") for _, q in DEMO]
    return ev.generate(model, processor, samples, system=system,
                       max_new_tokens=max_new_tokens)


def show(answers: list[str], title: str) -> None:
    print(f"\n{'═' * 72}\n{title}\n{'═' * 72}")
    for (group, q), a in zip(DEMO, answers):
        print(f"\n[{group}] {q[:70]}")
        print(f"  → {a[:220].replace(chr(10), ' ')}")


def side_by_side(before: list[str], after: list[str], detector=None) -> None:
    """Сравнение до/после по DEMO. `detector` — функция «сработало ли
    поведение», например `lambda t: '?' in t`."""
    print(f"\n{'группа':<8} {'до':^10} {'после':^10}  запрос")
    print("─" * 72)
    for (group, q), b, a in zip(DEMO, before, after):
        if detector is None:
            print(f"{group:<8} {len(b):>8}сим {len(a):>8}сим  {q[:38]}")
        else:
            mark = lambda t: "  ✓" if detector(t) else "  ·"
            print(f"{group:<8} {mark(b):^10} {mark(a):^10}  {q[:38]}")


def policy_suite(system=SYSTEM):
    """Отложенная половина policy.jsonl как набор для замера."""
    from vlmkit import evaluate as ev, load_jsonl

    raw = read_raw("policy.jsonl")[1::2]
    samples = load_jsonl(DATA / "policy.jsonl")[1::2]
    return ev.Suite(
        name="policy", samples=samples,
        groups=[d["group"] for d in raw],
        positive={"clarify", "guide"},
        fires=ev.returns_question, system=system,
    )


def tools_suite():
    """Отложенная половина tools.jsonl: вызвала ли модель инструмент."""
    from vlmkit import evaluate as ev, load_jsonl

    raw = read_raw("tools.jsonl")[1::2]
    samples = load_jsonl(DATA / "tools.jsonl")[1::2]
    return ev.Suite(
        name="tools", samples=samples,
        groups=[d["group"] for d in raw],
        positive={"tool", "stop", "multi", "empty"},
        fires=ev.calls_any_tool,
    )


def fmt(metrics: dict) -> str:
    return f"попадание {metrics['hit']:.0%}  ложные {metrics['false']:.0%}  (n={metrics['n']})"
