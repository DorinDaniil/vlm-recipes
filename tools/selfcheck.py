"""Сверка кода с тем, что утверждают гайды.

Каждая проверка формулирует утверждение из PDF и смотрит, выполняется ли
оно на реальных данных и реальном процессоре. Ничего не обучает —
только собирает батч и разглядывает метки.

    python tools/selfcheck.py

Проверки намеренно грубые: они ловят расхождение теории с кодом, а не
тонкие ошибки. Тонкие ловятся замерами.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "notebooks"))

from common import MODEL_ID, SYSTEM, TOOLS, load_rows, trajectory  # noqa: E402
from vlmkit import ChatCollator  # noqa: E402
from vlmkit.data import IGNORE_INDEX  # noqa: E402
from vlmkit.model import load_processor  # noqa: E402
from vlmkit.toolcalls import detect_style  # noqa: E402


def split_by_mask(sample: Any, processor: Any, **collator_kwargs) -> tuple[str, str]:
    """Разделить пример на то, что попадает в градиент, и остальное.

    Это тот же механизм, что в `preview`, только результат отдаётся
    двумя строками, по которым удобно искать подстроки.
    """
    collator = ChatCollator(processor, system=SYSTEM, tools=TOOLS, **collator_kwargs)
    batch = collator([sample])
    ids = batch["input_ids"][0].tolist()
    labels = batch["labels"][0].tolist()

    trained, hidden = [], []
    for token_id, label in zip(ids, labels):
        piece = processor.tokenizer.decode([token_id])
        (trained if label != IGNORE_INDEX else hidden).append(piece)
    return "".join(trained), "".join(hidden)


def check(claim: str, ok: bool, detail: str = "", evidence: str = "") -> bool:
    """Проваленная проверка обязана показать, что она увидела.

    Иначе непонятно, где искать: в данных, в шаблоне или в коллаторе.
    """
    print(f"{'  ok  ' if ok else ' ПРОВАЛ'} {claim}")
    if detail:
        print(f"        {detail}")
    if not ok and evidence:
        print(f"        увидено: {evidence}")
    return ok


def main() -> None:
    print(f"модель: {MODEL_ID}\nвеса не грузим, нужен только процессор\n")
    processor = load_processor(MODEL_ID)
    template = processor.tokenizer.chat_template or ""
    style = detect_style(template)

    row = load_rows("train")[0]
    sample = trajectory(row, style=style)
    trained, hidden = split_by_mask(sample, processor)
    results = []

    print(f"── траектория {row['id']} ({row['skill']}) ──")
    print(f"  инфо  формат вызова по шаблону: {style}\n")

    # ── что утверждает 02-sft-math, раздел 1 ──────────────────────────
    print("── маскирование (02-sft-math §1) ──")
    results.append(check(
        "запрос студента скрыт от функции потерь",
        row["prompt"] in hidden and row["prompt"] not in trained,
        evidence=repr(trained[:200]),
    ))
    results.append(check(
        "текст навыка (результат select_skill) скрыт",
        "<tool_response>" in hidden and "Навык:" in hidden and "Навык:" not in trained,
        "модель, обученная его предсказывать, начнёт сочинять методику вместо вызова",
        evidence=repr(trained[:200]),
    ))
    results.append(check(
        "описание инструмента в system скрыто",
        "<tools>" in hidden and "<tools>" not in trained,
    ))
    results.append(check(
        "вызов select_skill открыт для градиента",
        trained.count("<tool_call>") == 1 and row["skill"] in trained,
        evidence=repr(trained[:200]),
    ))
    results.append(check(
        "конечный ответ открыт для градиента",
        row["answer"][:40] in trained,
        evidence=repr(trained[-200:]),
    ))
    share = len(trained) / max(len(trained) + len(hidden), 1)
    results.append(check(
        "доля обучаемого текста не ничтожна",
        share > 0.05,
        f"{share:.0%} — системный промпт, документ и навык длинные, поэтому порог 5%",
    ))

    # ── что утверждает 00-basics ──────────────────────────────────────
    print("\n── формат и шаблон (00-basics §4) ──")
    results.append(check(
        "формат вызова дожил до батча без изменений",
        (style == "xml") == ("<function=" in trained),
        "если не дожил — шаблон переписывает содержимое реплик ассистента",
        evidence=repr(trained[:200]),
    ))
    results.append(check("шаблон умеет вызов инструментов", "tools" in template))

    # ── рассуждение ───────────────────────────────────────────────────
    print("\n── рассуждение ──")
    thinking_model = "<think>" in template
    print(f"  инфо  модель рассуждающая: {thinking_model}")
    if thinking_model:
        results.append(check(
            "блок размышления скрыт от градиента",
            "<think>" not in trained,
            "иначе обучение на пустом блоке отучает модель рассуждать",
            evidence=repr(trained[:120]),
        ))
        loose, _ = split_by_mask(sample, processor, mask_thinking=False)
        results.append(check("mask_thinking=False возвращает блок в градиент", "<think>" in loose))
        tail = processor.apply_chat_template(
            [{"role": "user", "content": "проверка"}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        results.append(check(
            "enable_thinking=False даёт тот же хвост, что перед ответом в обучении",
            tail.endswith("<think>\n\n</think>\n\n") and "<think>\n\n</think>" in hidden,
            evidence=repr(tail[-40:]),
        ))

    print("\n── границы замера ──")
    print("  инфо  answer_many делает два хода: выбор навыка и ответ с текстом навыка.")
    print("        Автопроверки ловят механические пункты рубрик; содержательные — судья.")

    print(f"\nпройдено {sum(results)} из {len(results)}")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
