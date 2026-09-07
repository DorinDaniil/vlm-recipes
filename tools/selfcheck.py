"""Сверка кода с тем, что утверждают гайды.

Каждая проверка формулирует утверждение из PDF и смотрит, выполняется ли
оно на реальных данных и реальном процессоре. Ничего не обучает —
только собирает батч и разглядывает метки.

    python tools/selfcheck.py

Проверки намеренно грубые: они ловят расхождение теории с кодом, а не
тонкие ошибки. Тонкие ловятся замерами.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vlmkit import ChatCollator, LoadConfig, load_jsonl
from vlmkit.data import IGNORE_INDEX
from vlmkit.model import load_processor
from vlmkit.toolcalls import detect_style

DATA = Path(__file__).resolve().parents[1] / "data"


def split_by_mask(sample: Any, processor: Any, **collator_kwargs) -> tuple[str, str]:
    """Разделить пример на то, что попадает в градиент, и остальное.

    Это тот же механизм, что в `preview`, только результат отдаётся
    двумя строками, по которым удобно искать подстроки.
    """
    collator = ChatCollator(processor, **collator_kwargs)
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
    cfg = LoadConfig()
    print(f"модель: {cfg.model_id}\nвеса не грузим, нужен только процессор\n")
    processor = load_processor(cfg)

    template = processor.tokenizer.chat_template or ""
    tools = load_jsonl(DATA / "tools.jsonl")
    raw = [json.loads(l) for l in (DATA / "tools.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]

    multi = next(s for s, r in zip(tools, raw) if r["group"] == "multi")
    trained, hidden = split_by_mask(multi, processor)

    # Что лежит на диске — отдельно от того, что пережило коллацию.
    # Без этого разделения провал ниже нельзя отнести ни к данным,
    # ни к шаблону: непонятно, где искать.
    on_disk = json.dumps(raw[0], ensure_ascii=False) + json.dumps(
        next(r for r in raw if r["group"] == "multi"), ensure_ascii=False
    )
    disk_style = "xml" if "<function=" in on_disk else "json"

    results = []

    print("── данные на диске ──")
    print(f"  инфо  формат вызова в tools.jsonl: {disk_style}")
    if disk_style != detect_style(template):
        print("        не совпадает с шаблоном — пересоберите:")
        print("        python data/build_tools.py")
    print()

    # ── что утверждает 02-sft-math, раздел 1 ──────────────────────────
    print("── маскирование (02-sft-math §1) ──")

    results.append(check(
        "промпт пользователя скрыт от функции потерь",
        "Проверь, обоснована ли методика" in hidden
        and "Проверь, обоснована ли методика" not in trained,
    ))
    results.append(check(
        "результат вызова инструмента скрыт",
        "отбор выборки, обоснование объёма" in hidden
        and "отбор выборки, обоснование объёма" not in trained,
        "модель, обученная его предсказывать, начнёт сочинять содержимое",
    ))
    # Ищем не по формату вызова, а по нейтральному признаку: сам тег
    # <tool_call> одинаков в обоих форматах, и проверка не развалится
    # от того, что данные пересобрали в другом стиле.
    results.append(check(
        "реплики ассистента открыты для градиента",
        "<tool_call>" in trained,
        evidence=repr(trained[:200]),
    ))
    results.append(check(
        "открыты ВСЕ реплики ассистента, а не последняя",
        trained.count("<tool_call>") == 2,
        "в траектории multi их две",
        evidence=f"нашлось {trained.count('<tool_call>')}",
    ))

    share = len(trained) / max(len(trained) + len(hidden), 1)
    results.append(check(
        "доля обучаемого текста не ничтожна",
        share > 0.05,
        f"{share:.0%} — ниже 5% почти весь батч уходит впустую",
    ))

    # ── что утверждает 00-basics ──────────────────────────────────────
    print("\n── формат и шаблон (00-basics §4) ──")

    style = detect_style(template)
    results.append(check(
        "формат вызова в данных совпадает с ожиданиями модели",
        style == disk_style,
        f"шаблон ждёт {style}, на диске {disk_style}"
        + ("" if style == disk_style else " — пересоберите build_tools.py"),
    ))
    results.append(check(
        "формат вызова дожил до батча без изменений",
        (disk_style == "xml") == ("<function=" in trained),
        "если не дожил — шаблон переписывает содержимое реплик ассистента",
        evidence=repr(trained[:200]),
    ))
    results.append(check(
        "шаблон умеет вызов инструментов",
        "tools" in template,
        "иначе формат придётся разбирать самому",
    ))

    # ── рассуждение: расхождение, которое стоит увидеть ───────────────
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

        # Проверяем, что флаг действительно управляет поведением,
        # а не просто присутствует в сигнатуре.
        loose, _ = split_by_mask(multi, processor, mask_thinking=False)
        results.append(check(
            "mask_thinking=False возвращает блок в градиент",
            "<think>" in loose,
            "флаг должен работать в обе стороны",
        ))

        # Замер идёт с выключенным рассуждением. Хвост промпта при этом
        # должен совпадать с тем, что стоит перед ответом в обучении,
        # иначе модель на инференсе видит префикс, которого не видела.
        tail = processor.apply_chat_template(
            [{"role": "user", "content": "проверка"}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        results.append(check(
            "enable_thinking=False даёт тот же хвост, что перед ответом в обучении",
            tail.endswith("<think>\n\n</think>\n\n") and "<think>\n\n</think>" in hidden,
            "иначе инференс и обучение видят разные префиксы ответа",
            evidence=repr(tail[-40:]),
        ))

    # ── ограничение замера, о котором стоит знать ─────────────────────
    print("\n── границы замера ──")
    print("  инфо  evaluate.generate строит промпт до ПЕРВОЙ реплики ассистента")
    print("        значит меряется только первый шаг траектории,")
    print("        а не доведение её до конца")

    print(f"\nпройдено {sum(results)} из {len(results)}")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
