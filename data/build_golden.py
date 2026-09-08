#!/usr/bin/env python
"""Сборка тест-сета из голд-сета продукта.

Источник — таблица команды продукта `data/source/golden_set_feedback.xlsx`:
36 ситуаций, у каждой промт, открытый фрагмент документа, критерии судьи
(PASS/FAIL), ответ продового агента, вердикты судьи и человека. К ней
добавляются авторские эталонный и заведомо плохой ответы и назначенные
автопроверки из `data/golden_answers.json`.

Результат:
    data/golden.jsonl    — 36 строк, одна на ситуацию
    data/documents.json  — фрагменты документов: из таблицы + data/documents_new.json

    python data/build_golden.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
XLSX = HERE / "source" / "golden_set_feedback.xlsx"

#: Имена фрагментов из таблицы — по началу текста.
GOLDEN_DOCS = {
    "Введение\n\nАктуальность исследования обусловлена": "urfo-intro",
    "Ну короче, в современной науке": "urfo-colloquial-theory",
    "Глава 2. Анализ текущего состояния": "urfo-ch2",
    "МИНИСТЕРСТВО НАУКИ": "urfo-title",
    "Ну и вот, мы тут решили": "urfo-colloquial-methods",
}

COLUMNS = {
    "id": "ID",
    "category": "Категория",
    "scenario": "Сценарий (из таблицы)",
    "success": "Критерий успеха (из таблицы)",
    "prompt": "Промт",
    "fragment": "Фрагмент документа",
    "agent_answer": "Ответ агента",
    "skill_called": "Скилл вызван",
    "rubric": "Критерии судьи",
    "judge": "Judge",
    "judge_comment": "Judge комментарий",
    "human_feedback": "Обратная связь",
    "human_verdict": "вердикт",
}


def read_rows() -> list[dict[str, str]]:
    try:
        import openpyxl
    except ImportError:
        sys.exit("нужен openpyxl: pip install openpyxl")
    ws = openpyxl.load_workbook(XLSX, data_only=True)["Results"]
    rows = list(ws.iter_rows(values_only=True))
    header = [str(h).strip() if h else "" for h in rows[0]]
    out = []
    for row in rows[1:]:
        if not any(v is not None for v in row):
            continue
        record = {h: (str(v).strip() if v is not None else "") for h, v in zip(header, row)}
        out.append({key: record.get(col, "") for key, col in COLUMNS.items()})
    return out


def parse_rubric(text: str) -> dict[str, list[str]]:
    """«PASS если: [1] … [2] …» и «FAIL если: …» → списки пунктов."""
    result: dict[str, list[str]] = {"pass": [], "fail": []}
    for kind, key in (("PASS", "pass"), ("FAIL", "fail")):
        m = re.search(rf"{kind} если:(.*?)(?=(?:PASS|FAIL) если:|$)", text, re.S)
        if m:
            result[key] = [p.strip(" .,;") for p in re.split(r"\[\d+\]", m.group(1)) if p.strip(" .,;")]
    return result


def doc_id(fragment: str) -> str:
    for prefix, name in GOLDEN_DOCS.items():
        if fragment.startswith(prefix):
            return name
    if fragment:
        raise SystemExit(f"неизвестный фрагмент документа: {fragment[:60]!r}")
    return ""


def main() -> None:
    rows = read_rows()
    answers = json.loads((HERE / "golden_answers.json").read_text(encoding="utf-8"))

    documents = {}
    for row in rows:
        if row["fragment"]:
            documents[doc_id(row["fragment"])] = row["fragment"]
    documents.update(json.loads((HERE / "documents_new.json").read_text(encoding="utf-8")))
    (HERE / "documents.json").write_text(
        json.dumps(documents, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )

    with (HERE / "golden.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            extra = answers[row["id"]]
            record = {
                "id": row["id"],
                "split": "test",
                "category": row["category"],
                "skill": row["category"],
                "scenario": row["scenario"],
                "success": row["success"],
                "prompt": row["prompt"],
                "document": doc_id(row["fragment"]),
                "rubric": parse_rubric(row["rubric"]),
                "checks": extra["checks"],
                "must_include": extra.get("must_include", []),
                "must_not_include": extra.get("must_not_include", []),
                "answer": extra["answer"],
                "rejected": extra["rejected"],
                "production": {
                    "answer": row["agent_answer"],
                    "skill_called": row["skill_called"] == "Да",
                    "judge": row["judge"],
                    "judge_comment": row["judge_comment"],
                    "human_verdict": row["human_verdict"],
                    "human_feedback": row["human_feedback"],
                },
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    cats = {}
    for row in rows:
        cats[row["category"]] = cats.get(row["category"], 0) + 1
    print(f"golden.jsonl: {len(rows)} ситуаций, {cats}")
    print(f"documents.json: {len(documents)} фрагментов")


if __name__ == "__main__":
    main()
