"""Validate the rendered dataset without a model. Seconds, no torch, no GPU.

    python tools/check_data.py

For every split: required fields, alternating roles ending with the student,
reference answers pass the checks assigned to them, bad answers mostly fail
them, no reference answer opens with a refusal unless it should, and no
document is shared between training and either test set.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import metrics  # noqa: E402

DATA = ROOT / "data"
SPLITS = ("train", "dev", "test_product", "test_extended")
REQUIRED = {"id", "category", "prompt", "chosen", "rejected", "checks", "must_include",
            "must_not_include", "must_refuse", "rubric", "document"}


def load(name: str) -> list[dict]:
    return [json.loads(line) for line in (DATA / f"{name}.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    documents = json.loads((DATA / "documents.json").read_text(encoding="utf-8"))
    problems: list[str] = []
    judge_only: list[str] = []
    seen: set[str] = set()
    docs_by_split: dict[str, set[str]] = {}

    for name in SPLITS:
        rows = load(name)
        docs_by_split[name] = {r["document"] for r in rows if r["document"]}
        for row in rows:
            where = f"{name}:{row.get('id', '?')}"
            if missing := REQUIRED - set(row):
                problems.append(f"{where}: нет полей {sorted(missing)}")
                continue
            if row["id"] in seen:
                problems.append(f"{where}: повтор id")
            seen.add(row["id"])
            roles = [m["role"] for m in row["prompt"]]
            if roles[0] != "system" or roles[-1] != "user" or any(a == b for a, b in zip(roles[1:], roles[2:])):
                problems.append(f"{where}: роли идут не по очереди: {roles}")
            if row["document"] and row["document"] not in documents:
                problems.append(f"{where}: неизвестный документ {row['document']}")

            case = {**row, "document": documents.get(row["document"], "")}
            chosen, rejected = row["chosen"][0]["content"], row["rejected"][0]["content"]
            result = metrics.run(chosen, case)
            if not all(result.values()):
                problems.append(f"{where}: эталон не проходит {[k for k, v in result.items() if not v]}")
            if all(metrics.run(rejected, case).values()):
                judge_only.append(row["id"])
            if not row["must_refuse"] and "refuses" not in row["checks"] and metrics.refuses_opening(chosen, case):
                problems.append(f"{where}: эталон начинается с отказа, но отказ не размечен")

        print(f"{name:14} {len(rows):4}   отказ обязателен {sum(r['must_refuse'] for r in rows):3}"
              f"   многоходовых {sum(len(r['prompt']) > 2 for r in rows):3}"
              f"   без документа {sum(not r['document'] for r in rows):3}"
              f"   категорий {len(Counter(r['category'] for r in rows))}")

    training = docs_by_split["train"] | docs_by_split["dev"]
    for test in ("test_product", "test_extended"):
        if overlap := sorted(training & docs_by_split[test]):
            problems.append(f"документы обучения встречаются в {test}: {overlap}")

    total = sum(len(load(n)) for n in SPLITS)
    print(f"\nдокументов {len(documents)}: в обучении {len(training)}, в тестах {len(docs_by_split['test_product'] | docs_by_split['test_extended'])}")
    print(f"плохих ответов, проходящих все автопроверки: {len(judge_only)} из {total}, их ловит только судья")

    print()
    if problems:
        for line in problems:
            print(" ПРОБЛЕМА", line)
        sys.exit(1)
    print("данные в порядке")


if __name__ == "__main__":
    main()
