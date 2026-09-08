"""Насколько автопроверки различают хороший ответ от плохого.

    python tools/audit_checks.py

Проверка полезна, если она проходит на эталоне и не проходит на плохом
ответе. Разница этих двух долей — разделяющая способность. Колонка
«чужой» показывает вырожденность: если проверка проходит и на ответе
из другой ситуации, она почти ничего не измеряет.

Отдельно печатается, что будет, если оценивать только автопроверками:
сколько плохих ответов пройдут все назначенные и останутся для судьи.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vlmkit import rubric  # noqa: E402

DATA = ROOT / "data"


def load_rows() -> list[dict]:
    rows = []
    for path in [DATA / "golden.jsonl"] + sorted((DATA / "train").glob("*.jsonl")):
        rows += [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    return rows


def main() -> None:
    documents = json.loads((DATA / "documents.json").read_text(encoding="utf-8"))
    rows = load_rows()
    rng = random.Random(0)

    print(f"ситуаций {len(rows)}, документов {len(documents)}\n")
    print(f"{'проверка':16} {'назнач':>6} {'эталон':>7} {'плохой':>7} {'чужой':>7} {'разделяет':>10}")
    for name in sorted(rubric.CHECKS):
        used = [r for r in rows if name in rubric.run_checks("", {**r, "document": ""})]
        if not used:
            print(f"{name:16} {0:>6}   не назначена ни одной ситуации")
            continue
        ideal = bad = alien = 0
        for row in used:
            case = {**row, "document": documents.get(row["document"], "")}
            ideal += rubric.CHECKS[name](row["answer"], case)
            bad += rubric.CHECKS[name](row["rejected"], case)
            other = rng.choice([x for x in rows if x["id"] != row["id"]])
            alien += rubric.CHECKS[name](other["answer"], case)
        n = len(used)
        print(f"{name:16} {n:>6} {ideal/n:>7.0%} {bad/n:>7.0%} {alien/n:>7.0%} {(ideal-bad)/n:>10.0%}")

    grounded_rows = [r for r in rows if "grounded" in r["checks"] and r["document"]]
    passed = 0
    for row in grounded_rows:
        others = [d for name, d in documents.items() if name != row["document"]]
        passed += rubric.grounded(row["answer"], {"document": rng.choice(others)})
    print(f"\ngrounded на случайно подставленном чужом документе: {passed/len(grounded_rows):.0%}"
          f" из {len(grounded_rows)} — чем ниже, тем содержательнее проверка")

    only_judge = []
    for row in rows:
        case = {**row, "document": documents.get(row["document"], "")}
        if all(rubric.run_checks(row["rejected"], case).values()):
            only_judge.append(row["id"])
    print(f"плохих ответов, проходящих все свои автопроверки: {len(only_judge)} из {len(rows)}"
          f" — их ловит только судья")
    if only_judge:
        print("  " + ", ".join(only_judge))


if __name__ == "__main__":
    main()
