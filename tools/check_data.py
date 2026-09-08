"""Проверка наборов данных. Ни модели, ни GPU не нужно — секунды.

    python tools/check_data.py

Что проверяется:
  * структура строк: обязательные поля, известный навык, известный документ;
  * эталонные ответы проходят назначенные им автопроверки — иначе метрика
    недостижима по построению;
  * плохие ответы автопроверки НЕ проходят; те немногие, что проходят,
    печатаются: их ловит только судья, и полезно знать, сколько их;
  * эталон не начинается с отказа там, где отказ не требуется, — иначе
    ложные отказы завышены уже на разметке;
  * документы трейна и теста не пересекаются;
  * поля `route` и `split` заполнены известными значениями: `route`
    разделяет ответственность фильтра запросов и агента, `split` —
    обучение и свою отложенную выборку.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vlmkit import rubric, skills  # noqa: E402

DATA = ROOT / "data"
REQUIRED = {"id", "split", "category", "skill", "prompt", "document", "checks", "answer", "rejected"}


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    documents = json.loads((DATA / "documents.json").read_text(encoding="utf-8"))
    known_skills = set(skills.names())
    files = [DATA / "golden.jsonl"] + sorted((DATA / "train").glob("*.jsonl"))

    problems: list[str] = []
    seen: set[str] = set()
    judge_only: list[str] = []
    rows_by_file: dict[str, list[dict]] = {}

    for path in files:
        rows = load(path)
        rows_by_file[path.stem] = rows
        bad_ideal, easy_rejected = [], []
        for row in rows:
            where = f"{path.name}:{row.get('id', '?')}"
            if missing := REQUIRED - set(row):
                problems.append(f"{where}: нет полей {sorted(missing)}")
            if row["id"] in seen:
                problems.append(f"{where}: повтор id")
            seen.add(row["id"])
            if row["skill"] not in known_skills:
                problems.append(f"{where}: неизвестный навык {row['skill']}")
            if row.get("route") not in ("agent", "guard"):
                problems.append(f"{where}: route = {row.get('route')!r}")
            if row["split"] not in ("train", "dev", "test"):
                problems.append(f"{where}: split = {row['split']!r}")
            if row["document"] and row["document"] not in documents:
                problems.append(f"{where}: неизвестный документ {row['document']}")
            for turn in row.get("history", []):
                if set(turn) != {"user", "skill", "assistant"} or turn["skill"] not in known_skills:
                    problems.append(f"{where}: испорченный ход истории")

            case = {**row, "document": documents.get(row["document"], "")}
            checks = rubric.run_checks(row["answer"], case)
            if not all(checks.values()):
                bad_ideal.append(f"{row['id']} ({', '.join(k for k, v in checks.items() if not v)})")
            if all(rubric.run_checks(row["rejected"], case).values()):
                easy_rejected.append(row["id"])
            if "refuses" not in row["checks"] and rubric.refuses_opening(row["answer"], case):
                problems.append(f"{where}: эталон начинается с отказа, но отказ не размечен")

        judge_only += easy_rejected
        problems += [f"{path.name}: эталон не проходит свои проверки — {x}" for x in bad_ideal]
        guard = sum(r.get("route") == "guard" for r in rows)
        dev = sum(r["split"] == "dev" for r in rows)
        print(f"{path.stem:26} {len(rows):4}  guard {guard:3}  dev {dev:3}  "
              f"многоходовых {sum(bool(r.get('history')) for r in rows):3}")

    train = [r for name, rows in rows_by_file.items() if name != "golden" for r in rows]
    test_docs = {r["document"] for r in rows_by_file["golden"] if r["document"]}
    overlap = sorted({r["document"] for r in train} & test_docs)
    if overlap:
        problems.append(f"документы трейна пересекаются с тестом: {overlap}")

    total = len(train) + len(rows_by_file["golden"])
    own_train = [r for r in train if r["split"] == "train"]
    own_dev = [r for r in train if r["split"] == "dev"]
    print(f"\nтест продукта {len(rows_by_file['golden'])}, обучение {len(own_train)}, "
          f"свой dev {len(own_dev)}, документов {len(documents)}")
    print("навыки в обучении:", dict(Counter(r["skill"] for r in own_train)))
    print("навыки в dev:     ", dict(Counter(r["skill"] for r in own_dev)))
    print(f"маршрут guard: {sum(r.get('route') == 'guard' for r in train)} из {len(train)}"
          f"; с пустым документом: {sum(not r['document'] for r in train)}"
          f"; заканчиваются вопросом: {sum(r['answer'].rstrip().endswith('?') for r in train)}")
    print(f"плохих ответов, которые проходят автопроверки (ловит только судья): {len(judge_only)} из {total}")
    if judge_only:
        print("  " + ", ".join(judge_only))

    print()
    if problems:
        for line in problems:
            print(" ПРОБЛЕМА", line)
        sys.exit(1)
    print("данные в порядке")


if __name__ == "__main__":
    main()
