"""Split and render the request-filter dataset.

    python tools/build_filter.py

Reads data/filter/raw.jsonl, assigns a stratified deterministic split
(test 100, dev 40, the rest train) and writes data/filter/{train,dev,test}.jsonl
in the conversational preference format: `prompt` with the system prompt and
the document fragment, `chosen` the right label line, `rejected` the wrong one.
"""

from __future__ import annotations

import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import filter as F  # noqa: E402

TEST, DEV = 100, 40


def rejected_for(row: dict, rng: random.Random) -> str:
    """The wrong label: PASS for a block, a plausible wrong block for a pass."""
    if row["label"] == "BLOCK":
        return "PASS"
    category = rng.choice(F.CATEGORIES)
    return F.answer_for("BLOCK", category)


def render(row: dict, split: str, rng: random.Random) -> dict:
    return {
        "id": row["id"], "split": split, "label": row["label"], "category": row["category"],
        "request": row["request"], "document": row["document"], "note": row["note"],
        "prompt": F.prompt_for(row["request"], row["document"]),
        "chosen": [{"role": "assistant", "content": F.answer_for(row["label"], row["category"])}],
        "rejected": [{"role": "assistant", "content": rejected_for(row, rng)}],
    }


def main() -> None:
    rng = random.Random(7)
    raw = [json.loads(l) for l in (F.DATA / "raw.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in raw:
        groups[r["category"] or ("trap" if r["note"] == "ловушка" else "pass")].append(r)

    split_of: dict[str, str] = {}
    for key, rows in groups.items():
        rows = rows[:]
        rng.shuffle(rows)
        share = len(rows) / len(raw)
        n_test, n_dev = round(TEST * share), round(DEV * share)
        for i, r in enumerate(rows):
            split_of[r["id"]] = "test" if i < n_test else "dev" if i < n_test + n_dev else "train"

    out = {s: [] for s in F.SPLITS}
    for r in raw:
        out[split_of[r["id"]]].append(render(r, split_of[r["id"]], rng))
    for split, rows in out.items():
        (F.DATA / f"{split}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
        labels = Counter(r["label"] for r in rows)
        traps = sum(r["note"] == "ловушка" for r in rows)
        docs = sum(bool(r["document"]) for r in rows)
        print(f"{split:6} {len(rows):4}  PASS {labels['PASS']:3}  BLOCK {labels['BLOCK']:3}  ловушек {traps:3}  с документом {docs:3}")

    # every reference answer must parse back to its own label
    for split, rows in out.items():
        for r in rows:
            label, category = F.parse(r["chosen"][0]["content"])
            assert label == r["label"] and (category or "") == r["category"], r["id"]
    print("эталоны разбираются в свои же метки")


if __name__ == "__main__":
    main()
