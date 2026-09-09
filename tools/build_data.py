"""Render the readable situations in data/raw into the files trainers read.

    python tools/build_data.py

A raw row is one situation as a person would write it: the document id,
earlier turns, the student's request, a reference answer and a bad one.
The rendered row is the conversational preference format of TRL: `prompt`
is the full message list with the system prompt and the document fragment
in place, `chosen` and `rejected` are single assistant messages. Every
rendered row is self-contained, so a training file can be read without
this repository.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import DATA, DOCUMENTS, SPLITS, prompt_for  # noqa: E402

RAW = DATA / "raw"


def render(row: dict) -> dict:
    """One raw situation to one rendered row."""
    document = DOCUMENTS.get(row.get("document") or "", "")
    prompt = prompt_for(row["prompt"], document, row.get("dialog"))

    out = {
        "id": row["id"],
        "category": row["category"],
        "scenario": row.get("scenario", ""),
        "prompt": prompt,
        "chosen": [{"role": "assistant", "content": row["chosen"]}],
        "rejected": [{"role": "assistant", "content": row["rejected"]}],
        "checks": row.get("checks") or [],
        "must_include": row.get("must_include") or [],
        "must_not_include": row.get("must_not_include") or [],
        "must_refuse": bool(row.get("must_refuse")),
        "rubric": row.get("rubric") or [],
        "document": row.get("document") or "",
    }
    if "reference" in row:
        out["reference"] = row["reference"]
    return out


def main() -> None:
    for split in SPLITS:
        source = RAW / f"{split}.jsonl"
        if not source.exists():
            print(f"{split:14} нет data/raw/{split}.jsonl, пропущен")
            continue
        rows = [render(json.loads(line)) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
        target = DATA / f"{split}.jsonl"
        target.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
        turns = sum(len(r["prompt"]) > 2 for r in rows)
        print(f"{split:14} {len(rows):4} ситуаций, многоходовых {turns:3}, отказ обязателен {sum(r['must_refuse'] for r in rows):3}")


if __name__ == "__main__":
    main()
