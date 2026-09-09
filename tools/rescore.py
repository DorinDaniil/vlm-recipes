"""Recompute metrics for saved runs from their stored answers. No model needed.

    python tools/rescore.py                 # every runs/*.json
    python tools/rescore.py sft-extended    # named runs only

A run file keeps every answer and the judge's verdict per row, so when a
check or a rubric changes the numbers can be rebuilt in seconds instead of
regenerating anything. Answers and verdicts are left untouched; metrics and
per-row check results are replaced.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import data, metrics, report  # noqa: E402

SPLIT_OF = {"product": "test_product", "extended": "test_extended", "dev": "dev"}


def rescore(path: Path, rows_by_split: dict[str, dict[str, dict]]) -> str:
    run = json.loads(path.read_text(encoding="utf-8"))
    split = SPLIT_OF.get(path.stem.rsplit("-", 1)[-1])
    if split is None:
        return f"{path.name}: не понял, какой это тест, пропущен"
    rows = rows_by_split[split]
    ids = [r["id"] for r in run["rows"]] or list(run["answers"])
    cases = [data.case(rows[i]) for i in ids]
    answers = [run["answers"][i] for i in ids]
    verdicts = [r.get("judge") for r in run["rows"]] if run["rows"] and "judge" in run["rows"][0] else None
    before = run["metrics"]
    result = metrics.score(cases, answers, verdicts)
    for key in ("perplexity", "pref_acc"):        # measured with the model, cannot be recomputed here
        if key in before:
            result["metrics"][key] = before[key]
    run["metrics"], run["rows"] = result["metrics"], result["rows"]
    path.write_text(json.dumps(run, ensure_ascii=False, indent=1), encoding="utf-8")
    changed = [f"{k} {before[k]:.0%}→{result['metrics'][k]:.0%}" for k in ("checks_all", "refusal", "false_refusal")
               if k in before and abs(before[k] - result["metrics"][k]) >= 0.005]
    return f"{path.stem:22} " + (", ".join(changed) if changed else "без изменений")


def main() -> None:
    names = sys.argv[1:]
    paths = [report.RUNS / f"{n}.json" for n in names] if names else sorted(report.RUNS.glob("*.json"))
    rows_by_split = {s: {r["id"]: r for r in data.load(s)} for s in SPLIT_OF.values()}
    for path in paths:
        if path.exists():
            print(rescore(path, rows_by_split))
        else:
            print(f"{path.name}: нет такого файла")


if __name__ == "__main__":
    main()
