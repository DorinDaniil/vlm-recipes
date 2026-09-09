"""Run files, comparison tables and charts.

Every experiment notebook ends with `save_run(...)`. The results notebook
reads those files and nothing else, so it needs neither a GPU nor the model
that produced them.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

RUNS = Path(__file__).resolve().parents[1] / "runs"

#: Metrics shown on the headline chart, with the direction that is better.
HEADLINE = [
    ("judge", "судья", "up"),
    ("checks_all", "все проверки", "up"),
    ("checks", "проверки в среднем", "up"),
    ("refusal", "отказ где нужен", "up"),
    ("false_refusal", "ложные отказы", "down"),
]


def save_run(name: str, result: dict, cases: list[dict], answers: list[str], *, note: str = "") -> Path:
    """Write `runs/<name>.json`: metrics, a note, and every answer by row id."""
    RUNS.mkdir(exist_ok=True)
    path = RUNS / f"{name}.json"
    path.write_text(json.dumps({
        "name": name,
        "note": note,
        "metrics": result["metrics"],
        "rows": result["rows"],
        "answers": {case["id"]: answer for case, answer in zip(cases, answers)},
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    return path


def load_runs(names: list[str] | None = None) -> dict[str, dict[str, Any]]:
    """Read run files back as {name: run}. Without names, read all of them."""
    paths = [RUNS / f"{n}.json" for n in names] if names else sorted(RUNS.glob("*.json"))
    return {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in paths if p.exists()}


def _metrics(run: dict) -> dict:
    return run.get("metrics", run)


def table(runs: dict[str, dict], keys: list[str] | None = None) -> str:
    """Aligned text table, one row per run, one column per metric."""
    keys = keys or [k for k, _, _ in HEADLINE] + ["length"]
    width = max((len(n) for n in runs), default=6) + 2
    head = "прогон".ljust(width) + "".join(k.rjust(15) for k in keys)
    lines = [head, "─" * len(head)]
    for name, run in runs.items():
        m = _metrics(run)
        cells = []
        for key in keys:
            value = m.get(key)
            if value is None:
                cells.append("—".rjust(15))
            elif key == "length":
                cells.append(f"{value:.0f}".rjust(15))
            else:
                cells.append(f"{value:.0%}".rjust(15))
        lines.append(name.ljust(width) + "".join(cells))
    return "\n".join(lines)


def deltas(base: dict, other: dict, keys: list[str] | None = None) -> str:
    """Change against a baseline in percentage points, marked good or bad."""
    keys = keys or [k for k, _, _ in HEADLINE]
    better = {k: d for k, _, d in HEADLINE}
    b, o = _metrics(base), _metrics(other)
    lines = []
    for key in keys:
        if b.get(key) is None or o.get(key) is None:
            continue
        delta = (o[key] - b[key]) * 100
        good = delta > 0 if better.get(key) == "up" else delta < 0
        lines.append(f"{key:15} {b[key]:>5.0%} → {o[key]:>5.0%}   {delta:+5.1f} п.п.  {'лучше' if good else 'хуже'}")
    return "\n".join(lines)


def bars(runs: dict[str, dict], keys: list[str] | None = None, title: str = "") -> Any:
    """Grouped bars: metrics along x, one bar per run, Wilson whiskers where known."""
    import matplotlib.pyplot as plt

    keys = keys or [k for k, _, _ in HEADLINE]
    labels = {k: label for k, label, _ in HEADLINE}
    names = list(runs)
    step = 0.8 / max(len(names), 1)
    fig, ax = plt.subplots(figsize=(1.9 * len(keys) + 3, 4))
    for i, name in enumerate(names):
        m = _metrics(runs[name])
        xs = [x + i * step for x in range(len(keys))]
        values = [m.get(k) or 0.0 for k in keys]
        errors = []
        for k, v in zip(keys, values):
            lo, hi = m.get(f"{k}_ci", (v, v))
            errors.append((max(v - lo, 0), max(hi - v, 0)))
        ax.bar(xs, values, width=step * 0.9, label=name,
               yerr=list(zip(*errors)) if any(e != (0, 0) for e in errors) else None,
               capsize=2, error_kw={"lw": 0.8})
    ax.set_xticks([x + 0.4 - step / 2 for x in range(len(keys))])
    ax.set_xticklabels([labels.get(k, k) for k in keys], rotation=15, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("доля")
    ax.set_title(title)
    ax.legend(frameon=False, ncols=min(len(names), 4), fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def category_table(runs: dict[str, dict]) -> str:
    """Mean share of passed checks per category, one column per run."""
    names = list(runs)
    categories = sorted({c for run in runs.values() for c in _metrics(run).get("per_category", {})})
    width = max((len(c) for c in categories), default=10) + 2
    head = "категория".ljust(width) + "".join(n.rjust(14) for n in names)
    lines = [head, "─" * len(head)]
    for category in categories:
        cells = [f"{_metrics(runs[n]).get('per_category', {}).get(category, float('nan')):.0%}".rjust(14) for n in names]
        lines.append(category.ljust(width) + "".join(cells))
    return "\n".join(lines)
