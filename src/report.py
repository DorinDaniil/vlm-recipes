"""One evaluation entry point and one place that prints results.

`evaluate` runs a model over both test sets and the dev sample, scores it,
and writes `runs/<name>-product.json` and `runs/<name>-extended.json`.
`show` reads whatever runs exist and prints one table per test, base first,
with changes against the base. Nothing else in the project prints metrics.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src import data, infer, metrics

RUNS = data.ROOT / "runs" / "assistant"
TESTS = {"product": "test_product", "extended": "test_extended"}

#: Columns of every table, in order: what they mean and which way is better.
COLUMNS = [
    ("judge",         "судья",           "up"),
    ("refusal_judge", "отказ по судье",  "up"),
    ("checks_all",    "все проверки",    "up"),
    ("false_refusal", "ложные отказы",   "down"),
    ("length",        "длина",           None),
    ("perplexity",    "ppl эталонов",    "down"),
    ("pref_acc",      "pref. accuracy",  "up"),
]


def evaluate(model, tokenizer, name: str, *, note: str = "", dev_size: int = 24, with_judge: bool = True) -> dict[str, dict]:
    """Score `model` on both tests and the dev sample, save the runs, return {test: result}."""
    dev = list(data.load("dev"))[:dev_size]
    ppl = infer.perplexity(model, tokenizer, dev)
    pref = infer.preference_accuracy(model, tokenizer, dev)
    results = {}
    for test, split in TESTS.items():
        rows = list(data.load(split))
        answers = infer.generate(model, tokenizer, [r["prompt"] for r in rows])
        verdicts = infer.judge(model, tokenizer, rows, answers) if with_judge else None
        result = metrics.score([data.case(r) for r in rows], answers, verdicts)
        result["metrics"].update(perplexity=ppl, pref_acc=pref)
        result["answers"] = answers
        save_run(f"{name}-{test}", result, rows, note=note)
        results[test] = result
    return results


def save_run(name: str, result: dict, rows: list[dict], *, note: str = "") -> Path:
    RUNS.mkdir(parents=True, exist_ok=True)
    path = RUNS / f"{name}.json"
    path.write_text(json.dumps({
        "name": name, "note": note, "metrics": result["metrics"], "rows": result["rows"],
        "answers": {row["id"]: a for row, a in zip(rows, result["answers"])},
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    return path


def load_runs(test: str | None = None) -> dict[str, dict[str, Any]]:
    """All saved runs, or only those of one test ("product" / "extended"), keyed by method name."""
    runs = {}
    for path in sorted(RUNS.glob("*.json")):
        method, _, suffix = path.stem.rpartition("-")
        if test is None or suffix == test:
            runs[path.stem if test is None else method] = json.loads(path.read_text(encoding="utf-8"))
    return runs


def _fmt(key: str, value: Any) -> str:
    if value is None:
        return "—"
    if key == "length":
        return f"{value:.0f}"
    if key == "perplexity":
        return f"{value:.2f}"
    return f"{value:.0%}"


def table(runs: dict[str, dict], base: str = "base") -> str:
    """One row per method, base first; other rows carry the change against the base in brackets."""
    names = ([base] if base in runs else []) + [n for n in runs if n != base]
    grid = []
    for name in names:
        m = runs[name]["metrics"]
        row = []
        for key, _, better in COLUMNS:
            cell = _fmt(key, m.get(key))
            if name != base and base in runs and better and m.get(key) is not None and runs[base]["metrics"].get(key) is not None:
                delta = m[key] - runs[base]["metrics"][key]
                cell += f" ({delta:+.1f})" if key == "perplexity" else f" ({delta * 100:+.0f})"
            row.append(cell)
        grid.append(row)
    widths = [max([len(label)] + [len(row[i]) for row in grid]) + 2 for i, (_, label, _) in enumerate(COLUMNS)]
    name_w = max(len(n) for n in names) + 2
    head = "".ljust(name_w) + "".join(label.rjust(w) for (_, label, _), w in zip(COLUMNS, widths))
    lines = [head, "─" * len(head)]
    for name, row in zip(names, grid):
        lines.append(name.ljust(name_w) + "".join(cell.rjust(w) for cell, w in zip(row, widths)))
    return "\n".join(lines)


def show(base: str = "base") -> None:
    """Print the table for every test that has saved runs."""
    for test, title in (("extended", "РАСШИРЕННЫЙ ТЕСТ, 100 ситуаций"), ("product", "ТЕСТ ПРОДУКТА, 33 ситуации")):
        runs = load_runs(test)
        if runs:
            print(title)
            print(table(runs, base))
            print()


def regressions(before: str, after: str, test: str = "extended") -> list[str]:
    """Row ids where `before` passed the judge and `after` failed, or a required refusal was lost."""
    runs = load_runs(test)
    rows = {r["id"]: r for r in data.load(TESTS[test])}
    a = {r["id"]: r for r in runs[before]["rows"]}
    b = {r["id"]: r for r in runs[after]["rows"]}
    lost = []
    for i in a:
        if a[i].get("judge") and not b[i].get("judge"):
            lost.append(f"{i}: судья PASS → FAIL")
        if rows[i]["must_refuse"] and a[i]["refused"] and not b[i]["refused"]:
            lost.append(f"{i}: пропал отказ")
    return lost


def bars(test: str = "extended", base: str = "base"):
    """Grouped bars of the share metrics with Wilson whiskers where available."""
    import matplotlib.pyplot as plt

    runs = load_runs(test)
    keys = [k for k, _, better in COLUMNS if better and k not in ("perplexity",)]
    labels = {k: label for k, label, _ in COLUMNS}
    names = ([base] if base in runs else []) + [n for n in runs if n != base]
    step = 0.8 / max(len(names), 1)
    fig, ax = plt.subplots(figsize=(2 * len(keys) + 3, 4))
    for i, name in enumerate(names):
        m = runs[name]["metrics"]
        values = [m.get(k) or 0.0 for k in keys]
        err = [[max(v - m.get(f"{k}_ci", (v, v))[0], 0) for k, v in zip(keys, values)],
               [max(m.get(f"{k}_ci", (v, v))[1] - v, 0) for k, v in zip(keys, values)]]
        ax.bar([x + i * step for x in range(len(keys))], values, width=step * 0.9, label=name, yerr=err, capsize=2)
    ax.set_xticks([x + 0.4 - step / 2 for x in range(len(keys))])
    ax.set_xticklabels([labels[k] for k in keys])
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, ncols=min(len(names), 4), fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig
