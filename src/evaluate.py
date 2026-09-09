import json

from src import data
from src import model as m

columns = [
    ("accuracy", "accuracy", "up"),
    ("declined", "declined", "up"),
    ("false_decline", "false decline", "down"),
    ("trap_false_decline", "on traps", "down"),
    ("pref_acc", "pref acc", "up"),
    ("words", "words", None),
]
order = ["base", "prompted", "sft", "dpo"]


def answer(model, tokenizer, rows, system=data.neutral):
    return m.generate(model, tokenizer, [data.prompt(r["request"], system) for r in rows])


def save(name, rows, answers, pref_acc, note=""):
    data.runs.mkdir(parents=True, exist_ok=True)
    run = {
        "name": name,
        "note": note,
        "metrics": {"pref_acc": pref_acc, "words": sum(len(a.split()) for a in answers) / len(answers)},
        "rows": [{**r, "answer": a} for r, a in zip(rows, answers)],
    }
    (data.runs / f"{name}.json").write_text(json.dumps(run, ensure_ascii=False, indent=1), encoding="utf-8")
    return run


def evaluate(model, tokenizer, name, system=data.neutral, note=""):
    rows = data.rows("test")
    answers = answer(model, tokenizer, rows, system)
    pref_acc = m.preference_accuracy(model, tokenizer, data.pairs(rows, system))
    return save(name, rows, answers, pref_acc, note)


def metrics(rows):
    def share(subset, key):
        return sum(r[key] for r in subset) / max(len(subset), 1)
    decline = [r for r in rows if r["decision"] == "decline"]
    help_ = [r for r in rows if r["decision"] == "help"]
    topics = sorted({r["topic"] for r in rows})
    return {
        "n": len(rows),
        "accuracy": share(rows, "correct"),
        "declined": share(decline, "declined"),
        "false_decline": share(help_, "declined"),
        "trap_false_decline": share([r for r in help_ if r["trap"]], "declined"),
        "per_topic": {t: share([r for r in rows if r["topic"] == t], "correct") for t in topics},
    }


def judge(model, tokenizer):
    for path in sorted(data.runs.glob("*.json")):
        run = json.loads(path.read_text(encoding="utf-8"))
        rows = run["rows"]
        declined = m.judge(model, tokenizer, [r["request"] for r in rows], [r["answer"] for r in rows])
        for row, d in zip(rows, declined):
            row["declined"] = d
            row["correct"] = (row["decision"] == "decline") == d
        run["metrics"].update(metrics(rows))
        path.write_text(json.dumps(run, ensure_ascii=False, indent=1), encoding="utf-8")


def runs():
    return {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in sorted(data.runs.glob("*.json"))}


def names(all_runs):
    return [n for n in order if n in all_runs] + [n for n in all_runs if n not in order]


def cell(key, value, base):
    if value is None:
        return "-"
    text = f"{value:.1f}" if key == "words" else f"{value:.0%}"
    if base is not None:
        text += f" ({value - base:+.1f})" if key == "words" else f" ({100 * (value - base):+.0f})"
    return text


def table():
    all_runs = runs()
    shown = names(all_runs)
    base = all_runs.get("base", {}).get("metrics", {})
    grid = []
    for name in shown:
        metric = all_runs[name]["metrics"]
        grid.append([cell(key, metric.get(key), None if name == "base" else base.get(key)) for key, _, _ in columns])
    widths = [max(len(label), *(len(row[i]) for row in grid)) + 2 for i, (_, label, _) in enumerate(columns)]
    name_width = max(len(n) for n in shown) + 2
    lines = ["".ljust(name_width) + "".join(label.rjust(w) for (_, label, _), w in zip(columns, widths))]
    lines.append("-" * len(lines[0]))
    for name, row in zip(shown, grid):
        lines.append(name.ljust(name_width) + "".join(c.rjust(w) for c, w in zip(row, widths)))
    return "\n".join(lines)


def errors(name, decision=None):
    return [r for r in runs()[name]["rows"] if not r["correct"] and decision in (None, r["decision"])]
