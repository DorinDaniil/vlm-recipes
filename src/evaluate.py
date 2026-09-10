import json
import math

from src import data
from src import model as m

columns = [
    ("pref_acc", "pref acc", "up"),
    ("ppl_good", "ppl good", "down"),
    ("ppl_bad", "ppl bad", "up"),
    ("tokens", "tokens", None),
]
order = ["base", "prompted", "sft", "dpo"]


def answer(model, tokenizer, rows, system=data.neutral):
    return m.generate(model, tokenizer, [data.prompt(r["request"], system) for r in rows])


def metrics(good, bad, answers, tokenizer):
    lengths = [len(tokenizer(a, add_special_tokens=False)["input_ids"]) for a in answers]
    return {
        "n": len(answers),
        "pref_acc": sum(g > b for g, b in zip(good, bad)) / len(good),
        "ppl_good": math.exp(-sum(good) / len(good)),
        "ppl_bad": math.exp(-sum(bad) / len(bad)),
        "tokens": sum(lengths) / len(lengths),
    }


def save(name, rows, answers, metrics, note=""):
    data.runs.mkdir(parents=True, exist_ok=True)
    run = {"name": name, "note": note, "metrics": metrics, "rows": [{**r, "answer": a} for r, a in zip(rows, answers)]}
    (data.runs / f"{name}.json").write_text(json.dumps(run, ensure_ascii=False, indent=1), encoding="utf-8")
    return run


def evaluate(model, tokenizer, name, system=data.neutral, note=""):
    rows = data.rows("test")
    answers = answer(model, tokenizer, rows, system)
    good, bad = m.logprobs(model, tokenizer, data.pairs(rows, system))
    return save(name, rows, answers, metrics(good, bad, answers, tokenizer), note)


def runs():
    return {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in sorted(data.runs.glob("*.json"))}


def names(all_runs):
    return [n for n in order if n in all_runs] + [n for n in all_runs if n not in order]


def cell(key, value, base):
    if value is None:
        return "-"
    text = f"{value:.0%}" if key == "pref_acc" else f"{value:.1f}"
    if base is not None:
        text += f" ({100 * (value - base):+.0f})" if key == "pref_acc" else f" ({value - base:+.1f})"
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


def transcript(name, decision=None, trap=None):
    lines = []
    for i, row in enumerate(runs()[name]["rows"]):
        if decision not in (None, row["decision"]) or trap not in (None, row["trap"]):
            continue
        tag = f"{row['decision']}{' · trap' if row['trap'] else ''} · {row['topic']}"
        lines.append(f"[{i}] {tag}\nЗАПРОС: {row['request']}\nОТВЕТ:  {row['answer']}\n")
    return "\n".join(lines)


def compare(names_, indices):
    all_runs = runs()
    lines = []
    for i in indices:
        row = all_runs[names_[0]]["rows"][i]
        lines.append(f"[{i}] {row['decision']}{' · trap' if row['trap'] else ''} · {row['topic']}\nЗАПРОС: {row['request']}")
        for name in names_:
            lines.append(f"  {name:9} {all_runs[name]['rows'][i]['answer']}")
        lines.append("")
    return "\n".join(lines)
