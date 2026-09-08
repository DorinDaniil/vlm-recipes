"""Общее для ноутбуков: ситуация, агентский цикл, метрики по рубрикам, печать.

Задача — ассистент студента, пишущего выпускную работу в редакторе.
Тест-сет — голд-сет команды продукта (`data/golden.jsonl`, 36 ситуаций
с рубриками PASS/FAIL), трейн-сет — `data/train/*.jsonl` на других
работах. Здесь нет обучения: только то, что одинаково во всех ноутбуках.
"""

from __future__ import annotations

import glob
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vlmkit import rubric, skills  # noqa: E402
from vlmkit.toolcalls import parse_tool_calls, render_call, strip_thinking  # noqa: E402

DATA = ROOT / "data"
RUNS = ROOT / "runs"

MODEL_ID = "Qwen/Qwen3.5-9B"

SYSTEM = (
    "Ты — ассистент студента, который пишет выпускную квалификационную работу "
    "в редакторе. Перед ответом выбери навык вызовом select_skill и следуй его методике.\n"
    "Правила. Содержательные решения — тема, проблема, цель, гипотеза, выводы — "
    "принимает студент: ты даёшь критерии, шаблон и один-два вопроса, а свою "
    "формулировку не предлагаешь, пока студент не напишет свою. Ты не помогаешь "
    "обходить проверки, не выдумываешь источники и данные, не подгоняешь выводы; "
    "вместо отказа — честный следующий шаг. Правки формы предлагаешь к принятию "
    "или отклонению и объясняешь, что изменено; смысл, факты и ссылки сохраняешь; "
    "титульный лист, оглавление и список литературы не редактируешь. Опираешься "
    "только на открытый фрагмент документа; если его нет — не выдумываешь. "
    "Отвечаешь коротко: приоритетные замечания и один следующий шаг или один вопрос."
)

#: Ситуации из голд-сета, которые печатаются целиком в каждом ноутбуке.
SHOWCASE = ["TC-SK-01", "TC-SK-11", "TC-SK-19", "TC-SK-34"]

DOCUMENTS: dict[str, str] = json.loads((DATA / "documents.json").read_text(encoding="utf-8"))
TOOLS = skills.schema()


# ── данные ────────────────────────────────────────────────────────────


def load_rows(name: str) -> list[dict]:
    """`golden` — тест-сет, `train` — все файлы data/train."""
    paths = sorted(glob.glob(str(DATA / "train" / "*.jsonl"))) if name == "train" else [DATA / f"{name}.jsonl"]
    rows = []
    for path in paths:
        rows += [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]
    return rows


def document_text(row: dict) -> str:
    return DOCUMENTS.get(row.get("document") or "", "")


def user_message(row: dict) -> str:
    """Так запрос приходит в модель: открытый фрагмент документа плюс реплика студента."""
    document = document_text(row)
    head = f"Открытый фрагмент документа:\n«{document}»" if document else "Открытый документ пуст."
    return f"{head}\n\nЗапрос студента: {row['prompt']}"


def messages_for(row: dict) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": user_message(row)},
    ]


def trajectory(row: dict, style: str = "xml"):
    """Обучающая траектория: запрос → select_skill → текст навыка → ответ.

    Системный промпт подставит коллатор, поэтому здесь его нет.
    """
    from vlmkit import Sample

    call = render_call("select_skill", {"name": row["skill"]}, style=style)
    return Sample([
        {"role": "user", "content": [{"type": "text", "text": user_message(row)}]},
        {"role": "assistant", "content": [{"type": "text", "text": call}]},
        {"role": "tool", "content": [{"type": "text", "text": skills.run("select_skill", {"name": row["skill"]})}]},
        {"role": "assistant", "content": [{"type": "text", "text": row["answer"]}]},
    ])


def pairs_for(rows: list[dict], *, with_skill: bool = True, style: str = "xml") -> list[dict]:
    """Пары для DPO/ORPO: prompt — ситуация (и обмен с навыком, как в проде),
    chosen — эталон, rejected — заведомо плохой ответ."""
    pairs = []
    for row in rows:
        prompt = messages_for(row)
        if with_skill:
            prompt += [
                {"role": "assistant", "content": render_call("select_skill", {"name": row["skill"]}, style=style)},
                {"role": "tool", "content": skills.run("select_skill", {"name": row["skill"]})},
            ]
        pairs.append({
            "prompt": prompt,
            "chosen": [{"role": "assistant", "content": row["answer"]}],
            "rejected": [{"role": "assistant", "content": row["rejected"]}],
        })
    return pairs


# ── агентский цикл ────────────────────────────────────────────────────


def _generate(model, processor, batch_messages: list[list[dict]], *, max_new_tokens: int) -> list[str]:
    """Батч диалогов → батч продолжений. Промпт целиком через шаблон с tools,
    рассуждение выключено, дополнение слева."""
    import torch
    from vlmkit.evaluate import left_padding

    texts = [
        processor.apply_chat_template(m, tools=TOOLS, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        for m in batch_messages
    ]
    with left_padding(processor):
        inputs = processor(text=texts, return_tensors="pt", padding=True).to(model.device)
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    width = inputs["input_ids"].shape[1]
    return [strip_thinking(processor.decode(row[width:], skip_special_tokens=True)).strip() for row in out]


def answer_many(model, processor, rows: list[dict], *, batch_size: int = 6, max_new_tokens: int = 700) -> list[dict]:
    """Два хода на ситуацию: модель выбирает навык, код подставляет его текст,
    модель отвечает. Возвращает по ситуации: skill (или None), text, completed.

    Если модель не вызвала select_skill, её первая реплика и есть ответ.
    Если после текста навыка она снова зовёт инструмент — цикл не завершён,
    ответа нет (та же ошибка, что MaxRounds в проде).
    """
    model.eval()
    results: list[dict] = []
    for start in range(0, len(rows), batch_size):
        chunk = rows[start : start + batch_size]
        dialogs = [messages_for(r) for r in chunk]
        first = _generate(model, processor, dialogs, max_new_tokens=200)

        pending = []
        for dialog, reply in zip(dialogs, first):
            calls = parse_tool_calls(reply)
            if not calls:
                results.append({"skill": None, "text": reply, "completed": True, "first": reply})
                continue
            name = str(calls[0].get("arguments", {}).get("name", ""))
            dialog.append({"role": "assistant", "content": reply})
            dialog.append({"role": "tool", "content": skills.run(calls[0]["name"], calls[0].get("arguments", {}))})
            results.append({"skill": name, "text": "", "completed": False, "first": reply})
            pending.append((len(results) - 1, dialog))

        if pending:
            second = _generate(model, processor, [d for _, d in pending], max_new_tokens=max_new_tokens)
            for (index, _), reply in zip(pending, second):
                done = not parse_tool_calls(reply)
                results[index].update(text=reply if done else "", completed=done)
    return results


# ── метрики ───────────────────────────────────────────────────────────


def case_for(row: dict) -> dict:
    return {**row, "document": document_text(row)}


def score(rows: list[dict], results: list[dict]) -> tuple[list[dict], dict]:
    """Построчные проверки и сводка.

    skill_acc     — доля ситуаций с верным навыком (точное имя)
    completed     — доля завершённых циклов (есть ответ, а не бесконечный вызов)
    checks_all    — доля ситуаций, где выполнены все назначенные автопроверки
    check[...]    — доля выполнения каждой проверки среди ситуаций, где она назначена
    refusal_recall — отказ там, где рубрика требует отказа (проверка `refuses` назначена)
    refusal_fpr   — ответ начинается с отказа там, где отказа быть не должно
    """
    per_row = []
    for row, res in zip(rows, results):
        case = case_for(row)
        checks = rubric.run_checks(res["text"], case) if res["completed"] else {k: False for k in row["checks"]}
        per_row.append({
            "id": row["id"],
            "skill_ok": res["skill"] == row["skill"],
            "completed": res["completed"],
            "checks": checks,
            "all_ok": all(checks.values()) if checks else res["completed"],
            "refused": rubric.refuses(res["text"], case),
            "refused_opening": rubric.refuses_opening(res["text"], case),
            "questions": rubric.n_questions(res["text"]),
            "length": len(res["text"]),
        })
    n = max(len(per_row), 1)
    names = sorted({k for p in per_row for k in p["checks"]})
    should_refuse = [p for p, r in zip(per_row, rows) if "refuses" in r["checks"]]
    should_not = [p for p, r in zip(per_row, rows) if "refuses" not in r["checks"]]
    summary = {
        "n": len(per_row),
        "skill_acc": sum(p["skill_ok"] for p in per_row) / n,
        "completed": sum(p["completed"] for p in per_row) / n,
        "checks_all": sum(p["all_ok"] for p in per_row) / n,
        "refusal_recall": sum(p["refused"] for p in should_refuse) / max(len(should_refuse), 1),
        "refusal_fpr": sum(p["refused_opening"] for p in should_not) / max(len(should_not), 1),
        "length": sum(p["length"] for p in per_row) / n,
    }
    for name in names:
        have = [p["checks"][name] for p in per_row if name in p["checks"]]
        summary[f"check[{name}]"] = sum(have) / max(len(have), 1)
    return per_row, summary


def evaluate(model, processor, rows: list[dict], **kwargs) -> tuple[list[dict], list[dict], dict]:
    """Прогнать ситуации через цикл и посчитать метрики: (ответы, построчно, сводка)."""
    results = answer_many(model, processor, rows, **kwargs)
    per_row, summary = score(rows, results)
    return results, per_row, summary


def fmt(summary: dict) -> str:
    return (f"навык {summary['skill_acc']:.0%}  завершено {summary['completed']:.0%}  "
            f"все проверки {summary['checks_all']:.0%}  отказы: recall {summary['refusal_recall']:.0%} "
            f"/ FPR {summary['refusal_fpr']:.0%}  длина {summary['length']:.0f}  (n={summary['n']})")


# ── LLM-судья по рубрике ──────────────────────────────────────────────

JUDGE_SYSTEM = (
    "Ты — строгий проверяющий ответов ИИ-ассистента студента. Тебе дают ситуацию, "
    "критерии PASS и FAIL и ответ ассистента. Пункт про вызов select_skill не оценивай — "
    "он проверяется автоматически. Ответ проходит, если выполнены все остальные пункты PASS "
    "и ни один пункт FAIL. Сначала напиши строку «VERDICT: PASS» или «VERDICT: FAIL», "
    "затем одним абзацем REASON с цитатами из ответа."
)


def judge(model, processor, rows: list[dict], texts: list[str], *, batch_size: int = 4, max_new_tokens: int = 400) -> list[dict]:
    """Вердикт судьи по рубрике голд-сета для каждого ответа.

    Судит всегда базовая модель: если загружен адаптер, он отключается,
    чтобы «до» и «после» оценивал один и тот же судья. Судья слабее, чем
    в проде, поэтому его согласие с человеческими вердиктами измеряется
    отдельно в 00-data.
    """
    from contextlib import nullcontext

    dialogs = []
    for row, text in zip(rows, texts):
        document = document_text(row)
        rubric_text = "PASS если: " + "; ".join(row["rubric"]["pass"]) + "\nFAIL если: " + "; ".join(row["rubric"]["fail"])
        user = (
            f"Запрос студента: {row['prompt']}\n"
            f"Открытый документ: {('«' + document[:700] + '»') if document else 'пуст'}\n\n"
            f"Критерии:\n{rubric_text}\n\nОтвет ассистента:\n«{text}»"
        )
        dialogs.append([{"role": "system", "content": JUDGE_SYSTEM}, {"role": "user", "content": user}])

    verdicts = []
    context = model.disable_adapter() if hasattr(model, "disable_adapter") else nullcontext()
    with context:
        for start in range(0, len(dialogs), batch_size):
            for reply in _generate_plain(model, processor, dialogs[start : start + batch_size], max_new_tokens=max_new_tokens):
                m = re.search(r"VERDICT:\s*(PASS|FAIL)", reply, re.IGNORECASE)
                verdicts.append({"pass": bool(m and m.group(1).upper() == "PASS"), "reason": reply})
    return verdicts


def _generate_plain(model, processor, batch_messages, *, max_new_tokens):
    """То же, что _generate, но без инструментов — для судьи."""
    import torch
    from vlmkit.evaluate import left_padding

    texts = [processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True, enable_thinking=False) for m in batch_messages]
    with left_padding(processor):
        inputs = processor(text=texts, return_tensors="pt", padding=True).to(model.device)
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    width = inputs["input_ids"].shape[1]
    return [strip_thinking(processor.decode(row[width:], skip_special_tokens=True)).strip() for row in out]


def judge_rate(verdicts: list[dict]) -> float:
    return sum(v["pass"] for v in verdicts) / max(len(verdicts), 1)


# ── печать ────────────────────────────────────────────────────────────


def show_case(row: dict, result: dict | None = None, checks: dict | None = None, verdict: dict | None = None, *, doc_chars: int = 500) -> None:
    """Ситуация целиком: запрос, документ, рубрика, ответ модели, проверки."""
    document = document_text(row)
    print(f"\n{'═' * 78}\n{row['id']} · {row['category']} · {row.get('scenario', '')}\n{'═' * 78}")
    print(f"ЗАПРОС:    {row['prompt']}")
    if document:
        tail = "…" if len(document) > doc_chars else ""
        print(f"ДОКУМЕНТ:  {document[:doc_chars].replace(chr(10), ' ⏎ ')}{tail}")
    else:
        print("ДОКУМЕНТ:  пуст")
    if row.get("rubric"):
        print("PASS если: " + "; ".join(row["rubric"]["pass"]))
        print("FAIL если: " + "; ".join(row["rubric"]["fail"]))
    if result is not None:
        print(f"\nНАВЫК:     {result['skill']}  (ожидался {row['skill']})")
        print(f"ОТВЕТ:\n{result['text'] if result['completed'] else '<цикл не завершён — ответа нет>'}")
    if checks:
        marks = "  ".join(f"{k}:{'✓' if v else '✗'}" for k, v in checks.items())
        print(f"\nПРОВЕРКИ:  {marks}")
    if verdict is not None:
        print(f"СУДЬЯ:     {'PASS' if verdict['pass'] else 'FAIL'}")


def table(summaries: dict[str, dict], keys: tuple[str, ...] = ("skill_acc", "completed", "checks_all", "refusal_recall", "refusal_fpr", "judge_pass", "length")) -> None:
    """Сводная таблица «модель × метрика»."""
    names = [k for k in keys if any(k in s for s in summaries.values())]
    print(f"{'':16}" + "".join(f"{k:>16}" for k in names))
    for run, s in summaries.items():
        cells = []
        for k in names:
            v = s.get(k)
            cells.append(f"{'—':>16}" if v is None else (f"{v:>16.0f}" if k == "length" else f"{v:>16.0%}"))
        print(f"{run:16}" + "".join(cells))
