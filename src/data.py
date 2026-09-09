"""Dataset access and the few text constants every notebook needs.

Rows use the conversational preference format of TRL: `prompt` is a list of
chat messages ending with the student's turn, `chosen` and `rejected` are
single assistant messages. SFT reads `prompt` + `chosen`, preference
methods read all three, KTO unpairs them. Nothing has to be reshaped in a
notebook beyond a column rename.

Every row is self-contained: the system message and the document fragment
are already inside `prompt`. The `document` id is kept only so that the
grounding check can find the source text.
"""

from __future__ import annotations

import json
from pathlib import Path

from datasets import Dataset, load_dataset

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
SPLITS = ("train", "dev", "test_product", "test_extended")

#: Product rules as sent to the model. Changing this text means rebuilding
#: the data files, see `tools/build_data.py`.
SYSTEM = (
    "Ты ассистент студента, который пишет выпускную квалификационную работу "
    "в редакторе. Рядом с диалогом открыт фрагмент его документа.\n"
    "\n"
    "Правила.\n"
    "1. Содержательные решения принимает студент: тему, проблему, цель, задачи, "
    "гипотезу, объект, предмет и выводы за него не формулируй. Дай критерии, "
    "шаблон и один-два вопроса; свои варианты предлагай только после того, как "
    "студент напишет свой.\n"
    "2. Опирайся только на открытый фрагмент. Если фрагмента нет, так и скажи "
    "и не описывай его содержимое.\n"
    "3. Правки формы (стиль, сокращение, оформление) выдавай как предложение "
    "к принятию или отклонению и говори, что именно изменено. Смысл, числа "
    "и ссылки сохраняй. Титульный лист, оглавление и список литературы "
    "не редактируй, объясни почему.\n"
    "4. Если просьба явно нечестная — обойти проверку на заимствования, "
    "выдумать источники или данные, подогнать выводы, написать работу за "
    "студента — откажи одним абзацем и назови честный путь.\n"
    "5. Не соглашайся с ошибкой студента из вежливости: если он неверно "
    "трактует цифру или метод, скажи прямо и объясни коротко.\n"
    "\n"
    "Форма ответа: обычный текст без заголовков и жирного выделения, "
    "нумерованный список только для приоритетных замечаний и не длиннее пяти "
    "пунктов. Заканчивай одним следующим шагом или одним вопросом."
)

DOCUMENTS: dict[str, str] = json.loads((DATA / "documents.json").read_text(encoding="utf-8"))


def load(split: str) -> Dataset:
    """One split as a `datasets.Dataset`."""
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}, expected one of {SPLITS}")
    return load_dataset("json", data_files=str(DATA / f"{split}.jsonl"), split="train")


def to_sft(ds: Dataset) -> Dataset:
    """Prompt-completion view for SFTTrainer: loss on the completion only."""
    return ds.rename_column("chosen", "completion").remove_columns(["rejected"])


def to_kto(ds: Dataset) -> Dataset:
    """Unpaired view for KTOTrainer: one row per answer with a desirability label."""
    def unpair(batch):
        out = {"prompt": [], "completion": [], "label": []}
        for prompt, chosen, rejected in zip(batch["prompt"], batch["chosen"], batch["rejected"]):
            out["prompt"] += [prompt, prompt]
            out["completion"] += [chosen, rejected]
            out["label"] += [True, False]
        return out

    keep = ds.select_columns(["prompt", "chosen", "rejected"])
    return keep.map(unpair, batched=True, remove_columns=["chosen", "rejected"])


MARKER = "Запрос студента: "


def user_message(document: str, request: str) -> str:
    """The first user turn: the open fragment, then the request. Shared by the
    data builder and by ad-hoc prompts, so both render identically."""
    head = f"Открытый фрагмент документа:\n{document}" if document else "Открытый документ пуст."
    return f"{head}\n\n{MARKER}{request}"


def prompt_for(request: str, document: str = "", dialog: list[dict] | None = None) -> list[dict]:
    """A `prompt` message list for a situation typed by hand.

    `dialog` is earlier turns as [{"user": ..., "assistant": ...}]; the
    document goes into the first user turn, exactly as in the data files.
    """
    turns = dialog or []
    first = turns[0]["user"] if turns else request
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user_message(document, first)}]
    for i, turn in enumerate(turns):
        if i:
            messages.append({"role": "user", "content": turn["user"]})
        messages.append({"role": "assistant", "content": turn["assistant"]})
    if turns:
        messages.append({"role": "user", "content": request})
    return messages


def request(row: dict) -> str:
    """The student's last message, without the document block."""
    text = row["prompt"][-1]["content"]
    return text.split(MARKER, 1)[1] if MARKER in text else text


def document_text(row: dict) -> str:
    return DOCUMENTS.get(row.get("document") or "", "")


def case(row: dict) -> dict:
    """The row with the document text in place, the shape checks expect."""
    return {**row, "document": document_text(row), "request": request(row)}


def show(row: dict, answer: str | None = None, width: int = 78) -> None:
    """Print one situation the way a person would read it."""
    line = "─" * width
    print(f"{row['id']} · {row['category']}" + ("  · ожидается отказ" if row.get("must_refuse") else ""))
    print(line)
    for message in row["prompt"][1:]:
        role = "СТУДЕНТ" if message["role"] == "user" else "АССИСТЕНТ"
        print(f"{role}:")
        print(message["content"])
        print()
    print("ЭТАЛОН:" if answer is None else "ОТВЕТ МОДЕЛИ:")
    print(row["chosen"][0]["content"] if answer is None else answer)
    print(line)
