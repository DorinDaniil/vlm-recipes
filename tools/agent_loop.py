"""Агентский цикл целиком: сборка промпта, think, префилл, вызовы.

Показывает, как три вещи складываются на практике, и где проходит
граница между тем, что решает код, и тем, что решает модель.

Запускать не обязательно — файл писался, чтобы его читать.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vlmkit.toolcalls import detect_style, parse_tool_calls, strip_thinking

MAX_STEPS = 8


# ── что умеет система ─────────────────────────────────────────────────

TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "read_document",
            "description": "Текст раздела работы",
            "parameters": {
                "type": "object",
                "properties": {"section_id": {"type": "string"}},
                "required": ["section_id"],
            },
        },
    },
]

#: Что реально исполняется. Модель имён этих функций не вызывает —
#: она порождает строку, а сопоставляет и запускает вот этот словарь.
TOOLS: dict[str, Callable[..., str]] = {
    "read_document": lambda section_id: f"Раздел {section_id}: ...",
}


# ── сборка промпта ────────────────────────────────────────────────────


def build_prompt(processor: Any, messages: list[dict], prefill: str = "") -> str:
    """Список реплик плюс необязательное начало ответа → одна строка.

    Промпт пересобирается на каждом шаге целиком: накапливается
    `messages`, а не строка. Так контекст остаётся управляемым — можно
    выкинуть лишнее перед сборкой, и это не сломает разметку.
    """
    prompt = processor.apply_chat_template(
        messages, tools=TOOLS_SCHEMA, tokenize=False, add_generation_prompt=True
    )
    # Хвост сейчас такой: '<|im_start|>assistant\n<think>\n'
    # Префилл дописывается уже после шаблона — это просто конкатенация,
    # никакого отдельного механизма за ним нет.
    return prompt + prefill


def choose_prefill(state: dict) -> str:
    """Префилл выбирает КОД по состоянию, а не модель по своему усмотрению.

    Пустой `<think>\\n\\n</think>` означает «рассуждать не нужно» —
    ровно то же, что шаблон подставляет сам для завершённых реплик.
    """
    if not state.get("problem_defined"):
        # Проблема не определена — гипотезу формулировать не из чего.
        return "\n</think>\n\nУточняющий вопрос:"
    if state.get("force_tool"):
        # Гарантируем вызов: имя функции модель уже не выбирает — она
        # его дописывает.
        return "\n</think>\n\n<tool_call>\n<function="
    return ""


# ── цикл ──────────────────────────────────────────────────────────────


def run_agent(model: Any, processor: Any, request: str, state: dict) -> list[dict]:
    """Один заход агента. Возвращает историю целиком.

    Три решения на каждом шаге, и все три принимает код, а не модель:
    какой префилл поставить, что положить в историю и когда остановиться.
    """
    style = detect_style(processor.tokenizer.chat_template)
    messages = [
        {"role": "system", "content": "Ты помогаешь студенту писать работу."},
        {"role": "user", "content": request},
    ]

    for _ in range(MAX_STEPS):
        prefill = choose_prefill(state)
        prompt = build_prompt(processor, messages, prefill)

        # Токенизируем готовую строку, минуя шаблон: он уже отработал.
        inputs = processor(text=[prompt], return_tensors="pt").to(model.device)
        out = model.generate(**inputs, max_new_tokens=512, do_sample=False)
        generated = processor.decode(
            out[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )

        # Префилл и продолжение — одна реплика. Разорвав их, вы положите
        # в историю ответ, начинающийся с середины фразы.
        reply = prefill.lstrip("\n") + generated

        calls = parse_tool_calls(reply)

        # В историю кладём БЕЗ рассуждения. Оно нужно было для текущего
        # шага и на следующем только съест контекст и собьёт модель.
        messages.append({"role": "assistant", "content": strip_thinking(reply)})

        if not calls:
            break  # модель ответила словами — цикл окончен

        for call in calls:
            handler = TOOLS.get(call["name"])
            result = (
                handler(**call["arguments"])
                if handler
                else f"Ошибка: инструмента {call['name']} не существует"
            )
            # Результат приходит извне. В обучающих данных эти позиции
            # маскируются, иначе модель начнёт его сочинять.
            messages.append({"role": "tool", "content": result})

        state["force_tool"] = False  # форсируем не более одного раза
    else:
        # Вышли по лимиту, а не по ответу: модель не остановилась сама.
        # На практике это самый частый способ спалить бюджет, поэтому
        # умение останавливаться выносят в обучающие данные отдельно.
        messages.append(
            {"role": "assistant", "content": "Не удалось завершить за отведённые шаги."}
        )

    del style  # нужен, только если собираете вызовы сами
    return messages


# ── что где решается ──────────────────────────────────────────────────

SPLIT = """
код решает                          модель решает
──────────────────────────────      ──────────────────────────────
что попадёт в system                вызывать ли инструмент и какой
какой префилл поставить             что написать в аргументах
выкидывать ли think из истории      когда перестать вызывать
выполнять ли вызов                  что ответить словами
когда оборвать по лимиту
"""

if __name__ == "__main__":
    print(__doc__)
    print(SPLIT)
