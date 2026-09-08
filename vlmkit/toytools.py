"""Игрушечные инструменты для агентских примеров.

Считают по-настоящему. Поэтому правильный ответ известен заранее,
и конечную реплику агента можно проверить, а не оценивать на глаз.

`SCHEMA` — то, что видит модель через ветку `tools` шаблона.
`TOOLS` — то, что реально исполняется. Модель порождает строку; имя
из неё сопоставляет с функцией и запускает код, см. `run`.
"""

from __future__ import annotations

import ast
import operator
import random
from typing import Any, Callable

_OPS: dict[type, Callable[..., Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def calc(expression: str) -> str:
    """Арифметика: + - * / // % ** и скобки. Ничего другого — это не eval."""

    def value(node: ast.AST) -> float:
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](value(node.left), value(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](value(node.operand))
        raise ValueError(f"недопустимое выражение: {expression!r}")

    cleaned = expression.replace("×", "*").replace("÷", "/").replace("^", "**")
    try:
        result = value(ast.parse(cleaned, mode="eval").body)
    except (SyntaxError, ValueError, ZeroDivisionError, OverflowError) as exc:
        return f"ошибка: {exc}"
    return str(int(result)) if float(result).is_integer() else f"{result:.4g}"


def word_count(text: str) -> str:
    return str(len(text.split()))


def reverse(text: str) -> str:
    return text[::-1]


def roll_dice(sides: int | str = 6) -> str:
    return str(random.randint(1, int(sides)))


TOOLS: dict[str, Callable[..., str]] = {
    "calc": calc,
    "word_count": word_count,
    "reverse": reverse,
    "roll_dice": roll_dice,
}


def _spec(name: str, description: str, **params: str) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": {k: {"type": "string", "description": v} for k, v in params.items()},
                "required": list(params),
            },
        },
    }


SCHEMA: list[dict[str, Any]] = [
    _spec("calc", "Вычислить арифметическое выражение", expression="выражение, например 17*23"),
    _spec("word_count", "Посчитать число слов в тексте", text="текст"),
    _spec("reverse", "Записать строку наоборот", text="строка"),
    _spec("roll_dice", "Бросить кубик", sides="число граней, обычно 6"),
]


def run(name: str, arguments: dict[str, Any]) -> str:
    """Выполнить вызов, разобранный из ответа модели.

    Ошибка — тоже результат: она возвращается текстом, и модель должна
    уметь её прочитать, а не падать вместе с циклом.
    """
    if name not in TOOLS:
        return f"ошибка: инструмента {name!r} нет"
    try:
        return TOOLS[name](**arguments)
    except (TypeError, ValueError) as exc:
        return f"ошибка: {exc}"
