"""Навыки: единственный инструмент ассистента студента.

Навык в этой архитектуре — не способность модели, а текст методики,
который подставляется в контекст после вызова `select_skill`. Модель
решает, какой навык нужен, код подставляет его текст, модель отвечает.
Тексты лежат в `data/skills.json`; тот же файл читают сборка данных,
обучение и агентский цикл, поэтому расхождений между ними нет.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

SKILLS_PATH = Path(__file__).resolve().parents[1] / "data" / "skills.json"

_skills: dict[str, dict[str, str]] | None = None


def skills() -> dict[str, dict[str, str]]:
    """{имя: {description, text}} из data/skills.json."""
    global _skills
    if _skills is None:
        _skills = json.loads(SKILLS_PATH.read_text(encoding="utf-8"))
    return _skills


def names() -> list[str]:
    return list(skills())


def schema() -> list[dict[str, Any]]:
    """Описание инструмента для ветки `tools` шаблона.

    Перечисление имён с описаниями лежит в самой схеме: модель выбирает
    навык по ним, а не по памяти об обучении.
    """
    listing = "; ".join(f"{n} — {s['description']}" for n, s in skills().items())
    return [{
        "type": "function",
        "function": {
            "name": "select_skill",
            "description": "Выбрать методику работы под запрос студента. Вызывается первым, до ответа. Навыки: " + listing,
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "enum": names(), "description": "имя навыка"},
                },
                "required": ["name"],
            },
        },
    }]


def run(name: str, arguments: dict[str, Any]) -> str:
    """Результат вызова: текст навыка или понятная ошибка.

    Ошибка возвращается текстом, а не исключением: модель должна её
    прочитать и исправить выбор, а цикл — не падать.
    """
    if name != "select_skill":
        return f"ошибка: инструмента {name!r} нет, доступен только select_skill"
    skill = str(arguments.get("name", "")).strip()
    if skill not in skills():
        return f"ошибка: навыка {skill!r} нет; доступны: {', '.join(names())}"
    return skills()[skill]["text"]
