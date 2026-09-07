"""Разбор вызовов инструментов.

Единого формата не существует. Одни модели порождают внутри
``<tool_call>`` объект JSON, другие — вложенные XML-подобные теги
``<function=...><parameter=...>``. Какой у вашей, видно из шаблона:

    print(processor.tokenizer.chat_template)

Разбираются оба, потому что выбирать не приходится: формат диктует
модель, а не мы.

Отдельно: у рассуждающих моделей шаблон открывает блок ``<think>``,
и его содержимое к ответу не относится. Перед разбором оно отсекается.
"""

from __future__ import annotations

import json
import re
from typing import Any

#: Внешняя обёртка вызова — общая для обоих форматов.
CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)

#: Вложенный XML: <function=имя> ... </function>
FUNCTION_RE = re.compile(r"<function=([^>]+)>\s*(.*?)\s*</function>", re.DOTALL)
PARAMETER_RE = re.compile(r"<parameter=([^>]+)>\s*(.*?)\s*</parameter>", re.DOTALL)

#: Блок размышления рассуждающей модели.
THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def strip_thinking(text: str) -> str:
    """Убрать блок размышления.

    Закрывающий тег может отсутствовать, если генерация оборвалась по
    лимиту токенов — тогда рассуждение так и не кончилось, и ответа нет.
    В этом случае возвращается пустая строка, что честнее, чем выдать
    обрывок рассуждения за ответ.
    """
    text = THINK_RE.sub("", text)
    if "<think>" in text:
        return ""
    return text.strip()


def _parse_xml_call(body: str) -> dict[str, Any] | None:
    """<function=имя><parameter=ключ>значение</parameter></function>"""
    match = FUNCTION_RE.search(body)
    if match is None:
        return None
    name, inner = match.group(1).strip(), match.group(2)
    arguments = {k.strip(): v for k, v in PARAMETER_RE.findall(inner)}
    return {"name": name, "arguments": arguments}


def _parse_json_call(body: str) -> dict[str, Any] | None:
    """{"name": ..., "arguments": {...}}"""
    try:
        payload = json.loads(body)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict) or "name" not in payload:
        return None
    payload.setdefault("arguments", {})
    return payload


def parse_tool_calls(text: str, *, drop_thinking: bool = True) -> list[dict[str, Any]]:
    """Вытащить вызовы из ответа модели.

        >>> parse_tool_calls('<tool_call>\\n<function=read_document>\\n'
        ...                  '<parameter=section_id>\\n3.2\\n</parameter>\\n'
        ...                  '</function>\\n</tool_call>')
        [{'name': 'read_document', 'arguments': {'section_id': '3.2'}}]

    Разбор терпимый: тег, который не удалось разобрать ни одним
    способом, пропускается, а не роняет замер. Доля таких пропусков
    сама по себе метрика — если она велика, модель не держит формат.
    """
    if drop_thinking:
        text = strip_thinking(text)

    calls = []
    for body in CALL_RE.findall(text):
        call = _parse_xml_call(body) or _parse_json_call(body)
        if call is not None:
            calls.append(call)
    return calls


def render_call(name: str, arguments: dict[str, Any], *, style: str = "xml") -> str:
    """Собрать вызов в том виде, в каком его порождает модель.

    Нужно для обучающих данных: формат в выборке обязан совпадать
    с тем, которого ждёт шаблон модели, иначе вы учите её одному,
    а инструментальная обвязка разбирает другое.
    """
    if style == "json":
        payload = json.dumps({"name": name, "arguments": arguments}, ensure_ascii=False)
        return f"<tool_call>\n{payload}\n</tool_call>"

    if style != "xml":
        raise ValueError(f"стиль {style!r}, доступны: xml, json")

    lines = [f"<function={name}>"]
    for key, value in arguments.items():
        lines.append(f"<parameter={key}>\n{value}\n</parameter>")
    lines.append("</function>")
    body = "\n".join(lines)
    return f"<tool_call>\n{body}\n</tool_call>"


def detect_style(chat_template: str | None) -> str:
    """Какой формат ждёт шаблон модели: "xml" или "json".

        >>> detect_style(processor.tokenizer.chat_template)

    Опознаётся по тому, упоминает ли шаблон вложенный тег ``<function=``.
    При неудаче возвращается "json" как более распространённый.
    """
    if not chat_template:
        return "json"
    return "xml" if "<function=" in chat_template else "json"
