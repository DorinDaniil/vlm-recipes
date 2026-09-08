"""vlmkit — части, которые скучно писать в каждой ячейке.

Обёрток над обучением здесь нет: `LoraConfig`, `Trainer` и тренеры TRL
вызываются напрямую в ноутбуке, иначе непонятно, что происходит.

    data        формат примера и коллатор с маскированием
    evaluate    perplexity и предпочтение пар по логитам
    skills      единственный инструмент ассистента: select_skill
    rubric      автопроверки ответа по рубрикам голд-сета
    toolcalls   разбор и сборка вызовов инструментов
    steering    векторы управления
    compat      фильтр аргументов и поиск тренеров TRL по версии
    model       процессор и очистка видеопамяти

Модули, которым нужен torch (`data`, `evaluate`, `steering`, `model`),
подгружаются по первому обращению: проверки данных и автопроверки
работают на чистой стандартной библиотеке, без установленного torch.

Сборка ситуации, агентский цикл, метрики и печать — в `notebooks/common.py`.
"""

import importlib
from typing import Any

from vlmkit import compat, rubric, skills, toolcalls

_LAZY = {
    "data": "vlmkit.data",
    "evaluate": "vlmkit.evaluate",
    "model": "vlmkit.model",
    "steering": "vlmkit.steering",
    "ChatCollator": "vlmkit.data",
    "Sample": "vlmkit.data",
    "preview": "vlmkit.data",
    "cleanup": "vlmkit.model",
    "load_processor": "vlmkit.model",
    "memory_report": "vlmkit.model",
}


def __getattr__(name: str) -> Any:
    if name not in _LAZY:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = importlib.import_module(_LAZY[name])
    value = module if _LAZY[name].endswith(name) else getattr(module, name)
    globals()[name] = value
    return value


__all__ = [
    "ChatCollator",
    "Sample",
    "cleanup",
    "compat",
    "data",
    "evaluate",
    "load_processor",
    "memory_report",
    "model",
    "preview",
    "rubric",
    "skills",
    "steering",
    "toolcalls",
]

__version__ = "0.4.0"
