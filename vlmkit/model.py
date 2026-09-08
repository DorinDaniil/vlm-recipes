"""Процессор и видеопамять. Саму модель ноутбуки грузят сами.

Загрузка весов намеренно оставлена в ячейках: `from_pretrained` со всеми
аргументами должно быть видно, а не спрятано за обёрткой.
"""

from __future__ import annotations

from typing import Any

import torch
from transformers import AutoProcessor

#: Потолок визуальных токенов на картинку. Страница A4 в полном разрешении
#: даёт тысячи токенов, поэтому ограничивать надо до токенизации.
MAX_PIXELS = 1003520


def load_processor(model_id: str, *, max_pixels: int | None = MAX_PIXELS) -> Any:
    """Процессор: текст и картинки → тензоры."""
    try:
        return AutoProcessor.from_pretrained(
            model_id, trust_remote_code=True, max_pixels=max_pixels
        )
    except (TypeError, ValueError):
        # Не все процессоры принимают max_pixels в конструкторе.
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        if (images := getattr(processor, "image_processor", None)) is not None and max_pixels:
            images.max_pixels = max_pixels
        return processor


def memory_report() -> str:
    if not torch.cuda.is_available():
        return "CUDA недоступна"
    parts = []
    for i in range(torch.cuda.device_count()):
        used = torch.cuda.memory_allocated(i) / 2**30
        total = torch.cuda.get_device_properties(i).total_memory / 2**30
        parts.append(f"cuda:{i} {used:.1f}/{total:.0f} GiB")
    return "  ".join(parts)


def cleanup(*objects: Any) -> str:
    """Вернуть видеопамять в ноутбуке. Возвращает отчёт по памяти.

        >>> del model, processor, trainer
        >>> print(cleanup())

    В ноутбуке недостаточно `del` и `empty_cache`, потому что ссылки на
    модель остаются ещё в двух местах, и оба невидимы:

    **Трассировка последнего исключения.** После ошибки — особенно после
    OOM — Python держит `sys.last_traceback`, а тот держит кадры стека
    со всеми локальными переменными, включая наполовину загруженную модель.
    Именно поэтому после OOM повторная загрузка падает снова.

    **Кэш вывода IPython.** Ячейка, вернувшая модель, кладёт её в `Out[N]`
    и в `_`, `__`, `___`. Переменную вы удалили, а кэш держит.

    Функция чистит оба места и переносит переданные объекты на процессор.
    Свои ссылки удаляйте сами — из чужой функции это невозможно.
    """
    import gc
    import sys

    for obj in objects:
        move = getattr(obj, "to", None)
        if move is not None:
            try:
                move("cpu")
            except (NotImplementedError, ValueError, RuntimeError):
                pass  # квантованные модели на процессор не переносятся

    for name in ("last_traceback", "last_value", "last_type", "last_exc"):
        try:
            delattr(sys, name)
        except AttributeError:
            pass

    try:
        from IPython import get_ipython

        if (ip := get_ipython()) is not None:
            ip.displayhook.flush()  # чистит Out[N] и _, __, ___
            for name in ("_", "__", "___"):
                ip.user_ns.pop(name, None)
    except ImportError:
        pass

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    return memory_report()
