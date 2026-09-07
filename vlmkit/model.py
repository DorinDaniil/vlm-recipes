"""Загрузка модели — единственное место, где веса попадают в память."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

from vlmkit.config import settings


@dataclass(slots=True)
class LoadConfig:
    """Параметры загрузки."""

    model_id: str = settings.model_id

    #: "cuda" — всё на одну карту. "auto" — разложить по всем видимым
    #: (нужно, когда модель на карту не помещается; при обучении на
    #: нескольких картах через torchrun ставьте "cuda", шардированием
    #: занимается DDP или FSDP).
    device: str = "cuda"

    #: bfloat16, а не float16: шире экспонента, не переполняется на активациях.
    dtype: torch.dtype = torch.bfloat16

    #: Разрядность базы: None — как есть, 4 или 8 — квантовать при загрузке.
    #: Для QLoRA передайте `SFTConfig.bits`.
    bits: int | None = None

    #: Потолок визуальных токенов на картинку. Влияет и на память,
    #: и на длину промпта: страница A4 в полном разрешении даёт тысячи токенов.
    max_pixels: int | None = settings.max_pixels
    min_pixels: int | None = None

    #: "flash_attention_2" быстрее на длинном контексте, но требует
    #: отдельно собранного пакета.
    attn_implementation: str = "sdpa"
    trust_remote_code: bool = True


def _quantization_config(cfg: LoadConfig):
    if cfg.bits is None:
        return None

    from transformers import BitsAndBytesConfig

    if cfg.bits == 8:
        return BitsAndBytesConfig(load_in_8bit=True)

    # nf4 подобран под то, как реально распределены веса сети, и точнее
    # обычного int4. compute_dtype — тип, в который веса разжимаются
    # перед умножением; храниться они продолжают в 4 битах.
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=cfg.dtype,
        bnb_4bit_use_double_quant=True,
    )


def load_processor(cfg: LoadConfig) -> Any:
    """Процессор: текст и картинки → тензоры.

    Разрешение ограничивается здесь, а не при генерации: картинку нужно
    ужать до того, как она станет визуальными токенами.
    """
    kwargs: dict[str, Any] = {"trust_remote_code": cfg.trust_remote_code}
    if cfg.max_pixels is not None:
        kwargs["max_pixels"] = cfg.max_pixels
    if cfg.min_pixels is not None:
        kwargs["min_pixels"] = cfg.min_pixels

    try:
        return AutoProcessor.from_pretrained(cfg.model_id, **kwargs)
    except (TypeError, ValueError):
        # Не все процессоры принимают max_pixels в конструкторе.
        processor = AutoProcessor.from_pretrained(
            cfg.model_id, trust_remote_code=cfg.trust_remote_code
        )
        if (image_processor := getattr(processor, "image_processor", None)) is not None:
            if cfg.max_pixels is not None:
                image_processor.max_pixels = cfg.max_pixels
            if cfg.min_pixels is not None:
                image_processor.min_pixels = cfg.min_pixels
        return processor


def _device_map(cfg: LoadConfig) -> Any:
    if cfg.device == "auto":
        return "auto"
    if cfg.device == "cuda":
        # Явный индекс, а не "auto": под torchrun каждый процесс должен
        # держать свою копию целиком, иначе DDP и accelerate поспорят.
        import os

        return {"": int(os.environ.get("LOCAL_RANK", 0))}
    return cfg.device


def load(cfg: LoadConfig | None = None) -> tuple[Any, Any]:
    """Загрузить модель и процессор.

    Занимает минуты и десятки гигабайт трафика при первом запуске —
    в ноутбуке держите в отдельной ячейке.
    """
    cfg = cfg or LoadConfig()

    model = AutoModelForImageTextToText.from_pretrained(
        cfg.model_id,
        dtype=cfg.dtype,
        device_map=_device_map(cfg),
        quantization_config=_quantization_config(cfg),
        attn_implementation=cfg.attn_implementation,
        trust_remote_code=cfg.trust_remote_code,
    )
    return model.eval(), load_processor(cfg)


def free(*objects: Any) -> None:
    """Вернуть видеопамять между прогонами.

    Веса переносятся на процессор, а не просто удаляются: `del` внутри
    функции убирает только локальное имя, тогда как у вызывающего ссылка
    остаётся, и память не возвращается. Перенос на CPU освобождает карту
    независимо от того, кто ещё держит объект.

    Вызывающему всё равно стоит сделать `del` своих ссылок до вызова —
    иначе модель останется висеть в оперативной памяти.
    """
    import gc

    for obj in objects:
        move = getattr(obj, "to", None)
        if move is None:
            continue
        try:
            move("cpu")
        except (NotImplementedError, ValueError, RuntimeError):
            # Квантованные модели переносить на процессор не умеют —
            # им остаётся только сборка мусора.
            pass

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


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
    со всеми локальными переменными, включая наполовину загруженную
    модель. Именно поэтому после OOM повторная загрузка падает снова.

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
                pass

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
