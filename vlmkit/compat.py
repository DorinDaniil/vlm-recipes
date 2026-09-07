"""Совместимость с версиями transformers и trl.

Сигнатуры `TrainingArguments` и конфигов TRL меняются между версиями:
аргументы переименовывают, переносят и удаляют, а тренеры сливают в один
с разными значениями `loss_type`. Передача словаря вслепую роняет запуск
на первом расхождении — причём в середине прогона.

Модуль решает две задачи и обе скучные: отсеять неизвестные аргументы
и найти, каким классом ваша версия TRL реализует нужный метод. Сами
вызовы обучения живут в ноутбуке — их прятать незачем.
"""

from __future__ import annotations

import dataclasses
import inspect
import typing
from typing import Any

#: Что уже сообщали — чтобы не повторять на каждом прогоне.
_reported: set[tuple[str, str]] = set()


# ── фильтрация аргументов ─────────────────────────────────────────────


def accepted_names(cls: type) -> set[str]:
    """Имена аргументов, которые принимает конструктор."""
    if dataclasses.is_dataclass(cls):
        return {f.name for f in dataclasses.fields(cls)}
    try:
        return set(inspect.signature(cls.__init__).parameters) - {"self"}
    except (TypeError, ValueError):
        return set()


def supported(cls: type, kwargs: dict[str, Any], *, quiet: bool = False) -> dict[str, Any]:
    """Оставить только аргументы, которые `cls` действительно принимает.

        >>> TrainingArguments(**supported(TrainingArguments, kwargs))

    Отсеянное печатается один раз на класс: молчаливое игнорирование хуже
    ошибки, потому что вы считали бы разогрев включённым, а его нет.
    """
    allowed = accepted_names(cls)
    if not allowed:  # сигнатуру прочитать не удалось — отдаём как есть
        return kwargs

    kept = {k: v for k, v in kwargs.items() if k in allowed}
    if (dropped := sorted(set(kwargs) - set(kept))) and not quiet:
        key = (cls.__name__, ",".join(dropped))
        if key not in _reported:
            _reported.add(key)
            print(f"[vlmkit] {cls.__name__} не принимает: {', '.join(dropped)} — пропущены")
    return kept


def first_accepted(cls: type, candidates: dict[str, Any]) -> dict[str, Any]:
    """Первое имя из списка синонимов, которое класс принимает.

        >>> first_accepted(TrainingArguments,
        ...                {"warmup_ratio": 0.03, "warmup_steps": 10})

    Нужно там, где аргумент не исчез, а переименован.
    """
    allowed = accepted_names(cls)
    for name, value in candidates.items():
        if name in allowed:
            return {name: value}
    return {}


# ── поиск тренеров TRL ────────────────────────────────────────────────

#: Какими классами TRL реализует каждый метод выравнивания. Список, а не
#: пара: между версиями метод переезжает. В trl 1.x отдельные ORPOTrainer
#: и CPOTrainer слили в DPOTrainer с разными значениями `loss_type`.
ALIGNMENT: dict[str, list[tuple[str, str, dict[str, Any]]]] = {
    "dpo": [("DPOConfig", "DPOTrainer", {})],
    "orpo": [
        ("ORPOConfig", "ORPOTrainer", {}),
        ("DPOConfig", "DPOTrainer", {"loss_type": "orpo"}),
    ],
    "simpo": [
        ("CPOConfig", "CPOTrainer", {"loss_type": "simpo"}),
        ("DPOConfig", "DPOTrainer", {"loss_type": "simpo"}),
    ],
    "kto": [("KTOConfig", "KTOTrainer", {})],
}


def loss_types(config: type) -> set[str]:
    """Какие значения `loss_type` принимает конфиг.

    Читается из аннотации поля: в TRL это Literal с перечислением.
    Пустое множество означает, что прочитать не удалось, — тогда проверку
    пропускаем и полагаемся на ошибку самого TRL.
    """
    if not dataclasses.is_dataclass(config):
        return set()
    for item in dataclasses.fields(config):
        if item.name == "loss_type":
            return {a for a in typing.get_args(item.type) if isinstance(a, str)}
    return set()


def _try_alignment(method: str) -> tuple[Any, Any, dict[str, Any]] | None:
    """Первый подходящий кандидат или None. Исключений не поднимает.

    Кандидат с `loss_type` принимается, только если конфиг это значение
    действительно знает: иначе метод числился бы доступным, а падал бы
    посреди обучения.
    """
    import trl

    for config_name, trainer_name, extra in ALIGNMENT[method]:
        config = getattr(trl, config_name, None)
        trainer = getattr(trl, trainer_name, None)
        if config is None or trainer is None:
            continue
        wanted = extra.get("loss_type")
        known = loss_types(config)
        if wanted is not None and known and wanted not in known:
            continue
        return config, trainer, extra
    return None


def available_alignment() -> dict[str, str]:
    """Какие методы выравнивания есть в установленном trl.

        >>> available_alignment()
        {'dpo': 'DPOTrainer', 'orpo': 'DPOTrainer(loss_type=orpo)'}
    """
    found = {}
    for method in ALIGNMENT:
        if (resolved := _try_alignment(method)) is not None:
            _, trainer, extra = resolved
            suffix = f"(loss_type={extra['loss_type']})" if "loss_type" in extra else ""
            found[method] = trainer.__name__ + suffix
    return found


def alignment_trainer(method: str) -> tuple[Any, Any, dict[str, Any]]:
    """Классы TRL для метода: `(Config, Trainer, доп. аргументы)`.

        >>> Config, Trainer, extra = alignment_trainer("orpo")
        >>> trainer = Trainer(model=model, args=Config(**extra, **kwargs), ...)

    Если метода нет ни в одном виде — объясняет, что искали и что есть.
    """
    if method not in ALIGNMENT:
        raise ValueError(f"метод {method!r}, доступны: {list(ALIGNMENT)}")
    if (resolved := _try_alignment(method)) is not None:
        return resolved

    import trl

    # Список доступного строится через _try_alignment, а не через
    # available_alignment: иначе получилась бы взаимная рекурсия.
    have = [m for m in ALIGNMENT if _try_alignment(m) is not None]
    tried = ", ".join(
        c + (f"(loss_type={e['loss_type']})" if "loss_type" in e else "")
        for c, _, e in ALIGNMENT[method]
    )
    raise ImportError(
        f"{method}: в trl {getattr(trl, '__version__', '?')} не нашлось ни одного "
        f"из вариантов — {tried}.\n"
        f"Доступны: {', '.join(have) or 'ничего из четырёх'}.\n"
        f"Возьмите доступный метод либо смените версию trl."
    )
