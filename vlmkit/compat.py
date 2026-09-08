"""Совместимость с версиями transformers и trl.

Сигнатуры `TrainingArguments` и конфигов TRL меняются между версиями:
аргументы переименовывают, переносят и удаляют, а тренеры переезжают
между модулями. Передача словаря вслепую роняет запуск на первом
расхождении — причём в середине прогона.

Модуль решает две задачи и обе скучные: отсеять неизвестные аргументы
и найти, где в вашей версии TRL лежит класс под нужный метод. Сами
вызовы обучения живут в ноутбуке — их прятать незачем.
"""

from __future__ import annotations

import dataclasses
import importlib
import inspect
import re
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

#: Где каждый метод выравнивания лежит в TRL: (модуль, конфиг, тренер,
#: обязательные аргументы конфига). Список, а не одна запись: между
#: версиями класс переезжает. В trl 1.x `ORPOTrainer` и `CPOTrainer`
#: вынесены в `trl.experimental`, `KTOTrainer` остался на верхнем уровне.
#: Отдельного SimPO в TRL нет: это `CPOTrainer` с `loss_type="simpo"`
#: и выключенным NLL-слагаемым (`cpo_alpha=0`).
Candidate = tuple[str, str, str, dict[str, Any]]

SIMPO: dict[str, Any] = {"loss_type": "simpo", "cpo_alpha": 0.0}

ALIGNMENT: dict[str, list[Candidate]] = {
    "dpo": [("trl", "DPOConfig", "DPOTrainer", {})],
    "orpo": [
        ("trl", "ORPOConfig", "ORPOTrainer", {}),
        ("trl.experimental.orpo", "ORPOConfig", "ORPOTrainer", {}),
    ],
    "simpo": [
        ("trl", "CPOConfig", "CPOTrainer", SIMPO),
        ("trl.experimental.cpo", "CPOConfig", "CPOTrainer", SIMPO),
    ],
    "kto": [
        ("trl", "KTOConfig", "KTOTrainer", {}),
        ("trl.experimental.kto", "KTOConfig", "KTOTrainer", {}),
    ],
}

_QUOTED = re.compile(r"'(\w+)'")


def loss_types(config: type) -> set[str]:
    """Какие значения `loss_type` принимает конфиг.

    Источники по убыванию надёжности: Literal в аннотации, `choices`
    в метаданных поля, имена в кавычках в тексте подсказки. Пустое
    множество — прочитать не удалось; тогда проверка пропускается
    и остаётся ошибка самого TRL.
    """
    if not dataclasses.is_dataclass(config):
        return set()
    for item in dataclasses.fields(config):
        if item.name != "loss_type":
            continue
        annotation = item.type
        found = {a for a in typing.get_args(annotation) if isinstance(a, str)}
        if not found and isinstance(annotation, str) and "Literal" in annotation:
            found = set(_QUOTED.findall(annotation))
        if not found:
            found = {str(c) for c in item.metadata.get("choices", ())}
        if not found:
            found = set(_QUOTED.findall(str(item.metadata.get("help", ""))))
        return found
    return set()


def _load(module: str, name: str) -> Any:
    try:
        return getattr(importlib.import_module(module), name, None)
    except ImportError:
        return None


def _try_alignment(method: str) -> tuple[Any, Any, dict[str, Any], str] | None:
    """Первый кандидат, который импортируется и знает нужный `loss_type`.

    Возвращает `(Config, Trainer, extra, где_нашли)` или None; исключений
    не поднимает. Кандидат с `loss_type` принимается, только если конфиг
    это значение знает: иначе метод числился бы доступным, а падал бы
    на первом шаге обучения.
    """
    for module, config_name, trainer_name, extra in ALIGNMENT[method]:
        config = _load(module, config_name)
        trainer = _load(module, trainer_name)
        if config is None or trainer is None:
            continue
        wanted = extra.get("loss_type")
        known = loss_types(config)
        if wanted is not None and known and wanted not in known:
            continue
        return config, trainer, dict(extra), f"{module}.{trainer_name}"
    return None


def available_alignment() -> dict[str, str]:
    """Какие методы есть в установленном trl и каким классом.

        >>> available_alignment()
        {'dpo': 'trl.DPOTrainer', 'orpo': 'trl.experimental.orpo.ORPOTrainer',
         'simpo': 'trl.experimental.cpo.CPOTrainer(loss_type=simpo)', 'kto': 'trl.KTOTrainer'}
    """
    found = {}
    for method in ALIGNMENT:
        if (resolved := _try_alignment(method)) is not None:
            _, _, extra, where = resolved
            suffix = f"(loss_type={extra['loss_type']})" if "loss_type" in extra else ""
            found[method] = where + suffix
    return found


def alignment_trainer(method: str) -> tuple[Any, Any, dict[str, Any]]:
    """Классы TRL для метода: `(Config, Trainer, обязательные аргументы)`.

        >>> Config, Trainer, extra = alignment_trainer("simpo")
        >>> trainer = Trainer(model=model, args=Config(**extra, **kwargs), ...)

    `extra` подмешивается в конфиг обязательно: для SimPO это `loss_type`
    и `cpo_alpha=0`, без них получится CPO. Если метода нет ни в одном
    виде — объясняет, что искали и что есть.
    """
    if method not in ALIGNMENT:
        raise ValueError(f"метод {method!r}, доступны: {list(ALIGNMENT)}")
    if (resolved := _try_alignment(method)) is not None:
        config, trainer, extra, _ = resolved
        return config, trainer, extra

    import trl

    have = available_alignment()
    tried = ", ".join(f"{m}.{t}" for m, _, t, _ in ALIGNMENT[method])
    raise ImportError(
        f"{method}: в trl {getattr(trl, '__version__', '?')} не нашлось ни одного "
        f"из вариантов — {tried}.\n"
        f"Доступны: {', '.join(have) or 'ничего из четырёх'}.\n"
        f"Возьмите доступный метод либо смените версию trl."
    )
