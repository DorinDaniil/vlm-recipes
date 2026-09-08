"""Векторы управления: изменение поведения без дообучения.

Абстрактные свойства ответа представлены в активациях приблизительно
линейно. Берём два набора текстов, различающихся только интересующим
свойством, снимаем активации на среднем слое и вычитаем средние.
Полученную разность прибавляем к скрытым состояниям при генерации.

Границы применимости — в `03-alignment.pdf`.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator, Sequence

import torch


def decoder_layers(model: Any) -> Any:
    """Слои декодера. Путь различается между семействами, а у
    мультимодальных моделей языковая часть ещё и вложена."""
    for path in (
        "model.language_model.layers",
        "model.model.language_model.layers",
        "model.layers",
        "model.model.layers",
        "language_model.model.layers",
    ):
        node = model
        for attr in path.split("."):
            if (node := getattr(node, attr, None)) is None:
                break
        if node is not None:
            return node
    raise AttributeError("не найдены слои декодера — укажите путь вручную")


@dataclass(slots=True)
class SteeringVector:
    """Единичное направление, его исходная длина и слой.

    `direction` нормировано, `scale` — норма разности средних до
    нормировки. При `strength=1` прибавляется ровно разность средних:
    это естественная единица, в ней написаны работы про вектор отказа.
    """

    direction: torch.Tensor
    scale: float
    layer: int

    @classmethod
    def from_contrast(
        cls,
        model: Any,
        processor: Any,
        positive: Sequence[str],
        negative: Sequence[str],
        layer: int,
    ) -> SteeringVector:
        """Построить вектор по двум наборам текстов.

        Наборы должны различаться именно интересующим свойством: если
        они различаются ещё и темой, вектор выучит тему. Удобно давать
        пары с общим началом — общая часть в разности сокращается.
        Хватает нескольких десятков пар.
        """
        pos = _mean_activation(model, processor, positive, layer)
        neg = _mean_activation(model, processor, negative, layer)
        diff = pos - neg
        return cls(diff / diff.norm(), float(diff.norm()), layer)

    @contextmanager
    def applied(self, model: Any, strength: float = 1.0) -> Iterator[None]:
        """Включить вектор на время блока.

        Начинайте со strength=1.0 и смотрите на сырые ответы: ниже 0.5
        эффект тонет в шуме, выше 2 у большинства моделей рушится
        связность. Отрицательные значения подавляют свойство.
        """
        shift = (strength * self.scale * self.direction).to(model.device)

        def hook(module: Any, args: Any, output: Any) -> Any:
            # Слой декодера возвращает кортеж, скрытые состояния первые.
            hidden = output[0] if isinstance(output, tuple) else output
            # Вектор хранится в float32 ради устойчивости усреднения,
            # а модель считает в bfloat16. Без приведения типа следующий
            # линейный слой падает на несовпадении dtype.
            shifted = hidden + shift.to(hidden.dtype)
            return (shifted, *output[1:]) if isinstance(output, tuple) else shifted

        handle = decoder_layers(model)[self.layer].register_forward_hook(hook)
        try:
            yield
        finally:
            handle.remove()

    def save(self, path: str) -> None:
        torch.save({"direction": self.direction, "scale": self.scale, "layer": self.layer}, path)

    @classmethod
    def load(cls, path: str) -> SteeringVector:
        data = torch.load(path)
        return cls(data["direction"], data["scale"], data["layer"])


def _mean_activation(
    model: Any, processor: Any, texts: Sequence[str], layer: int
) -> torch.Tensor:
    """Средняя активация на слое. Усредняем по всем позициям: вариант
    с последним токеном чувствительнее к формулировке и шумит сильнее."""
    captured: list[torch.Tensor] = []

    def hook(module: Any, args: Any, output: Any) -> None:
        hidden = output[0] if isinstance(output, tuple) else output
        captured.append(hidden.detach().float().mean(dim=1).squeeze(0).cpu())

    handle = decoder_layers(model)[layer].register_forward_hook(hook)
    try:
        for text in texts:
            inputs = processor(text=[text], return_tensors="pt").to(model.device)
            with torch.inference_mode():
                model(**inputs)
    finally:
        handle.remove()

    return torch.stack(captured).mean(dim=0)


def suggest_layer(model: Any) -> int:
    """Откуда начинать подбор: ранние слои кодируют форму, поздние —
    конкретные токены, абстрактные свойства лежат в середине."""
    return len(decoder_layers(model)) // 2
