"""Замер качества: генерация на отложенной выборке и подсчёт метрик.

Метрика всегда пара: доля правильных срабатываний и доля ложных.
Порознь они бессмысленны — модель, срабатывающая всегда, даёт идеальную
первую и катастрофическую вторую.
"""

from __future__ import annotations

import re
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Iterator, Sequence

import torch

from vlmkit.data import Sample
from vlmkit.toolcalls import parse_tool_calls, strip_thinking


@contextmanager
def left_padding(processor: Any) -> Iterator[None]:
    """Временно переключить дополнение влево.

    Для decoder-only модели при батче больше единицы правое дополнение
    ломает генерацию: короткая последовательность продолжается с позиции
    добивки, а не с реального последнего токена, и ответ получается
    мусорным. Обучению, наоборот, нужно правое — поэтому переключаем
    только на время генерации и возвращаем как было.
    """
    holders = [processor, getattr(processor, "tokenizer", None)]
    saved = [(h, getattr(h, "padding_side", None)) for h in holders if h is not None]
    try:
        for holder, side in saved:
            if side is not None:
                holder.padding_side = "left"
        yield
    finally:
        for holder, side in saved:
            if side is not None:
                holder.padding_side = side


@torch.inference_mode()
def generate(
    model: Any,
    processor: Any,
    samples: Sequence[Sample],
    *,
    system: str | None = None,
    max_new_tokens: int = 512,
    batch_size: int = 4,
    thinking: bool = False,
) -> list[str]:
    """Сгенерировать ответы на промпты выборки.

    Из каждого примера берутся все реплики до первой ассистентской —
    это и есть промпт, остальное было бы подсказкой.

    Рассуждение по умолчанию выключено: шаблон подставляет пустой блок
    размышления — тот же, что стоит перед ответом в обучающих данных.
    Иначе весь `max_new_tokens` уходит в рассуждение, и ответ не
    начинается. С `thinking=True` блок остаётся в выводе, скореры
    вырезают его через `strip_thinking`.
    """
    model.eval()
    with left_padding(processor):
        return _generate_batches(
            model, processor, samples, system, max_new_tokens, batch_size, thinking
        )


def _generate_batches(
    model: Any,
    processor: Any,
    samples: Sequence[Sample],
    system: str | None,
    max_new_tokens: int,
    batch_size: int,
    thinking: bool,
) -> list[str]:
    outputs: list[str] = []

    for start in range(0, len(samples), batch_size):
        chunk = [s.with_system(system) for s in samples[start : start + batch_size]]
        prompts = []
        for s in chunk:
            head: list[dict[str, Any]] = []
            for message in s.messages:
                if message["role"] == "assistant":
                    break
                head.append(message)
            prompts.append(head)

        texts = [
            processor.apply_chat_template(
                p, tokenize=False, add_generation_prompt=True, enable_thinking=thinking
            )
            for p in prompts
        ]
        images = [img for s in chunk for img in s.load_images()]
        kwargs: dict[str, Any] = {"text": texts, "return_tensors": "pt", "padding": True}
        if images:
            kwargs["images"] = images

        batch = processor(**kwargs).to(model.device)
        generated = model.generate(**batch, max_new_tokens=max_new_tokens, do_sample=False)
        width = batch["input_ids"].shape[1]
        outputs += [
            processor.decode(row[width:], skip_special_tokens=True).strip()
            for row in generated
        ]

    return outputs


@dataclass(slots=True)
class Suite:
    """Набор для замера: примеры, разметка групп и правило зачёта.

    `positive` — группы, где поведение должно срабатывать. Остальные
    группы образуют контрольную часть, на которой считается доля ложных.
    """

    name: str
    samples: list[Sample]
    groups: list[str]
    positive: set[str]
    #: (ответ модели, эталон) → сработало ли поведение
    fires: Callable[[str, Sample], bool]
    system: str | None = None

    def score(self, predictions: Sequence[str]) -> dict[str, float]:
        hits, false = [], []
        for group, prediction, sample in zip(self.groups, predictions, self.samples):
            (hits if group in self.positive else false).append(
                self.fires(prediction, sample)
            )
        return {
            "hit": sum(hits) / max(len(hits), 1),
            "false": sum(false) / max(len(false), 1),
            "n": len(predictions),
        }


def run(model: Any, processor: Any, suite: Suite, **kwargs: Any) -> dict[str, float]:
    """Прогнать один набор и вернуть метрики."""
    predictions = generate(model, processor, suite.samples, system=suite.system, **kwargs)
    return suite.score(predictions)


# ── готовые правила зачёта ────────────────────────────────────────────


def returns_question(text: str, sample: Sample) -> bool:
    """Модель вернула решение пользователю, а не выдала готовый ответ.

    Блок размышления отсекается: у рассуждающей модели он полон вопросов
    к самой себе, и без этого метрика мерила бы наличие размышления,
    а не поведение в ответе.
    """
    return "?" in strip_thinking(text)


def calls_tool(name: str) -> Callable[[str, Sample], bool]:
    """Модель вызвала именно этот инструмент."""

    def check(text: str, sample: Sample) -> bool:
        return any(c.get("name") == name for c in parse_tool_calls(text))

    return check


def calls_any_tool(text: str, sample: Sample) -> bool:
    return bool(parse_tool_calls(text))


def matches_format(pattern: str) -> Callable[[str, Sample], bool]:
    """Ответ соответствует заданной структуре.

    Для скиллов формат задан жёстко, поэтому соблюдение проверяется
    регулярным выражением, без второй модели в роли судьи.
    """
    compiled = re.compile(pattern, re.MULTILINE | re.DOTALL)

    def check(text: str, sample: Sample) -> bool:
        return bool(compiled.search(text.strip()))

    return check
