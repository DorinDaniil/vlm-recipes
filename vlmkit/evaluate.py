"""Вероятностные метрики: perplexity и предпочтение пар.

Генерация и подсчёт попаданий живут в `notebooks/common.py` — они зависят
от формата ситуации, а здесь только то, что считается по логитам и не
зависит ни от задачи, ни от рубрик.
"""

from __future__ import annotations

import math
from contextlib import contextmanager
from typing import Any, Iterator, Sequence

import torch

from vlmkit.data import IGNORE_INDEX, ChatCollator, Sample


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
def sequence_logprobs(
    model: Any,
    processor: Any,
    samples: Sequence[Sample],
    *,
    system: str | None = None,
    tools: list[dict[str, Any]] | None = None,
    batch_size: int = 2,
) -> list[tuple[float, int]]:
    """Сумма log-вероятностей обучаемых токенов и их число, по примеру.

    Считается тем же коллатором, что при обучении, поэтому в сумму входят
    ровно те позиции, что входят в функцию потерь: реплики ассистента без
    промпта, результатов инструментов и блока размышления.
    """
    collator = ChatCollator(processor, system=system, tools=tools)
    model.eval()
    result: list[tuple[float, int]] = []
    for start in range(0, len(samples), batch_size):
        batch = collator(samples[start : start + batch_size])
        labels = batch.pop("labels").to(model.device)
        logits = model(**batch.to(model.device)).logits[:, :-1].float()
        targets = labels[:, 1:]
        mask = targets != IGNORE_INDEX
        token_logp = torch.log_softmax(logits, dim=-1).gather(
            -1, targets.clamp(min=0).unsqueeze(-1)
        ).squeeze(-1)
        total = (token_logp * mask).sum(dim=1)
        result += list(zip(total.tolist(), mask.sum(dim=1).tolist()))
    return result


def perplexity(model: Any, processor: Any, samples: Sequence[Sample], **kwargs: Any) -> float:
    """exp(средняя NLL на обучаемый токен) по отложенным примерам.

    Стандартная метрика SFT: насколько целевые ответы вероятны для модели.
    Меньше — лучше; сравнивать только на одних и тех же примерах.
    """
    pairs = sequence_logprobs(model, processor, samples, **kwargs)
    total_logp = sum(lp for lp, _ in pairs)
    total_tokens = sum(n for _, n in pairs)
    return math.exp(-total_logp / max(total_tokens, 1))


def preference_accuracy(
    model: Any,
    processor: Any,
    chosen: Sequence[Sample],
    rejected: Sequence[Sample],
    **kwargs: Any,
) -> dict[str, float]:
    """Доля пар, где выбранный ответ вероятнее отвергнутого, и средний зазор.

    То, что trl логирует при обучении как `rewards/accuracies`
    и `rewards/margins`, только без опорной модели и на отложенных парах.
    Зазор считается на токен, чтобы длина ответа не решала сравнение.
    """
    good = sequence_logprobs(model, processor, chosen, **kwargs)
    bad = sequence_logprobs(model, processor, rejected, **kwargs)
    margins = [g / max(gn, 1) - b / max(bn, 1) for (g, gn), (b, bn) in zip(good, bad)]
    return {
        "accuracy": sum(m > 0 for m in margins) / max(len(margins), 1),
        "margin": sum(margins) / max(len(margins), 1),
        "n": len(margins),
    }
