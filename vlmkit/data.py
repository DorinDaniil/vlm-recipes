"""Формат данных и сборка батчей.

Единый формат — список реплик. Простые пары «вопрос-ответ», многоходовые
диалоги и агентские траектории с вызовами инструментов описываются им
одинаково, и коллатор для всех один.

JSONL, по примеру на строку. Две формы, обе понимаются автоматически:

    {"question": "...", "answer": "...", "context": "...", "image": "..."}
    {"messages": [{"role": "user", ...}, {"role": "assistant", ...}]}
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch
from PIL import Image

#: PyTorch пропускает эти позиции при подсчёте потерь.
IGNORE_INDEX = -100

#: Границы реплики ассистента в шаблоне Qwen. Обучаемся только внутри них.
QWEN_ASSISTANT_OPEN = "<|im_start|>assistant\n"
QWEN_TURN_CLOSE = "<|im_end|>"
QWEN_IMAGE_TOKENS = ("<|image_pad|>", "<|vision_start|>", "<|vision_end|>")

#: Блок размышления у рассуждающих моделей.
QWEN_THINK_OPEN = "<think>"
QWEN_THINK_CLOSE = "</think>"


@dataclass(slots=True)
class Sample:
    """Пример: последовательность реплик плюс картинки, если есть.

    Роли: system, user, assistant, tool. Роль tool несёт результат вызова
    инструмента — она приходит извне, и обучаться на ней нельзя.
    """

    messages: list[dict[str, Any]]
    images: list[str | Path] = field(default_factory=list)

    @classmethod
    def from_qa(
        cls,
        question: str,
        answer: str,
        *,
        context: str | None = None,
        system: str | None = None,
        image: str | Path | None = None,
    ) -> Sample:
        """Собрать пример из пары. Контекст идёт перед запросом."""
        text = f"{context}\n\n{question}" if context else question
        content: list[dict[str, Any]] = []
        if image is not None:
            content.append({"type": "image"})
        content.append({"type": "text", "text": text})

        messages: list[dict[str, Any]] = []
        if system:
            messages.append({"role": "system", "content": [{"type": "text", "text": system}]})
        messages.append({"role": "user", "content": content})
        messages.append({"role": "assistant", "content": [{"type": "text", "text": answer}]})
        return cls(messages, [image] if image is not None else [])

    def load_images(self) -> list[Image.Image]:
        return [Image.open(p).convert("RGB") for p in self.images]

    def with_system(self, system: str | None) -> Sample:
        """Копия с подставленным системным промптом.

        Он должен совпадать с боевым, иначе обучение и применение разойдутся.
        """
        if not system or self.messages[0].get("role") == "system":
            return self
        head = {"role": "system", "content": [{"type": "text", "text": system}]}
        return Sample([head, *self.messages], self.images)


def load_jsonl(path: str | Path) -> list[Sample]:
    """Прочитать датасет. Обе формы записи понимаются автоматически."""
    samples: list[Sample] = []
    with Path(path).open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            if not (line := line.strip()):
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}, строка {number}: {exc}") from exc

            if "messages" in record:
                samples.append(Sample(record["messages"], record.get("images", [])))
            else:
                samples.append(
                    Sample.from_qa(
                        record["question"],
                        record["answer"],
                        context=record.get("context"),
                        image=record.get("image"),
                    )
                )
    return samples


def save_jsonl(samples: Iterable[Sample], path: str | Path) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        for s in samples:
            record = {"messages": s.messages, "images": [str(i) for i in s.images]}
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _find_subsequence(haystack: list[int], needle: list[int], start: int) -> int:
    """Индекс первого вхождения начиная с `start`, или -1."""
    if not needle:
        return -1
    for i in range(start, len(haystack) - len(needle) + 1):
        if haystack[i : i + len(needle)] == needle:
            return i
    return -1


class ChatCollator:
    """Список примеров → батч тензоров.

    Обучаемся только внутри реплик ассистента. Всё прочее скрыто от функции
    потерь: промпт пользователя, системная инструкция, служебные токены
    картинки и — существенно для агентских данных — результаты вызовов
    инструментов. Модель, обученная предсказывать вывод инструмента, начнёт
    выдумывать содержимое документов вместо того, чтобы их запрашивать.

    Размечаются ВСЕ реплики ассистента, а не только последняя: в
    многоходовом диалоге и в траектории их несколько.
    """

    def __init__(
        self,
        processor: Any,
        *,
        system: str | None = None,
        mask_prompt: bool = True,
        mask_thinking: bool = True,
        assistant_open: str = QWEN_ASSISTANT_OPEN,
        turn_close: str = QWEN_TURN_CLOSE,
        think_open: str = QWEN_THINK_OPEN,
        think_close: str = QWEN_THINK_CLOSE,
        image_tokens: Sequence[str] = QWEN_IMAGE_TOKENS,
    ) -> None:
        self.processor = processor
        self.system = system
        self.mask_prompt = mask_prompt

        #: Скрывать блок размышления внутри реплики ассистента.
        #:
        #: У рассуждающей модели шаблон вставляет `<think>` даже в те
        #: реплики, где рассуждения нет, — получается пустой блок.
        #: Обучаясь на нём, модель усваивает «размышлять не нужно»,
        #: и дообучение под правила поведения заодно отучает её думать.
        #:
        #: Маскирование оставляет вопрос открытым: градиент не идёт ни
        #: за рассуждение, ни против. Выключите, если рассуждения есть
        #: в самих данных и вы хотите им учить.
        self.mask_thinking = mask_thinking

        tokenizer = processor.tokenizer
        self.open_ids = tokenizer.encode(assistant_open, add_special_tokens=False)
        self.close_ids = tokenizer.encode(turn_close, add_special_tokens=False)
        self.think_open_ids = tokenizer.encode(think_open, add_special_tokens=False)
        self.think_close_ids = tokenizer.encode(think_close, add_special_tokens=False)
        self.image_token_ids = [
            tid
            for token in image_tokens
            if (tid := tokenizer.convert_tokens_to_ids(token)) is not None and tid >= 0
        ]

    def _unmask_assistant(self, labels: torch.Tensor, row: int) -> None:
        """Скрыть всё, затем открыть каждую реплику ассистента."""
        ids = labels[row].tolist()
        masked = torch.full_like(labels[row], IGNORE_INDEX)

        cursor, found = 0, False
        while (start := _find_subsequence(ids, self.open_ids, cursor)) != -1:
            body = start + len(self.open_ids)
            end = _find_subsequence(ids, self.close_ids, body)
            end = len(ids) if end == -1 else end + len(self.close_ids)
            masked[body:end] = labels[row][body:end]
            cursor, found = end, True

        if not found:
            raise ValueError(
                "Не найдена ни одна реплика ассистента: шаблон модели отличается "
                "от ожидаемого. Задайте assistant_open или отключите mask_prompt."
            )

        if self.mask_thinking:
            self._hide_thinking(masked, ids)
        labels[row] = masked

    def _hide_thinking(self, masked: torch.Tensor, ids: list[int]) -> None:
        """Закрыть блоки размышления обратно.

        Вызывается после того, как реплики ассистента открыты: блок
        `<think>` лежит внутри них и иначе попал бы в градиент.
        Незакрытый блок скрывается до конца последовательности —
        обрыв означает, что ответа в примере нет вовсе.
        """
        cursor = 0
        while (start := _find_subsequence(ids, self.think_open_ids, cursor)) != -1:
            end = _find_subsequence(ids, self.think_close_ids, start)
            end = len(ids) if end == -1 else end + len(self.think_close_ids)
            masked[start:end] = IGNORE_INDEX
            cursor = end

    def __call__(self, samples: Sequence[Sample]) -> dict[str, torch.Tensor]:
        prepared = [s.with_system(self.system) for s in samples]
        texts = [
            self.processor.apply_chat_template(s.messages, tokenize=False)
            for s in prepared
        ]
        # Плоский список: процессор сопоставляет картинки с плейсхолдерами
        # по порядку во всём батче.
        images = [img for s in prepared for img in s.load_images()]

        kwargs: dict[str, Any] = {"text": texts, "return_tensors": "pt", "padding": True}
        if images:
            kwargs["images"] = images
        batch = self.processor(**kwargs)

        labels = batch["input_ids"].clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = IGNORE_INDEX
        for token_id in self.image_token_ids:
            labels[labels == token_id] = IGNORE_INDEX
        if self.mask_prompt:
            for row in range(labels.shape[0]):
                self._unmask_assistant(labels, row)

        batch["labels"] = labels
        return batch


def describe(samples: Sequence[Sample], processor: Any = None) -> str:
    """Сводка перед обучением.

    Смотрите на долю обучаемых токенов: если она меньше нескольких процентов,
    почти весь батч уходит впустую и стоит поднять размер примеров или
    включить упаковку.
    """
    if not samples:
        return "датасет пуст"

    turns = [len(s.messages) for s in samples]
    with_images = sum(bool(s.images) for s in samples)
    lines = [
        f"примеров {len(samples)}, с картинками {with_images}",
        f"реплик на пример: медиана {sorted(turns)[len(turns) // 2]}, максимум {max(turns)}",
    ]

    if processor is not None:
        collator = ChatCollator(processor)
        batch = collator(samples[: min(8, len(samples))])
        total = batch["labels"].numel()
        trained = int((batch["labels"] != IGNORE_INDEX).sum())
        lines.append(f"обучаемых токенов {trained}/{total} ({trained / total:.0%})")

    return "\n".join(lines)


def preview(sample: Sample, processor: Any, *, system: str | None = None) -> str:
    """Показать, на чём модель учится в этом примере.

    Обучаемые токены выделены ⟦скобками⟧, остальное скрыто от функции
    потерь. Самая быстрая проверка коллатора из существующих: если внутри
    скобок оказался вопрос пользователя или вывод инструмента —
    маскирование сломано, и обучение пойдёт не туда.

        >>> print(preview(samples[0], processor))
    """
    collator = ChatCollator(processor, system=system)
    batch = collator([sample])
    ids = batch["input_ids"][0].tolist()
    labels = batch["labels"][0].tolist()

    parts, trained = [], False
    for token_id, label in zip(ids, labels):
        piece = processor.tokenizer.decode([token_id])
        is_trained = label != IGNORE_INDEX
        if is_trained != trained:
            parts.append("⟦" if is_trained else "⟧")
            trained = is_trained
        parts.append(piece)
    if trained:
        parts.append("⟧")

    share = sum(l != IGNORE_INDEX for l in labels) / len(labels)
    return f"{''.join(parts)}\n\n[обучаемых токенов {share:.0%} из {len(labels)}]"
