"""Диалог с моделью. Модель грузится один раз и живёт в объекте.

История хранится в пределах сессии и умирает вместе с объектом:
долговременной памяти здесь нет намеренно.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import torch
from PIL import Image

from vlmkit.config import settings

ImageInput = str | Path | Image.Image | None


def load_image(image: ImageInput) -> Image.Image | None:
    """Привести к PIL RGB. Конвертация обязательна: PNG с альфой
    и одноканальные сканы иначе роняют препроцессор."""
    if image is None:
        return None
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    return Image.open(image).convert("RGB")


@dataclass(slots=True)
class Turn:
    """Реплика диалога."""

    role: str
    text: str
    image: Image.Image | None = None

    def to_message(self) -> dict[str, Any]:
        content: list[dict[str, Any]] = []
        if self.image is not None:
            content.append({"type": "image", "image": self.image})
        content.append({"type": "text", "text": self.text})
        return {"role": self.role, "content": content}


class VLMChat:
    """Диалог с моделью.

        >>> chat = VLMChat(model, processor, system="Отвечай кратко.")
        >>> chat.ask("Что здесь написано?", image="scan.jpg")

    `guard` проверяет запрос до вызова модели, см. `vlmkit.guardrails`.
    `keep_history=False` для независимых задач: история только занимает
    контекст и путает модель. `thinking=False` выключает блок рассуждения:
    с ним лимит токенов уходит в размышление, а ответ обрезается.
    """

    def __init__(
        self,
        model: Any,
        processor: Any,
        *,
        system: str | None = None,
        guard: Any = None,
        keep_history: bool = True,
        max_new_tokens: int = settings.max_new_tokens,
        thinking: bool = False,
    ) -> None:
        self.model = model
        self.processor = processor
        self.system = system
        self.guard = guard
        self.keep_history = keep_history
        self.max_new_tokens = max_new_tokens
        self.thinking = thinking
        self.history: list[Turn] = []

    def _messages(self, turn: Turn) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = []
        if self.system:
            messages.append(
                {"role": "system", "content": [{"type": "text", "text": self.system}]}
            )
        messages.extend(t.to_message() for t in self.history)
        messages.append(turn.to_message())
        return messages

    def _encode(self, turn: Turn) -> dict[str, Any]:
        """`add_generation_prompt` дописывает токены «говорит ассистент».
        Без них модель продолжает реплику пользователя вместо ответа."""
        inputs = self.processor.apply_chat_template(
            self._messages(turn),
            add_generation_prompt=True,
            enable_thinking=self.thinking,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        return inputs.to(self.model.device)

    def _sampling(self, temperature: float) -> dict[str, Any]:
        """temperature=0 — жадный выбор, результат воспроизводим.
        Для свободного диалога 0.6–0.8."""
        if temperature <= 0:
            return {"do_sample": False}
        return {"do_sample": True, "temperature": temperature, "top_p": 0.9}

    def ask(
        self,
        text: str,
        image: ImageInput = None,
        *,
        temperature: float = 0.0,
        max_new_tokens: int | None = None,
    ) -> str:
        """Задать вопрос и получить ответ целиком.

        Заблокированный `guard` запрос возвращает текст отказа, модель
        при этом не вызывается.
        """
        if self.guard is not None:
            verdict = self.guard.check(text, image=image)
            if not verdict.allowed:
                return verdict.refusal_text()

        turn = Turn("user", text, load_image(image))
        inputs = self._encode(turn)

        with torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens or self.max_new_tokens,
                **self._sampling(temperature),
            )

        # generate возвращает промпт вместе с ответом — обрезаем.
        answer = self.processor.decode(
            output[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True
        ).strip()

        if self.keep_history:
            self.history += [turn, Turn("assistant", answer)]
        return answer

    def stream(
        self,
        text: str,
        image: ImageInput = None,
        *,
        temperature: float = 0.0,
        max_new_tokens: int | None = None,
    ) -> Iterator[str]:
        """То же, что `ask`, но кусками по мере генерации.

        `generate` блокирует до конца, поэтому он уходит в отдельный поток,
        а мы читаем из очереди.
        """
        if self.guard is not None:
            verdict = self.guard.check(text, image=image)
            if not verdict.allowed:
                yield verdict.refusal_text()
                return

        from transformers import TextIteratorStreamer

        turn = Turn("user", text, load_image(image))
        inputs = self._encode(turn)
        streamer = TextIteratorStreamer(
            self.processor.tokenizer, skip_prompt=True, skip_special_tokens=True
        )

        thread = threading.Thread(
            target=self.model.generate,
            kwargs={
                **inputs,
                "streamer": streamer,
                "max_new_tokens": max_new_tokens or self.max_new_tokens,
                **self._sampling(temperature),
            },
        )
        thread.start()

        chunks: list[str] = []
        for chunk in streamer:
            chunks.append(chunk)
            yield chunk
        thread.join()

        if self.keep_history:
            self.history += [turn, Turn("assistant", "".join(chunks).strip())]

    def reset(self) -> None:
        """Забыть диалог. Системный промпт сохраняется."""
        self.history.clear()
