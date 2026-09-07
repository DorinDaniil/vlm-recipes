"""Настройки в одном месте. Всё читается из переменных окружения,
так что менять их можно не трогая код."""

from __future__ import annotations

import os
from dataclasses import dataclass, field


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _env_int(name: str, default: int) -> int:
    return int(raw) if (raw := os.environ.get(name)) else default


@dataclass(slots=True)
class Settings:
    #: Обычный bf16-репозиторий. Суффиксы -FP8, -NVFP4, -AWQ, -GPTQ
    #: означают квантованный чекпоинт — для обучения он не годится.
    model_id: str = field(default_factory=lambda: _env("VLM_MODEL_ID", "Qwen/Qwen3.5-9B"))

    #: Потолок визуальных токенов на картинку: 1003520 ≈ 1280 патчей.
    #: Страница A4 в полном разрешении даёт тысячи токенов и съедает контекст.
    max_pixels: int = field(default_factory=lambda: _env_int("VLM_MAX_PIXELS", 1_003_520))

    output_dir: str = field(default_factory=lambda: _env("VLM_OUTPUT_DIR", "runs"))
    max_new_tokens: int = field(default_factory=lambda: _env_int("VLM_MAX_NEW_TOKENS", 1024))


#: Импортируйте этот экземпляр, а не создавайте свой.
settings = Settings()
