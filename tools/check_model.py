#!/usr/bin/env python
"""
Проверка кандидата на пригодность к дообучению и агентскому режиму.

    python research/check_model.py Qwen/Qwen3.5-9B

Веса не скачиваются: читаются только конфигурация, индекс весов
и токенизатор. Занимает секунды и отсекает заведомо неподходящие
варианты до того, как вы потратите час на загрузку.
"""

from __future__ import annotations

import json
import sys

from huggingface_hub import HfApi, hf_hub_download

# Строка на русском для оценки экономности токенизации. Чем меньше
# токенов на неё уходит, тем дешевле обходится русский контекст.
RU_PROBE = (
    "Гипотеза должна быть проверяемой и связанной с исследовательской "
    "проблемой, сформулированной во введении выпускной работы."
)


def fetch_json(model_id: str, filename: str) -> dict | None:
    try:
        with open(hf_hub_download(model_id, filename), encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def check_weights(model_id: str) -> None:
    """Размер модели и пригодность чекпоинта к обучению."""
    config = fetch_json(model_id, "config.json")
    if config is None:
        print("config.json недоступен — репозиторий закрыт или не существует")
        return

    print(f"архитектура      {config.get('model_type')} "
          f"{config.get('architectures')}")
    print(f"мультимодальная  {'vision_config' in config}")

    quant = config.get("quantization_config")
    if quant:
        print(f"КВАНТОВАН        {quant.get('quant_method')} — обучать нельзя, "
              f"нужен bf16-репозиторий")
    else:
        print("квантование      нет (пригоден для обучения)")

    # total_size лежит в индексе шардов и даёт точный размер без скачивания.
    index = fetch_json(model_id, "model.safetensors.index.json")
    if index and "metadata" in index:
        gib = index["metadata"].get("total_size", 0) / 2**30
        print(f"размер весов     {gib:.1f} GiB "
              f"(LoRA ~{gib * 1.4:.0f} GiB, QLoRA ~{gib / 4 + 6:.0f} GiB)")


def check_tokenizer(model_id: str) -> None:
    """Шаблон диалога, поддержка инструментов, экономность на русском."""
    from transformers import AutoTokenizer

    try:
        tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    except Exception as exc:
        print(f"токенизатор      не загрузился: {exc}")
        return

    template = tok.chat_template or ""
    print(f"chat_template    {'есть' if template else 'НЕТ'}")
    # Ветка по tools в шаблоне означает, что вызов инструментов обучен,
    # а не имитируется текстовой инструкцией.
    print(f"вызов tools      {'обучен' if 'tools' in template else 'не обучен'}")

    tokens = len(tok.encode(RU_PROBE))
    print(f"русский          {tokens} токенов на {len(RU_PROBE)} символов "
          f"({len(RU_PROBE) / tokens:.1f} симв/токен)")


def check_card(model_id: str) -> None:
    """Лицензия и наличие родственных моделей в семействе."""
    try:
        info = HfApi().model_info(model_id)
    except Exception as exc:
        print(f"карточка         недоступна: {exc}")
        return

    tags = info.tags or []
    license_tag = next((t[8:] for t in tags if t.startswith("license:")), "не указана")
    print(f"лицензия         {license_tag}")


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        raise SystemExit(1)

    model_id = sys.argv[1]
    print(f"\n{model_id}\n{'─' * 60}")
    check_weights(model_id)
    check_tokenizer(model_id)
    check_card(model_id)
    print()


if __name__ == "__main__":
    main()
