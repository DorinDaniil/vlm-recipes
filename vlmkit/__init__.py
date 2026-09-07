"""vlmkit — вспомогательные части для дообучения VLM.

Здесь нет обёрток над обучением: `LoraConfig`, `Trainer` и тренеры TRL
вызываются напрямую в ноутбуке, иначе непонятно, что происходит.
В пакете лежит только то, что скучно писать каждый раз:

    model       загрузка, выгрузка памяти, отчёт по картам
    data        формат примеров и коллатор с маскированием
    evaluate    генерация на отложенной выборке и парные метрики
    compat      фильтр аргументов и поиск тренеров TRL по версии
    chat        диалог с моделью
    guardrails  фильтрация запросов без дообучения

Полный цикл обучения со всеми вызовами видно в
`research/experiments.ipynb`.
"""

from vlmkit import compat, evaluate, steering, toolcalls
from vlmkit.chat import VLMChat
from vlmkit.config import Settings, settings
from vlmkit.data import ChatCollator, Sample, describe, load_jsonl, preview, save_jsonl
from vlmkit.guardrails import (
    GuardPipeline,
    KeywordGuard,
    ModelJudgeGuard,
    OutputGuard,
    Verdict,
    refusal_system_prompt,
)
from vlmkit.model import LoadConfig, cleanup, free, load, memory_report

__all__ = [
    "ChatCollator",
    "GuardPipeline",
    "KeywordGuard",
    "LoadConfig",
    "ModelJudgeGuard",
    "OutputGuard",
    "Sample",
    "Settings",
    "VLMChat",
    "Verdict",
    "chat",
    "cleanup",
    "compat",
    "toolcalls",
    "describe",
    "evaluate",
    "free",
    "load",
    "load_jsonl",
    "memory_report",
    "preview",
    "refusal_system_prompt",
    "save_jsonl",
    "settings",
    "steering",
]

__version__ = "0.3.0"
