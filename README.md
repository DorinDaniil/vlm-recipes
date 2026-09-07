# vlm-recipes

Дообучение открытых LLM/VLM под агентные задачи: SFT (LoRA и варианты),
выравнивание по предпочтениям (DPO, ORPO, SimPO, KTO), управление
поведением через векторы активаций. Целевая модель в примерах —
Qwen3.5-9B; код к семейству не привязан.

## Структура

    books/        теория, PDF; исходники в books/tex
    notebooks/    эксперименты, один метод на ноутбук
    data/         обучающие и тестовые наборы, JSONL
    tools/        проверка модели, данных, окружения
    docs/         бенчмарки
    vlmkit/       коллатор, генерация, парсер вызовов, совместимость версий

## Установка

    pip install -r requirements.txt

`torch` устанавливается отдельно под версию CUDA, поддерживаемую
драйвером.

Проверка модели и данных:

    python tools/check_model.py Qwen/Qwen3.5-9B
    python tools/selfcheck.py

## Ноутбуки

Каждый ноутбук загружает модель, измеряет поведение до вмешательства,
применяет метод, измеряет после. Тестовые запросы и метрики общие,
в `notebooks/common.py`.

| | метод | данные |
|---|---|---|
| `00-setup` | — | окружение, маскирование, базовые метрики |
| `01-sft` | LoRA | policy.jsonl |
| `02-steering` | векторы активаций | policy.jsonl |
| `03-dpo` | ORPO / DPO / SimPO | prefs.jsonl |
| `04-tools` | LoRA, агентский цикл | tools.jsonl |
| `05-compare` | LoRA, rsLoRA, DoRA | policy + tools |

Вызовы `peft`, `transformers`, `trl` в ячейках без обёрток.

## Данные

| файл | n | разметка |
|---|---|---|
| `policy.jsonl` | 143 | clarify / guide / answer |
| `tools.jsonl` | 66 | tool / direct / stop / multi / empty |
| `prefs.jsonl` | 60 | chosen / rejected |
| `skills.jsonl` | 49 | 7 типов, формат проверяется регулярным выражением |

В каждом наборе есть контрольная группа. Метрики парные: доля
срабатываний на целевых группах и доля ложных срабатываний
на контрольных.

`tools.jsonl` генерируется `data/build_tools.py`. Формат вызова
инструмента (`STYLE`) должен совпадать с шаблоном модели; проверяется
`tools/selfcheck.py`.

## Книги

| | |
|---|---|
| [00-basics](books/00-basics.pdf) | авторегрессия, chat template, префилл, вызов инструментов, блок `<think>` |
| [01-model-choice](books/01-model-choice.pdf) | критерии выбора, бенчмарки, семейства моделей |
| [02-sft-math](books/02-sft-math.pdf) | LoRA, rsLoRA, DoRA, QLoRA/NF4, GaLore |
| [03-alignment](books/03-alignment.pdf) | RLHF, вывод DPO, ORPO, SimPO, KTO, векторы активаций |
| [slides](books/slides.pdf) | презентация, 29 слайдов |

Сборка: `xelatex` для книг, `pdflatex` для слайдов.

## vlmkit

- `data` — `Sample`, `ChatCollator` (маскирует промпт, результаты
  инструментов, блок `<think>`), `preview`
- `evaluate` — генерация с left padding, `Suite`, парные метрики
- `toolcalls` — разбор `<tool_call>` в JSON и XML
- `steering` — `SteeringVector`
- `compat` — фильтр аргументов `TrainingArguments` и конфигов TRL
  под установленную версию
- `model`, `chat`, `guardrails` — загрузка, диалог, фильтрация запросов

## Лицензия

MIT
