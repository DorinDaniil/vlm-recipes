# Бенчмарки

Что измерять при выборе модели для агента с инструментами и где
смотреть. Данные на середину 2026 года; лидерборды обновляются,
адреса проверять.

## Что важно

| свойство | бенчмарк |
|---|---|
| вызов инструмента при необходимости | BFCL: simple, multiple, parallel |
| отказ от вызова без необходимости | BFCL: irrelevance detection |
| соблюдение правил на длинной траектории | τ-bench, pass^k |
| соблюдение формата ответа | IFEval, strict |

Общий балл лидерборда усредняет несравнимые категории. Смотреть
подкатегории. Вторая строка соответствует группе `direct`
в `data/tools.jsonl`.

## Вызов инструментов

**BFCL** — Berkeley Function Calling Leaderboard.
<https://gorilla.cs.berkeley.edu/leaderboard.html>
Категории: simple, multiple, parallel, multi-turn (с v3), irrelevance
detection. Разброс по irrelevance между моделями с близким общим
баллом достигает 2×.

**τ-bench.** <https://github.com/sierra-research/tau-bench>
Агент в домене (retail, airline) с инструментами и политикой поведения.
Метрика pass^k — доля задач, решённых во всех k попытках. Смотреть k = 4.
τ²-bench — расширение с двусторонним диалогом.

**ToolBench** <https://github.com/OpenBMB/ToolBench>, **API-Bank**,
**NexusRaven** <https://huggingface.co/Nexusflow> — более ранние,
частично насыщены.

**MINT** — многоходовое взаимодействие с инструментами и обратной
связью.

## Следование инструкциям

**IFEval.** <https://arxiv.org/abs/2311.07911>
Инструкции с проверяемыми ограничениями («ровно три абзаца», «ответ
в JSON»). Метрика prompt-level strict accuracy. Входит в Open LLM
Leaderboard.

**FollowBench**, **InfoBench** — многоуровневые ограничения.

## Агентские среды

**AgentBench** <https://github.com/THUDM/AgentBench>,
**GAIA** <https://huggingface.co/spaces/gaia-benchmark/leaderboard>,
**WebArena**, **OSWorld**.

## VLM

**OpenVLM Leaderboard.**
<https://huggingface.co/spaces/opencompass/open_vlm_leaderboard>
Для документов: DocVQA, ChartQA, InfoVQA, OCRBench. MMMU, MathVista —
общее визуальное рассуждение.

## Дашборды

| | | |
|---|---|---|
| Open LLM Leaderboard | <https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard> | открытые модели, воспроизводимо, IFEval |
| LMArena | <https://lmarena.ai> | парные сравнения людьми, Elo |
| Artificial Analysis | <https://artificialanalysis.ai> | скорость, цена |
| LiveBench | <https://livebench.ai> | обновляемые задания |
| OpenCompass | <https://rank.opencompass.org.cn> | детально по Qwen |

## Модели с сильным вызовом инструментов

Общего назначения с нативной поддержкой инструментов: Qwen 2.5 / 3.x,
Llama 3.x (70B), GLM-4, DeepSeek-V3, Mistral.

Специализированные под вызов функций: xLAM (Salesforce), Hermes
(Nous Research), ToolACE, Functionary (MeetKai). На BFCL выше общих
моделей того же размера; в свободном диалоге слабее.

## Ограничения

- Контаминация: ранние бенчмарки присутствуют в обучающих данных.
  IFEval и BFCL устойчивее за счёт процедурно генерируемых проверок;
  LiveBench обновляется намеренно.
- Общий балл скрывает разброс по категориям.
- Сравнивать модели одного размера.
- Ни один бенчмарк не покрывает конкретную политику поведения.
  Финальный выбор — собственный замер: `tools/selfcheck.py`,
  `notebooks/`.

## Порядок отбора

1. BFCL, категория irrelevance
2. τ-bench, pass^4
3. IFEval, strict
4. Для VLM — DocVQA, OCRBench
5. Собственный замер на своих данных
