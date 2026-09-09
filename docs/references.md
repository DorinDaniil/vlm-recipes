# Литература

Статьи, на которых стоят методы репозитория, по одной строке о том, зачем
каждая здесь. Теория подробнее в `books/`: `00-basics` про авторегрессию,
chat template и вызов инструментов, `01-model-choice` про выбор модели,
`02-sft-math` про LoRA и его варианты, `03-alignment` про RLHF, DPO
и семейство, `slides` для презентации.

## Адаптеры

- Hu et al. LoRA: Low-Rank Adaptation of Large Language Models. 2021.
  <https://arxiv.org/abs/2106.09685>. Обновление $W + BA$ малого ранга;
  то, что учится во всех ноутбуках.
- Kalajdzievski. A Rank Stabilization Scaling Factor for Fine-Tuning with
  LoRA. 2023. <https://arxiv.org/abs/2312.03732>. Множитель $\alpha/\sqrt{r}$
  вместо $\alpha/r$, флаг `use_rslora`.
- Dettmers et al. QLoRA: Efficient Finetuning of Quantized LLMs. 2023.
  <https://arxiv.org/abs/2305.14314>. Адаптер поверх 4-битной базы, запас
  на случай нехватки памяти.
- Liu et al. DoRA: Weight-Decomposed Low-Rank Adaptation. 2024.
  <https://arxiv.org/abs/2402.09353>. Разложение на норму и направление,
  разобрано в `02-sft-math`.
- Zhao et al. GaLore: Memory-Efficient LLM Training by Gradient Low-Rank
  Projection. 2024. <https://arxiv.org/abs/2403.03507>. Низкий ранг
  у градиента, а не у весов; альтернатива адаптерам.

## Обучение на предпочтениях

- Bradley, Terry. Rank Analysis of Incomplete Block Designs: The Method of
  Paired Comparisons. Biometrika 39, 1952. Модель $\sigma(r_w - r_l)$,
  из которой выводятся все методы ниже.
- Ouyang et al. Training Language Models to Follow Instructions with Human
  Feedback. 2022. <https://arxiv.org/abs/2203.02155>. RLHF, точка отсчёта.
- Rafailov et al. Direct Preference Optimization: Your Language Model is
  Secretly a Reward Model. 2023. <https://arxiv.org/abs/2305.18290>.
  Награда как $\beta\log\frac{\pi_\theta}{\pi_{\text{ref}}}$, без RL.
- Hong, Lee, Thorne. ORPO: Monolithic Preference Optimization without
  Reference Model. 2024. <https://arxiv.org/abs/2403.07691>. SFT-лосс плюс
  штраф на отношение шансов, референс не нужен.
- Xu et al. Contrastive Preference Optimization. 2024.
  <https://arxiv.org/abs/2401.08417>. CPO; в trl SimPO реализован
  как режим `CPOTrainer`.
- Meng, Xia, Chen. SimPO: Simple Preference Optimization with a
  Reference-Free Reward. 2024. <https://arxiv.org/abs/2405.14734>.
  Средний лог-правдоподобия на токен и порог $\gamma$.
- Ethayarajh et al. KTO: Model Alignment as Prospect Theoretic Optimization.
  2024. <https://arxiv.org/abs/2402.01306>. Обучение без пар, функция
  ценности из Kahneman, Tversky. Prospect Theory: An Analysis of Decision
  under Risk. Econometrica 47, 1979.

## Векторы активаций

- Turner et al. Steering Language Models with Activation Engineering. 2023.
  <https://arxiv.org/abs/2308.10248>. Прибавление разности активаций
  к скрытым состояниям при генерации.
- Zou et al. Representation Engineering: A Top-Down Approach to AI
  Transparency. 2023. <https://arxiv.org/abs/2310.01405>. Линейные
  направления понятий в активациях, чтение и управление.
- Arditi et al. Refusal in Language Models Is Mediated by a Single
  Direction. 2024. <https://arxiv.org/abs/2406.11717>. Разность средних
  как направление отказа; та же конструкция в `05_steering`.

## Оценка

- Zhou et al. Instruction-Following Evaluation for Large Language Models.
  2023. <https://arxiv.org/abs/2311.07911>. IFEval: проверяемые кодом
  правила; наши автопроверки формы устроены так же.
- Zheng et al. Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena.
  2023. <https://arxiv.org/abs/2306.05685>. Судья-модель и его смещения,
  почему согласие с людьми надо замерять.
- Yao et al. τ-bench: A Benchmark for Tool-Agent-User Interaction in
  Real-World Domains. 2024. <https://arxiv.org/abs/2406.12045>. Метрика
  pass^k и проверка по состоянию базы.
- Chen et al. Evaluating Large Language Models Trained on Code. 2021.
  <https://arxiv.org/abs/2107.03374>. Несмещённая оценка pass@k.
- Wilson. Probable Inference, the Law of Succession, and Statistical
  Inference. JASA 22, 1927. Интервал для доли, которым снабжены все наши
  метрики.
- Berkeley Function Calling Leaderboard.
  <https://gorilla.cs.berkeley.edu/leaderboard.html>. AST-сравнение
  вызовов, колонки irrelevance и relevance.

Актуальные лидерборды и то, что они меряют, в `benchmarks.md`.
