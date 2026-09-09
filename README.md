# vlm-recipes

Comparing fine-tuning methods on two product tasks of a thesis-writing
assistant. Methods: LoRA SFT, preference training with DPO, ORPO, SimPO and
KTO, and a steering vector. The base model in the examples is Qwen3.5-9B;
nothing in the code is tied to it.

**Filter** is the primary track. A request filter reads the student's message
and the open document fragment and answers with one line: `PASS`, or `BLOCK`
plus one of five categories of academic dishonesty. Short answers, exact
metrics, no judge: accuracy, block recall, false blocks on lookalike traps,
category accuracy. Differences between methods are visible at a glance.

**Assistant** is the harder track kept for later: the full answer to the
student, scored by deterministic checks and an LLM judge against the
product's rubrics.

## Layout

    notebooks/filter/      01_data … 06_results, the primary track
    notebooks/assistant/   01_data … 07_playground, the long-answer track
    src/                   data, metrics, infer, report (assistant), filter (filter task)
    data/filter/           raw.jsonl and the rendered train / dev / test
    data/                  the assistant's raw situations and rendered splits
    tools/                 build and validate data, rescore saved runs
    runs/filter, runs/assistant   one json per run: metrics and every answer
    books/, docs/          theory as PDF, benchmark notes, references

Training is written out in the cells: LoRA config, the trainer, the loop over
preference methods, the steering hook. What repeats sits in `src`:
`infer.generate` and `infer.judge` are thin wrappers over `apply_chat_template`
and `generate`; `filter.evaluate` / `report.evaluate` score a model and write
the run; `filter.show` / `report.show` print one table with the change against
the base in brackets.

## Install

    pip install -r requirements.txt

`torch` is installed separately for the CUDA build your driver supports.
Data files are committed; to rebuild them from the raw situations:

    python tools/build_filter.py   # data/filter/raw.jsonl -> train / dev / test
    python tools/build_data.py     # assistant: data/raw/*.jsonl -> data/*.jsonl
    python tools/check_data.py     # assistant data validation, no model
    python tools/rescore.py        # assistant: recompute saved runs after a check changes

## Filter track

| | |
|---|---|
| `01_data` | the task, the one-line format, cases decided by the document, the traps |
| `02_baseline` | base model: metrics, false blocks and missed violations listed |
| `03_sft` | LoRA SFT on the labels, mask check, errors that remain |
| `04_preference` | DPO, ORPO, SimPO, KTO on top of the SFT adapter |
| `05_steering` | a block-direction vector from BLOCK rows, swept by strength |
| `06_results` | table, recall against false blocks, recall per category, errors of the best method |

Data: 303 situations, 164 PASS and 139 BLOCK, 65 lookalike traps, 78 with a
short document fragment; split 162 / 40 / 101. Every row is `prompt`,
`chosen`, `rejected` in the TRL conversational format, so SFT, preference
trainers and the steering vector read the same file. Wilson interval on the
test is about ±10 points.

| metric | what it counts | better |
|---|---|---|
| accuracy | right PASS / BLOCK label | up |
| block_recall | violations labelled BLOCK | up |
| false_block | legitimate requests labelled BLOCK | down |
| trap_false_block | the same on lookalike traps only | down |
| category_acc | right category among correctly blocked rows | up |
| format_ok | answer starts with PASS or BLOCK | up |
| perplexity, pref_acc | reference likelihood and pair preference on dev | down / up |

## Assistant track

Same notebooks and the same `src` conventions on the long-answer task: 210
training situations, a 33-row product golden set and a 100-row extended
test, deterministic checks plus an LLM judge, `report.evaluate` / `report.show`.
Details in the notebooks; benchmarks and references in `docs/`.

## Reading

| | |
|---|---|
| [books/00-basics.pdf](books/00-basics.pdf) | autoregression, chat template, prefill, tool calling, the `<think>` block |
| [books/01-model-choice.pdf](books/01-model-choice.pdf) | choosing a base model, benchmarks, model families |
| [books/02-sft-math.pdf](books/02-sft-math.pdf) | LoRA, rsLoRA, DoRA, QLoRA, GaLore |
| [books/03-alignment.pdf](books/03-alignment.pdf) | RLHF, the DPO derivation, ORPO, SimPO, KTO, activation vectors |
| [books/slides.pdf](books/slides.pdf) | slide deck on SFT and alignment |
| [docs/benchmarks.md](docs/benchmarks.md) | current tool-use and agent benchmarks, what each metric computes |
| [docs/references.md](docs/references.md) | the papers behind every method in the notebooks |

## License

MIT
