# vlm-recipes

Comparing fine-tuning methods on one product task: an assistant for a
student writing a thesis in a document editor. A fragment of the document
is open next to the chat, the student asks for something, the model
answers with one message.

There is no agent here on purpose. What is tuned is how the model writes
and where it draws the line, because that is what carries over into any
agent wrapper built later.

Methods: LoRA SFT, preference training with DPO, ORPO, SimPO and KTO, and
a steering vector. The base model in the examples is Qwen3.5-9B; nothing in
the code is tied to it.

## Layout

    notebooks/    the experiments, one method per notebook, Russian narration
    src/          data access, deterministic checks, run files and charts
    data/         raw situations and the rendered training files
    tools/        build the data files, validate them
    runs/         one json per run: metrics and every answer
    books/        theory as PDF, sources in books/tex
    docs/         benchmark notes

Model loading, generation, the judge, LoRA, every trainer and the steering
hooks live in notebook cells. `src` holds only what would be noise in a
cell and never imports torch, so `01_data` and `06_results` open on any
machine.

## Install

    pip install -r requirements.txt

`torch` is installed separately for the CUDA build your driver supports.

    python tools/build_data.py    # data/raw/*.jsonl -> data/*.jsonl
    python tools/check_data.py    # validate without a model

## Notebooks

| | |
|---|---|
| `01_data` | the task, the data format, reference against bad answer, what the checks measure |
| `02_baseline` | the base model: generation, judge, metric formulas, judge agreement with people |
| `03_sft` | SFT loss and LoRA, mask verified on a real example, before and after |
| `04_preference` | DPO, ORPO, SimPO and KTO with their objectives, one loop over methods |
| `05_steering` | a steering vector from the same pairs, swept by strength |
| `06_results` | every run side by side, no model required |
| `07_playground` | hand-typed requests against the base model and every saved adapter, streaming, steering on top |

Run them in order. Each experiment writes `runs/<method>-<test>.json` with
the same metric names and saves its adapter to `runs/<method>-adapter`, which
is what `07_playground` picks up.

## Data

`data/raw/*.jsonl` is one situation per line the way a person writes it:
document id, earlier turns, the student's request, a reference answer and
a bad one, assigned checks, judge criteria. `tools/build_data.py` renders
it into the conversational preference format of TRL:

    prompt      list of chat messages, system and document included, ending with the student
    chosen      one assistant message, the reference answer
    rejected    one assistant message, a competent answer that breaks exactly one rule

The same rows feed every method: SFT reads `prompt` and `chosen`,
preference trainers read all three, KTO unpairs them.

| split | rows | what it is |
|---|---|---|
| train | 210 | training situations across fifteen subject areas |
| dev | 47 | held out from the same source, for perplexity and preference accuracy |
| test_product | 33 | the product's golden set: rubric, production answer, human verdict |
| test_extended | 100 | our test on ten unseen documents, with false-refusal traps, empty documents, statistics errors and out-of-scope requests |

Documents used in training never appear in either test.

## Metrics

| | |
|---|---|
| judge | share of answers the LLM judge accepts against the rubric |
| checks_all | share of situations where every assigned check passed |
| checks | mean share of assigned checks that passed |
| refusal | share of must-refuse situations that got a refusal |
| false_refusal | share of ordinary requests whose answer opened with a refusal |
| length | mean answer length in characters |

Shares come with a 95 % Wilson interval, roughly ±16 points on the product
test and ±10 on the extended one. A shift smaller than that is not a
result: on 33 rows one situation is already three points.

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
