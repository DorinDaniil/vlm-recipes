# vlm-recipes

Teaching a small model one behaviour: help the student, but do not do the work for them.
Base model Qwen3.5-9B, Hugging Face `transformers`, `peft`, `trl`. The system prompt is
neutral, so the boundary has to come from training. Prompting, SFT, DPO and a steering
vector are compared on the same 120 test requests, half of which must be declined.

## Layout

    data/cases.jsonl     400 rows: split, decision, trap, topic, request, good, bad
    src/                 data, model, steering, evaluate, tools, chat
    train.py             python train.py sft | dpo | steer
    evaluate.py          python evaluate.py base | prompted | sft | dpo | steer [alpha ...]
    runs/                one json per run; adapters and vectors in runs/<name>/
    runs/judge/          verdicts of the external judge, one json per run
    notebooks/           data, results (no GPU), tools, chat
    docs/                theory as PDF with LaTeX sources, references

## Run

    pip install -r requirements.txt
    python evaluate.py base
    python evaluate.py prompted
    python train.py sft   && python evaluate.py sft
    python train.py dpo   && python evaluate.py dpo
    python train.py steer && python evaluate.py steer 0.5 1 2

Every method starts from the base model. `evaluate.py` saves the answers and prints
pref acc, perplexity of the good and bad references and answer length.

## Judge

Whether an answer helped or declined is decided by Claude Fable 5.1 reading every answer,
not by a model. `runs/judge/<name>.json` holds a verdict (`help`, `decline`, `neither`) and a
one-line comment per row. `results.ipynb` turns them into accuracy, true decline, false
decline, trap help and garbage share.

## Tools and chat

`notebooks/tools.ipynb` calls a `draw` tool (Z-Image-Turbo) through the model's own chat
template; `notebooks/chat.ipynb` is a multi-turn prototype that switches between the base,
an adapter or the steering vector. Both are separate from training and evaluation.

## Reading

`docs/books/` covers the basics, model choice, LoRA and its variants, RLHF, DPO and
activation steering. `docs/references.md` lists the papers.

## License

MIT
