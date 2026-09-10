import sys

import torch

from src import data, evaluate, steering
from src import model as m


def base(model, tokenizer):
    evaluate.evaluate(model, tokenizer, "base", note="base model, neutral system prompt")


def prompted(model, tokenizer):
    evaluate.evaluate(model, tokenizer, "prompted", data.strict, note="base model, strict system prompt")


def sft(model, tokenizer):
    evaluate.evaluate(m.adapter(model, "sft"), tokenizer, "sft", note="LoRA SFT")


def dpo(model, tokenizer):
    evaluate.evaluate(m.adapter(model, "dpo"), tokenizer, "dpo", note="LoRA DPO from the base model")


def steer(model, tokenizer, alphas=(0.5, 1.0, 2.0)):
    saved = torch.load(steering.vector_path)
    rows = data.rows("test")
    for alpha in alphas:
        with steering.Steer(model, saved["vector"], saved["layer"], alpha):
            answers = evaluate.answer(model, tokenizer, rows)
            good, bad = m.logprobs(model, tokenizer, data.pairs(rows))
        evaluate.save(f"steer{alpha:+g}", rows, answers, evaluate.metrics(good, bad, answers, tokenizer),
                      note=f"steering vector, alpha {alpha}")


methods = {"base": base, "prompted": prompted, "sft": sft, "dpo": dpo, "steer": steer}

if __name__ == "__main__":
    model, tokenizer = m.load()
    alphas = [float(a) for a in sys.argv[2:]]
    methods[sys.argv[1]](model, tokenizer, *([alphas] if alphas else []))
    print(evaluate.table())
