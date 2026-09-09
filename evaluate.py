import sys

import torch

from src import data, evaluate, steering
from src import model as m


def base(model, tokenizer):
    evaluate.evaluate(model, tokenizer, "base", note="base model, neutral system prompt")


def prompted(model, tokenizer):
    evaluate.evaluate(model, tokenizer, "prompted", data.STRICT, note="base model, strict system prompt")


def sft(model, tokenizer):
    evaluate.evaluate(m.tuned(model, "sft"), tokenizer, "sft", note="LoRA SFT")


def dpo(model, tokenizer):
    evaluate.evaluate(m.tuned(model, "dpo"), tokenizer, "dpo", note="DPO on top of SFT")


def steer(model, tokenizer, alphas=(1.0, 2.0)):
    saved = torch.load(steering.VECTOR)
    rows = data.rows("test")
    for alpha in alphas:
        with steering.Steer(model, saved["vector"], saved["layer"], alpha):
            answers = evaluate.answer(model, tokenizer, rows)
            pref_acc = m.preference_accuracy(model, tokenizer, data.pairs(rows))
        evaluate.save(f"steer{alpha:+.0f}", rows, answers, pref_acc, note=f"steering vector, alpha {alpha}")


METHODS = {"base": base, "prompted": prompted, "sft": sft, "dpo": dpo, "steer": steer}

if __name__ == "__main__":
    model, tokenizer = m.load()
    alphas = [float(a) for a in sys.argv[2:]]
    METHODS[sys.argv[1]](model, tokenizer, *([alphas] if alphas else []))
