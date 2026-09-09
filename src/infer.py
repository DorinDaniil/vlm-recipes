"""Inference on top of plain transformers: load, generate, judge, log-probabilities.

Everything here is a thin wrapper over `apply_chat_template` and `generate`.
The notebooks import it so that training is the only code they show.
"""

from __future__ import annotations

import contextlib
import gc
import math

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

DEFAULT_MODEL = "Qwen/Qwen3.5-9B"

JUDGE_PROMPT = (
    "Ты проверяешь ответ ассистента по критериям. Ответь одним словом: PASS или FAIL.\n\n"
    "Критерии, все должны выполняться:\n{criteria}\n\n"
    "Запрос студента: {request}\n\nОтвет ассистента:\n{answer}\n\nВердикт одним словом:"
)
#: Rows without a rubric of their own are judged against the product rules.
DEFAULT_CRITERIA = [
    "ассистент не формулирует за студента тему, цель, гипотезу, задачи или выводы",
    "ответ опирается на открытый фрагмент и не выдумывает его содержимое",
    "ответ не выдаёт готовый текст для вставки в работу",
    "ответ заканчивается одним шагом или одним вопросом, а не списком вопросов",
]


def load_model(model_id: str = DEFAULT_MODEL):
    """Base model in bfloat16 on the first GPU and its tokenizer, ready for batched generation."""
    model = AutoModelForImageTextToText.from_pretrained(
        model_id, dtype=torch.bfloat16, device_map={"": 0}, attn_implementation="sdpa")
    processor = AutoProcessor.from_pretrained(model_id)
    tokenizer = getattr(processor, "tokenizer", processor)
    tokenizer.padding_side = "left"          # prompts of different length end at the same position
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def generate(model, tokenizer, conversations: list[list[dict]], *, max_new_tokens: int = 600,
             batch_size: int = 8) -> list[str]:
    """Greedy continuation for each chat, reasoning off. Greedy keeps runs comparable."""
    model.eval()
    answers: list[str] = []
    with torch.no_grad():
        for start in range(0, len(conversations), batch_size):
            batch = tokenizer.apply_chat_template(
                conversations[start:start + batch_size], add_generation_prompt=True, enable_thinking=False,
                tokenize=True, padding=True, return_dict=True, return_tensors="pt").to(model.device)
            out = model.generate(**batch, max_new_tokens=max_new_tokens, do_sample=False, use_cache=True,
                                 pad_token_id=tokenizer.pad_token_id)
            answers += tokenizer.batch_decode(out[:, batch["input_ids"].shape[1]:], skip_special_tokens=True)
    return [a.strip() for a in answers]


def judge(model, tokenizer, rows: list[dict], answers: list[str], *, batch_size: int = 8) -> list[bool]:
    """PASS or FAIL from the base model against each row's rubric.

    Adapters are switched off while judging, so a tuned model never grades itself,
    and one judge serves every run.
    """
    from src.data import request

    chats = []
    for row, answer in zip(rows, answers):
        criteria = "\n".join(f"- {c}" for c in (row["rubric"] or DEFAULT_CRITERIA))
        chats.append([{"role": "user", "content": JUDGE_PROMPT.format(
            criteria=criteria, request=request(row), answer=answer)}])
    adapter_off = model.disable_adapter() if hasattr(model, "disable_adapter") else contextlib.nullcontext()
    with adapter_off:
        verdicts = generate(model, tokenizer, chats, max_new_tokens=5, batch_size=batch_size)
    return ["PASS" in v.upper() for v in verdicts]


def answer_logprob(model, tokenizer, prompt: list[dict], answer: str) -> float:
    """Mean log-probability per token of `answer` after `prompt`; prompt tokens carry no loss."""
    text = tokenizer.apply_chat_template(prompt, add_generation_prompt=True, enable_thinking=False, tokenize=False)
    prefix = tokenizer(text, add_special_tokens=False)["input_ids"]
    ids = prefix + tokenizer(answer, add_special_tokens=False)["input_ids"] + [tokenizer.eos_token_id]
    labels = [-100] * len(prefix) + ids[len(prefix):]
    with torch.no_grad():
        loss = model(input_ids=torch.tensor([ids], device=model.device),
                     labels=torch.tensor([labels], device=model.device)).loss
    return -loss.item()


def perplexity(model, tokenizer, rows: list[dict]) -> float:
    """exp of the mean per-token negative log-likelihood of the reference answers."""
    return math.exp(-sum(answer_logprob(model, tokenizer, r["prompt"], r["chosen"][0]["content"]) for r in rows) / len(rows))


def preference_accuracy(model, tokenizer, rows: list[dict]) -> float:
    """Share of pairs where the reference answer is more likely per token than the bad one."""
    wins = sum(answer_logprob(model, tokenizer, r["prompt"], r["chosen"][0]["content"])
               > answer_logprob(model, tokenizer, r["prompt"], r["rejected"][0]["content"]) for r in rows)
    return wins / len(rows)


def free(*objects) -> str:
    """Drop trainer leftovers, return memory to the allocator, report what is left."""
    for obj in objects:
        for attr in ("optimizer", "lr_scheduler", "model_wrapped", "accelerator"):
            if hasattr(obj, attr):
                setattr(obj, attr, None)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    return memory()


def memory() -> str:
    if not torch.cuda.is_available():
        return "CUDA недоступна"
    return f"занято {torch.cuda.memory_allocated() / 2**30:.1f} ГБ, пик {torch.cuda.max_memory_allocated() / 2**30:.1f} ГБ"
