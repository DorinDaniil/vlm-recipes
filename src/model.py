import torch
from peft import PeftModel
from transformers import AutoModelForImageTextToText, AutoTokenizer

from src.data import JUDGE, RUNS

MODEL = "Qwen/Qwen3.5-9B"


def load(model_id=MODEL):
    model = AutoModelForImageTextToText.from_pretrained(
        model_id, dtype=torch.bfloat16, device_map={"": 0}, attn_implementation="sdpa")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.padding_side = "left"
    return model, tokenizer


def adapter(model, name):
    return PeftModel.from_pretrained(model, RUNS / name / "adapter")


def merge(model, name):
    return adapter(model, name).merge_and_unload()


def tuned(model, name):
    if name == "dpo":
        model = merge(model, "sft")
    return adapter(model, name)


def generate(model, tokenizer, chats, tools=None, max_new_tokens=120, batch_size=16):
    model.eval()
    answers = []
    for start in range(0, len(chats), batch_size):
        batch = tokenizer.apply_chat_template(
            chats[start:start + batch_size], tools=tools, add_generation_prompt=True, enable_thinking=False,
            tokenize=True, padding=True, return_dict=True, return_tensors="pt").to(model.device)
        with torch.no_grad():
            ids = model.generate(**batch, max_new_tokens=max_new_tokens, do_sample=False,
                                 pad_token_id=tokenizer.pad_token_id)
        answers += tokenizer.batch_decode(ids[:, batch["input_ids"].shape[1]:], skip_special_tokens=True)
    return [a.strip() for a in answers]


def judge(model, tokenizer, requests, answers):
    chats = [[{"role": "user", "content": JUDGE.format(request=q, answer=a)}] for q, a in zip(requests, answers)]
    return ["ОТКАЗ" in v.upper() for v in generate(model, tokenizer, chats, max_new_tokens=5)]


def logprob(model, tokenizer, chat, answer):
    text = tokenizer.apply_chat_template(chat, add_generation_prompt=True, enable_thinking=False, tokenize=False)
    prefix = tokenizer(text, add_special_tokens=False)["input_ids"]
    ids = prefix + tokenizer(answer, add_special_tokens=False)["input_ids"] + [tokenizer.eos_token_id]
    labels = [-100] * len(prefix) + ids[len(prefix):]
    with torch.no_grad():
        loss = model(input_ids=torch.tensor([ids], device=model.device),
                     labels=torch.tensor([labels], device=model.device)).loss
    return -loss.item()


def preference_accuracy(model, tokenizer, pairs):
    wins = sum(logprob(model, tokenizer, r["prompt"], r["chosen"][0]["content"])
               > logprob(model, tokenizer, r["prompt"], r["rejected"][0]["content"]) for r in pairs)
    return wins / len(pairs)
