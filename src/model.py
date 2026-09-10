import torch
from peft import PeftModel
from transformers import AutoModelForImageTextToText, AutoTokenizer

from src import data

checkpoint = "Qwen/Qwen3.5-9B"


def load(model_id=checkpoint):
    model = AutoModelForImageTextToText.from_pretrained(
        model_id, dtype=torch.bfloat16, device_map={"": 0}, attn_implementation="sdpa")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.padding_side = "left"
    return model, tokenizer


def adapter(model, name):
    return PeftModel.from_pretrained(model, str(data.runs / name / "adapter"))


def adapters(model, names):
    model = PeftModel.from_pretrained(model, str(data.runs / names[0] / "adapter"), adapter_name=names[0])
    for name in names[1:]:
        model.load_adapter(str(data.runs / name / "adapter"), adapter_name=name)
    return model


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


def logprob(model, tokenizer, chat, answer):
    text = tokenizer.apply_chat_template(chat, add_generation_prompt=True, enable_thinking=False, tokenize=False)
    prefix = tokenizer(text, add_special_tokens=False)["input_ids"]
    ids = prefix + tokenizer(answer, add_special_tokens=False)["input_ids"] + [tokenizer.eos_token_id]
    labels = [-100] * len(prefix) + ids[len(prefix):]
    with torch.no_grad():
        loss = model(input_ids=torch.tensor([ids], device=model.device),
                     labels=torch.tensor([labels], device=model.device)).loss
    return -loss.item()


def logprobs(model, tokenizer, pairs):
    good = [logprob(model, tokenizer, r["prompt"], r["chosen"][0]["content"]) for r in pairs]
    bad = [logprob(model, tokenizer, r["prompt"], r["rejected"][0]["content"]) for r in pairs]
    return good, bad
