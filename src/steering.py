import torch

from src import data

vector_path = data.runs / "steer" / "vector.pt"


def layers(model):
    return model.model.language_model.layers


def hidden_of(out):
    return out[0] if isinstance(out, tuple) else out


def activations(model, tokenizer, texts, layer):
    states = []
    handle = layers(model)[layer].register_forward_hook(
        lambda module, args, out: states.append(hidden_of(out)[0].float().cpu()))
    with torch.no_grad():
        for text in texts:
            model(**tokenizer(text, return_tensors="pt", add_special_tokens=False).to(model.device))
    handle.remove()
    return states


def build(model, tokenizer, pairs, layer):
    def rendered(key):
        return [tokenizer.apply_chat_template(r["prompt"] + r[key], tokenize=False, enable_thinking=False) for r in pairs]
    heads = [tokenizer.apply_chat_template(r["prompt"], add_generation_prompt=True, enable_thinking=False, tokenize=False) for r in pairs]
    starts = [len(tokenizer(h, add_special_tokens=False)["input_ids"]) for h in heads]
    chosen = [h[s:] for h, s in zip(activations(model, tokenizer, rendered("chosen"), layer), starts)]
    rejected = [h[s:] for h, s in zip(activations(model, tokenizer, rendered("rejected"), layer), starts)]
    vector = torch.stack([h.mean(dim=0) for h in chosen]).mean(dim=0) - torch.stack([h.mean(dim=0) for h in rejected]).mean(dim=0)
    scale = torch.cat(chosen).norm(dim=-1).mean().item()
    return vector, scale


class Steer:
    def __init__(self, model, vector, layer, alpha):
        self.block = layers(model)[layer]
        self.shift = (alpha * vector).to(model.device)

    def __enter__(self):
        self.handle = self.block.register_forward_hook(self.hook)
        return self

    def __exit__(self, *exc):
        self.handle.remove()

    def hook(self, module, args, out):
        shifted = hidden_of(out) + self.shift.to(hidden_of(out).dtype)
        return (shifted, *out[1:]) if isinstance(out, tuple) else shifted
