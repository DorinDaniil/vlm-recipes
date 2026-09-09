import torch

from src import data

vector_path = data.runs / "steer" / "vector.pt"


def layers(model):
    return model.model.language_model.layers


def hidden_of(out):
    return out[0] if isinstance(out, tuple) else out


def mean_activation(model, tokenizer, texts, layer):
    states = []
    handle = layers(model)[layer].register_forward_hook(
        lambda module, args, out: states.append(hidden_of(out).float().mean(dim=1).squeeze(0).cpu()))
    with torch.no_grad():
        for text in texts:
            model(**tokenizer(text, return_tensors="pt", add_special_tokens=False).to(model.device))
    handle.remove()
    return torch.stack(states).mean(dim=0)


def build(model, tokenizer, pairs, layer):
    def rendered(key):
        return [tokenizer.apply_chat_template(r["prompt"] + r[key], tokenize=False, enable_thinking=False) for r in pairs]
    return mean_activation(model, tokenizer, rendered("chosen"), layer) - mean_activation(model, tokenizer, rendered("rejected"), layer)


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
