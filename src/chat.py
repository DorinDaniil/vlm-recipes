import contextlib

import torch

from src import data, steering, tools
from src import model as m


class Chat:
    def __init__(self, model, tokenizer, variant="base", system=data.neutral, schemas=None, painter=None,
                 max_new_tokens=200, max_calls=3):
        self.model, self.tokenizer = model, tokenizer
        self.variant = variant
        self.schemas = schemas
        self.executors = tools.executors(painter) if schemas else {}
        self.max_new_tokens, self.max_calls = max_new_tokens, max_calls
        self.messages = [{"role": "system", "content": system}]
        self.results = []

    def context(self):
        stack = contextlib.ExitStack()
        steered = isinstance(self.variant, tuple)
        if hasattr(self.model, "disable_adapter"):
            if self.variant == "base" or steered:
                stack.enter_context(self.model.disable_adapter())
            else:
                self.model.set_adapter(self.variant)
        if steered:
            saved = torch.load(steering.vector_path)
            stack.enter_context(steering.Steer(self.model, saved["vector"], saved["layer"], self.variant[1]))
        return stack

    def generate(self):
        with self.context():
            return m.generate(self.model, self.tokenizer, [self.messages], tools=self.schemas,
                              max_new_tokens=self.max_new_tokens)[0]

    def ask(self, text):
        self.messages.append({"role": "user", "content": text})
        for _ in range(self.max_calls + 1):
            raw = self.generate()
            calls = tools.calls(raw)
            if not calls:
                self.messages.append({"role": "assistant", "content": raw})
                return raw
            self.messages.append({"role": "assistant", "content": tools.plain(raw), "tool_calls": [
                {"type": "function", "function": {"name": name, "arguments": args}} for name, args in calls]})
            for name, args in calls:
                result = self.executors[name](args)
                self.results.append(result)
                self.messages.append({"role": "tool", "content": str(result)})
        return raw

    def reset(self):
        self.messages = self.messages[:1]
        self.results = []

    def show(self):
        for message in self.messages[1:]:
            print(f"{message['role'].upper():9} {message['content']}")
            for call in message.get("tool_calls", []):
                print(f"{'':9} -> {call['function']['name']}({call['function']['arguments']})")
