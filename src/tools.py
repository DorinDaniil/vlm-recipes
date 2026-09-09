import re

from src import data

image_checkpoint = "Tongyi-MAI/Z-Image-Turbo"
images = data.runs / "images"

system = data.neutral + " Если студент просит что-нибудь нарисовать, вызови инструмент draw и опиши картинку образно, как её увидит художник."

draw = {
    "type": "function",
    "function": {
        "name": "draw",
        "description": "Нарисовать картинку по просьбе студента: схему, иллюстрацию, образ. Описание на английском: что изображено, в каком стиле.",
        "parameters": {
            "type": "object",
            "properties": {"prompt": {"type": "string", "description": "What to draw and in what style, in English"}},
            "required": ["prompt"],
        },
    },
}
schemas = [draw]

call_pattern = re.compile(r"<tool_call>\s*<function=(\w+)>(.*?)</function>\s*</tool_call>", re.S)
param_pattern = re.compile(r"<parameter=(\w+)>\s*(.*?)\s*</parameter>", re.S)


def calls(text):
    return [(name, dict(param_pattern.findall(body))) for name, body in call_pattern.findall(text)]


def plain(text):
    return call_pattern.sub("", text).strip()


class Painter:
    def __init__(self, model_id=image_checkpoint, steps=9, size=1024):
        import torch
        from diffusers import ZImagePipeline

        self.pipe = ZImagePipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16, low_cpu_mem_usage=False)
        self.pipe.enable_sequential_cpu_offload()
        self.steps = steps
        self.size = size

    def __call__(self, prompt, name):
        images.mkdir(parents=True, exist_ok=True)
        path = images / f"{name}.png"
        image = self.pipe(prompt, height=self.size, width=self.size, num_inference_steps=self.steps, guidance_scale=0.0).images[0]
        image.save(path)
        return path


run = {"draw": lambda args, painter, name: painter(args["prompt"], name)}
