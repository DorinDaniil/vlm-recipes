import re
from pathlib import Path

IMAGE_MODEL = "Tongyi-MAI/Z-Image-Turbo"
IMAGES = Path(__file__).resolve().parents[1] / "runs" / "images"

DRAW = {
    "type": "function",
    "function": {
        "name": "draw",
        "description": "Нарисовать схему, иллюстрацию или график, когда картинка помогает объяснить студенту. Описание картинки на английском.",
        "parameters": {
            "type": "object",
            "properties": {"prompt": {"type": "string", "description": "What to draw, in English"}},
            "required": ["prompt"],
        },
    },
}
TOOLS = [DRAW]

CALL = re.compile(r"<tool_call>\s*<function=(\w+)>(.*?)</function>\s*</tool_call>", re.S)
PARAM = re.compile(r"<parameter=(\w+)>\s*(.*?)\s*</parameter>", re.S)


def calls(text):
    return [(name, dict(PARAM.findall(body))) for name, body in CALL.findall(text)]


def plain(text):
    return CALL.sub("", text).strip()


class Painter:
    def __init__(self, model_id=IMAGE_MODEL, steps=9, size=1024):
        import torch
        from diffusers import ZImagePipeline

        self.pipe = ZImagePipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16, low_cpu_mem_usage=False)
        self.pipe.enable_sequential_cpu_offload()
        self.steps = steps
        self.size = size

    def __call__(self, prompt, name):
        IMAGES.mkdir(parents=True, exist_ok=True)
        path = IMAGES / f"{name}.png"
        image = self.pipe(prompt, height=self.size, width=self.size, num_inference_steps=self.steps, guidance_scale=0.0).images[0]
        image.save(path)
        return path


RUN = {"draw": lambda args, painter, name: painter(args["prompt"], name)}
