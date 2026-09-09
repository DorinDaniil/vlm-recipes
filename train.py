import sys

import torch
from peft import LoraConfig
from trl import DPOConfig, DPOTrainer, SFTConfig, SFTTrainer

from src import data, steering
from src import model as m

LORA = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    use_rslora=True,
    task_type="CAUSAL_LM",
    target_modules=r"^(?!.*visual).*(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$",
)

COMMON = dict(
    bf16=True,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    max_length=512,
    lr_scheduler_type="cosine",
    warmup_steps=0.05,
    logging_steps=5,
    save_strategy="no",
    report_to=[],
    seed=42,
)


def fit(trainer, name):
    history = trainer.train()
    trainer.model.save_pretrained(data.RUNS / name / "adapter")
    print(f"{name}: loss {history.training_loss:.3f}, {history.metrics['train_runtime'] / 60:.1f} min")


def sft(model, tokenizer):
    config = SFTConfig(
        output_dir=str(data.RUNS / "sft"),
        num_train_epochs=3,
        learning_rate=1e-4,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        completion_only_loss=True,
        **COMMON,
    )
    train = data.sft(data.rows("train"))
    fit(SFTTrainer(model=model, args=config, train_dataset=train, processing_class=tokenizer, peft_config=LORA), "sft")


def dpo(model, tokenizer):
    model = m.merge(model, "sft")
    config = DPOConfig(
        output_dir=str(data.RUNS / "dpo"),
        num_train_epochs=2,
        learning_rate=5e-5,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        beta=0.1,
        **COMMON,
    )
    train = data.pairs(data.rows("train"))
    fit(DPOTrainer(model=model, args=config, train_dataset=train, processing_class=tokenizer, peft_config=LORA), "dpo")


def steer(model, tokenizer):
    layer = len(steering.layers(model)) // 2
    rows = [r for r in data.rows("train") if r["decision"] == "decline"]
    vector = steering.build(model, tokenizer, data.pairs(rows), layer)
    steering.VECTOR.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"vector": vector, "layer": layer}, steering.VECTOR)
    print(f"steer: layer {layer}, {len(rows)} rows, norm {vector.norm():.2f}")


METHODS = {"sft": sft, "dpo": dpo, "steer": steer}

if __name__ == "__main__":
    model, tokenizer = m.load()
    METHODS[sys.argv[1]](model, tokenizer)
