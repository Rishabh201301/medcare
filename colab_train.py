import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, TaskType


@dataclass
class Config:
    # Using Llama 3.2 3B Instruct model
    base_model: str = "meta-llama/Llama-3.2-3B-Instruct"
    data_dir: str = "data_prepared"
    output_dir: str = "outputs/lora-medqa-colab"
    bf16: bool = False  # many Colab GPUs (T4) lack bf16
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 8  # Reduced for better stability
    learning_rate: float = 5e-5  # Lower learning rate for stability
    num_train_epochs: int = 3
    max_steps: int = -1
    logging_steps: int = 5  # More frequent logging
    save_steps: int = 200
    eval_steps: int = 200
    warmup_ratio: float = 0.1  # More warmup
    lora_r: int = 8  # Smaller rank for stability
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    max_seq_len: int = 512  # Shorter sequences for stability


def load_jsonl(path: Path) -> List[Dict]:
    items = []
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))
    return items


def format_prompt(ex: Dict) -> str:
    system = ex.get("system", "You are a helpful medical assistant.")
    instruction = ex.get("instruction", "")
    user_input = ex.get("input", "")
    output = ex.get("output", "")
    return (
        f"<s>[SYSTEM]\n{system}\n[/SYSTEM]\n"
        f"[USER]\n{user_input or instruction}\n[/USER]\n"
        f"[ASSISTANT]\n{output}\n</s>"
    )


class JsonlDataset(Dataset):
    def __init__(self, items: List[Dict], tokenizer: AutoTokenizer, max_len: int):
        self.items = items
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        text = format_prompt(self.items[idx])
        toks = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_len,
            padding=False,
            return_tensors="pt",
        )
        toks = {k: v.squeeze(0) for k, v in toks.items()}
        toks["labels"] = toks["input_ids"].clone()
        return toks


def main():
    cfg = Config()
    root = Path(__file__).parent
    data_dir = root / cfg.data_dir
    train_path = data_dir / 'train.jsonl'
    val_path = data_dir / 'val.jsonl'
    assert train_path.exists(), "Run data_prep.py first"

    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
    )
    model = get_peft_model(base_model, lora_cfg)

    train_items = load_jsonl(train_path)
    val_items = load_jsonl(val_path) if val_path.exists() else []
    train_ds = JsonlDataset(train_items, tokenizer, cfg.max_seq_len)
    eval_ds = JsonlDataset(val_items, tokenizer, cfg.max_seq_len) if val_items else None

    args = TrainingArguments(
        output_dir=str(root / cfg.output_dir),
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        num_train_epochs=cfg.num_train_epochs,
        max_steps=cfg.max_steps,
        logging_steps=cfg.logging_steps,
        save_steps=cfg.save_steps,
        eval_strategy="steps" if eval_ds else "no",
        eval_steps=cfg.eval_steps,
        warmup_ratio=cfg.warmup_ratio,
        bf16=False,
        fp16=False,  # Disable fp16 for Mac compatibility
        dataloader_pin_memory=False,  # Disable for Mac compatibility
        save_total_limit=2,
        report_to=[],
        remove_unused_columns=False,  # Important for custom datasets
        gradient_checkpointing=True,  # Save memory
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=data_collator,
    )

    trainer.train()
    model.save_pretrained(str(root / cfg.output_dir / 'adapter'))
    tokenizer.save_pretrained(str(root / cfg.output_dir / 'adapter'))
    print("Saved LoRA adapter to", root / cfg.output_dir / 'adapter')


if __name__ == '__main__':
    main()


