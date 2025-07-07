"""Minimal Supervised Fine-Tuning (SFT) demo.

Trains `distilgpt2` for 1 epoch on the shared toy dataset and prints
examples before & after fine-tuning so learners can observe the change.

Run:
    python demos/sft.py --epochs 1 --lr 5e-5

Requirements (see requirements.txt): transformers, datasets, torch, tqdm
"""
import argparse
from pathlib import Path

import torch
from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
)

DATA_FILE = Path(__file__).resolve().parent.parent / "data" / "simple_tasks.jsonl"
MODEL_NAME = "distilgpt2"

def build_dataset():
    raw = load_dataset("json", data_files=str(DATA_FILE), split="train")
    # Convert prompt/response pairs into a single causal LM training string.
    def combine(example):
        example["text"] = example["prompt"] + "\n###\n" + example["response"]
        return example

    return raw.map(combine)

def tokenize_dataset(tokenizer: AutoTokenizer, dataset: Dataset):
    def tokenize(batch):
        tokens = tokenizer(batch["text"], truncation=True)
        # causal LM label is input_ids itself
        tokens["labels"] = tokens["input_ids"].copy()
        return tokens

    return dataset.map(tokenize, batched=True, remove_columns=dataset.column_names)


def sample_model(model, tokenizer, n_samples: int = 3):
    model.eval()
    prompts = [
        "Translate to French: 'Good morning'",
        "Add the numbers: 9 + 10",
        "What is the capital of France?",
    ]
    for p in prompts[:n_samples]:
        inputs = tokenizer(p, return_tensors="pt").to(model.device)
        out = model.generate(**inputs, max_new_tokens=20)
        print("\nPrompt:", p)
        print("Output:", tokenizer.decode(out[0], skip_special_tokens=True))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-5)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token  # needed for batching

    print("Building dataset…")
    ds = build_dataset()
    tokenized_ds = tokenize_dataset(tokenizer, ds)

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, use_safetensors=True)

    print("\n=== Samples before fine-tuning ===")
    sample_model(model, tokenizer)

    training_args = TrainingArguments(
        output_dir="/tmp/sft-demo",
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=4,
        logging_steps=5,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_ds,
    )
    trainer.train()

    print("\n=== Samples after fine-tuning ===")
    sample_model(model, tokenizer)


if __name__ == "__main__":
    torch.manual_seed(0)
    main()
