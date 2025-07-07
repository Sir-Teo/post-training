"""Minimal Supervised Fine-Tuning (SFT) demo.

Trains `distilgpt2` for 1 epoch on the shared toy dataset and prints
examples before & after fine-tuning so learners can observe the change.

Run:
    python demos/sft.py --epochs 1 --lr 5e-5

Requirements (see requirements.txt): transformers, datasets, torch, tqdm
"""
import argparse
import os
os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION_IMPORTS", "1")
from pathlib import Path

import torch
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

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
    parser = argparse.ArgumentParser(description="Supervised fine-tuning demo")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--no_run", action="store_true", help="Exit after arg parsing (tests)")
    parser.add_argument("--quick", action="store_true", help="Use minimal manual training loop")
    args = parser.parse_args()
    if args.no_run:
        return

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token  # needed for batching

    print("Building dataset…")
    ds = build_dataset()
    tokenized_ds = tokenize_dataset(tokenizer, ds)

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, use_safetensors=True)

    print("\n=== Samples before fine-tuning ===")
    sample_model(model, tokenizer)

    if args.quick:
        # Manual tiny training loop on first 2 examples and 1 epoch
        subset = tokenized_ds.select(range(2))
        model.train()
        optim = torch.optim.AdamW(model.parameters(), lr=args.lr)
        for epoch in range(args.epochs):
            for rec in subset:
                ids = torch.tensor(rec["input_ids"]).unsqueeze(0)
                out = model(ids, labels=ids)
                out.loss.backward()
                optim.step(); optim.zero_grad()
        print("(quick SFT loop completed)")
    else:
        # heavy training imports only when full mode requested
        from transformers import TrainingArguments, Trainer

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
