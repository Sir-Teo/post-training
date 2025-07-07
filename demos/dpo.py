"""Minimal Direct Preference Optimization (DPO) demo using TRL.

DPO directly optimises the policy to prefer preferred responses over
rejected ones without an explicit reward model.

Here we fabricate preferences from the toy dataset: the ground-truth
response is *preferred*; a policy-sampled wrong answer is *rejected*.

Run:
    python demos/dpo.py --steps 20
"""
from __future__ import annotations
import argparse
from pathlib import Path
from typing import List

import torch
from datasets import load_dataset, Dataset
import os
os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION_IMPORTS", "1")
from transformers import AutoTokenizer, AutoModelForCausalLM

DATA_FILE = Path(__file__).resolve().parent.parent / "data" / "simple_tasks.jsonl"
MODEL_NAME = "distilgpt2"

def build_pref_dataset(policy, tokenizer) -> Dataset:
    """Return HF dataset with fields: prompt, chosen, rejected."""
    raw = load_dataset("json", data_files=str(DATA_FILE), split="train")
    prompts, chosen, rejected = [], [], []
    for ex in raw:
        p = ex["prompt"]
        prompts.append(p)
        chosen.append(ex["response"])
        # generate a (likely) wrong answer as rejected example
        with torch.no_grad():
            gen = policy.generate(**tokenizer(p, return_tensors="pt"), max_new_tokens=10)
        wrong = tokenizer.decode(gen[0], skip_special_tokens=True)
        rejected.append(wrong)
    return Dataset.from_dict({"prompt": prompts, "chosen": chosen, "rejected": rejected})


def main():
    parser = argparse.ArgumentParser(description="DPO demo")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--no_run", action="store_true", help="Exit after arg parsing (tests)")
    args = parser.parse_args()
    if args.no_run:
        return

    # Heavy import only when actually running
    from trl import DPOConfig, DPOTrainer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    policy = AutoModelForCausalLM.from_pretrained(MODEL_NAME, use_safetensors=True)
    ref_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, use_safetensors=True)

    dpo_ds = build_pref_dataset(policy, tokenizer)

    dpo_cfg = DPOConfig(
        beta=0.1,
        learning_rate=1e-5,
        max_length=128,
        batch_size=2,
        log_with=None,
    )
    trainer = DPOTrainer(
        policy_model=policy,
        ref_model=ref_model,
        args=dpo_cfg,
        train_dataset=dpo_ds,
        tokenizer=tokenizer,
    )
    trainer.train(max_steps=args.steps)

    print("\n=== Samples after DPO ===")
    for ex in load_dataset("json", data_files=str(DATA_FILE), split="train"):
        p = ex["prompt"]
        out = policy.generate(**tokenizer(p, return_tensors="pt"), max_new_tokens=20)
        print(p)
        print(" ->", tokenizer.decode(out[0], skip_special_tokens=True))

if __name__ == "__main__":
    torch.manual_seed(0)
    main()
