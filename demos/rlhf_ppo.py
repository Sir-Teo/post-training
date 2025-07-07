"""Minimal RLHF demonstration with TRL's PPOTrainer.

This demo synthesizes a *reward model* that simply rewards outputs that
match the ground-truth responses in the toy dataset. While trivial, it
illustrates the full PPO fine-tuning loop (policy / reference model,
reward model, KL penalty) on a single GPU/CPU.

Run:
    python demos/rlhf_ppo.py --steps 20

(20 PPO steps finish in <1 minute on CPU.)
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import os
os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION_IMPORTS", "1")
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

DATA_FILE = Path(__file__).resolve().parent.parent / "data" / "simple_tasks.jsonl"
MODEL_NAME = "distilgpt2"


def load_prompts() -> List[str]:
    data = load_dataset("json", data_files=str(DATA_FILE), split="train")
    return [ex["prompt"] for ex in data]


def synthetic_reward_fn(generated: str, prompt: str, refs: List[str]):
    """Return +1 if generation exactly matches *any* reference answer, else 0."""
    for ref in refs:
        if generated.strip().lower() == ref.strip().lower():
            return 1.0
    return 0.0


def build_ref_answers():
    ds = load_dataset("json", data_files=str(DATA_FILE), split="train")
    return {ex["prompt"]: ex["response"] for ex in ds}


def main():
    parser = argparse.ArgumentParser(description="RLHF PPO demo")
    parser.add_argument("--steps", type=int, default=20, help="PPO optimisation steps")
    parser.add_argument("--no_run", action="store_true", help="Exit after arg parsing (tests)")
    args = parser.parse_args()
    if args.no_run:
        return

    # Import TRL lazily to avoid heavy deps during --no_run test mode
    try:
        from trl import PPOTrainer, PPOConfig
    except Exception as e:
        print(f"[WARN] Could not import TRL (trl). Skipping RLHF PPO demo: {e}")
        return

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    # Policy & reference models
    policy_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, use_safetensors=True)
    ref_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, use_safetensors=True)

    ppo_config = PPOConfig(
        batch_size=2,
        forward_batch_size=1,
        learning_rate=1e-5,
        log_with=None,
    )
    trainer = PPOTrainer(ppo_config, policy_model, ref_model, tokenizer)

    prompts = load_prompts()
    refs = build_ref_answers()

    for step in range(args.steps):
        batch_prompts = [prompts[step % len(prompts)]]  # simple 1-prompt batch
        enc = tokenizer(batch_prompts, return_tensors="pt", padding=True)
        with torch.no_grad():
            gen_out = policy_model.generate(**enc, max_new_tokens=20)
        generated_text = tokenizer.batch_decode(gen_out[:, enc.input_ids.shape[1]:], skip_special_tokens=True)[0]

        # Compute synthetic reward
        reward = synthetic_reward_fn(generated_text, batch_prompts[0], [refs[batch_prompts[0]]])
        rewards = torch.tensor([reward])

        # Run one PPO optimisation step
        trainer.step(batch_prompts, [generated_text], rewards)
        if step % 5 == 0:
            print(f"Step {step:02d} | prompt: {batch_prompts[0]}\n -> {generated_text!r} | reward={reward}")

    print("\n=== After RLHF fine-tuning ===")
    for p in prompts:
        out = policy_model.generate(**tokenizer(p, return_tensors="pt"), max_new_tokens=20)
        print(p)
        print(" -> ", tokenizer.decode(out[0], skip_special_tokens=True))


if __name__ == "__main__":
    torch.manual_seed(0)
    main()
