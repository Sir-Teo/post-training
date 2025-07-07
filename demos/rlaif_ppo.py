"""RLAIF demonstration: Reinforcement Learning from AI Feedback.

Identical loop to RLHF but the reward comes from a *trained* reward
model (small classifier) rather than human labels. For brevity we train a
1-layer reward head on synthetic preference pairs sampled from the toy
dataset, then run PPO fine-tuning using that reward model.

Run (~1-2 min CPU):
    python demos/rlaif_ppo.py --pretrain_steps 30 --ppo_steps 20
"""
from __future__ import annotations
import argparse
import os
os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION_IMPORTS", "1")
from pathlib import Path
import random
from typing import List
import torch
import torch.nn as nn
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM


DATA_FILE = Path(__file__).resolve().parent.parent / "data" / "simple_tasks.jsonl"
MODEL_NAME = "distilgpt2"

class RewardHead(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base = base_model
        hidden = base_model.config.n_embd
        self.scorer = nn.Linear(hidden, 1)

    def forward(self, input_ids, attention_mask=None):
        out = self.base(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        last_hidden = out.hidden_states[-1][:, -1, :]  # last token repr
        return self.scorer(last_hidden).squeeze(-1)


def build_pref_pairs(tokenizer) -> Dataset:
    """Construct synthetic preference dataset where correct answer wins."""
    data = load_dataset("json", data_files=str(DATA_FILE), split="train")
    prefs = {ex["prompt"]: ex["response"] for ex in data}
    prompts, better, worse = [], [], []
    for p, ans in prefs.items():
        wrong = random.choice([v for v in prefs.values() if v != ans])
        prompts.append(p)
        better.append(ans)
        worse.append(wrong)
    return Dataset.from_dict({"prompt": prompts, "better": better, "worse": worse})


def train_reward_model(reward_model, tokenizer, dataset: Dataset, steps=30):
    opt = torch.optim.AdamW(reward_model.parameters(), lr=1e-5)
    reward_model.train()
    for i in range(steps):
        batch = dataset.shuffle().select(range(8))  # mini-batch of 8 pairs
        loss = 0
        for p, b, w in zip(batch["prompt"], batch["better"], batch["worse"]):
            for txt, label in [(b, 1), (w, 0)]:
                ids = tokenizer(p + "\n" + txt, return_tensors="pt", truncation=True)
                score = reward_model(**ids)
                loss += ((score - label) ** 2).mean()
        loss /= 16
        loss.backward()
        opt.step(); opt.zero_grad()
        if i % 10 == 0:
            print(f"Reward pretrain step {i}: loss={loss.item():.4f}")
    reward_model.eval()


def get_reward_fn(reward_model, tokenizer):
    def fn(prompt: str, generation: str):
        ids = tokenizer(prompt + "\n" + generation, return_tensors="pt", truncation=True)
        with torch.no_grad():
            return reward_model(**ids).item()
    return fn


def main():
    parser = argparse.ArgumentParser(description="RLAIF PPO demo")
    parser.add_argument("--pretrain_steps", type=int, default=30)
    parser.add_argument("--ppo_steps", type=int, default=20)
    parser.add_argument("--no_run", action="store_true", help="Exit after arg parsing (tests)")
    args = parser.parse_args()
    if args.no_run:
        return

    # Lazy import after --no_run check
    try:
        from trl import PPOConfig, PPOTrainer
    except Exception as e:
        print(f"[WARN] Could not import TRL (trl). Skipping RLAIF PPO demo: {e}")
        return

    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    tok.pad_token = tok.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, use_safetensors=True)
    reward_base = AutoModelForCausalLM.from_pretrained(MODEL_NAME, use_safetensors=True)
    reward_model = RewardHead(reward_base.transformer)

    dataset = build_pref_pairs(tok)
    train_reward_model(reward_model, tok, dataset, steps=args.pretrain_steps)
    reward_fn = get_reward_fn(reward_model, tok)

    ref_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, use_safetensors=True)
    ppo_cfg = PPOConfig(batch_size=2, forward_batch_size=1, learning_rate=1e-5)
    trainer = PPOTrainer(ppo_cfg, base_model, ref_model, tok)

    prompts = [ex["prompt"] for ex in load_dataset("json", data_files=str(DATA_FILE), split="train")]

    for i in range(args.ppo_steps):
        p = prompts[i % len(prompts)]
        enc = tok(p, return_tensors="pt")
        gen = base_model.generate(**enc, max_new_tokens=20)
        gen_text = tok.decode(gen[0, enc.input_ids.shape[1]:], skip_special_tokens=True)
        reward = torch.tensor([reward_fn(p, gen_text)])
        trainer.step([p], [gen_text], reward)
        if i % 5 == 0:
            print(f"PPO step {i}: reward={reward.item():.3f} | gen={gen_text}")

    print("\n=== After RLAIF fine-tuning ===")
    for p in prompts:
        out = base_model.generate(**tok(p, return_tensors="pt"), max_new_tokens=20)
        print(p)
        print(" ->", tok.decode(out[0], skip_special_tokens=True))

if __name__ == "__main__":
    torch.manual_seed(0)
    main()
