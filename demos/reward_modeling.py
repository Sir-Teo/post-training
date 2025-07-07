"""Reward modeling illustration.

Trains a simple regression model (linear head) to score (prompt+response)
pairs using the toy dataset. Demonstrates explicit reward modeling.
Implicit reward modeling is shown by simulating click-through-like
signals (random noisy reward) and fitting the same model.
Run:
    python demos/reward_modeling.py
"""
import os
os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION_IMPORTS", "1")
import argparse
from pathlib import Path
import random
import torch, torch.nn as nn
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel

DATA = Path(__file__).resolve().parent.parent / "data" / "simple_tasks.jsonl"
MODEL = "distilbert-base-uncased"

class RewardModel(nn.Module):
    def __init__(self, base):
        super().__init__()
        self.base = base
        self.scorer = nn.Linear(base.config.hidden_size, 1)
    def forward(self, **kw):
        out = self.base(**kw, return_dict=True)
        cls = out.last_hidden_state[:,0]
        return self.scorer(cls).squeeze(-1)

def main():
    parser = argparse.ArgumentParser(description="Reward modeling demo")
    parser.add_argument("--no_run", action="store_true", help="Exit after arg parsing (tests)")
    args = parser.parse_args()
    if args.no_run:
        return
    ds = load_dataset("json", data_files=str(DATA), split="train")
    tok = AutoTokenizer.from_pretrained(MODEL, use_fast=True)
    try:
        base = AutoModel.from_pretrained(MODEL, use_safetensors=True)
    except Exception as e:
        print(f"[WARN] Could not load DistilBERT model; skipping Reward Modeling demo: {e}")
        return
    model = RewardModel(base)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-5)

    # Explicit rewards: 1 for correct answer, 0 for wrong random answer
    pairs = []
    for ex in ds:
        pairs.append((ex["prompt"]+ex["response"], 1.0))
        wrong = random.choice([e["response"] for e in ds if e!=ex])
        pairs.append((ex["prompt"]+wrong, 0.0))

    for epoch in range(2):
        random.shuffle(pairs)
        for txt,label in pairs:
            enc = tok(txt, return_tensors="pt", truncation=True)
            pred = model(**enc)
            loss=((pred-label)**2).mean()
            loss.backward(); opt.step(); opt.zero_grad()
        print(f"epoch {epoch} loss {loss.item():.4f}")

    # implicit reward modeling example omitted for brevity
    print("Trained reward model scores:")
    for ex in ds:
        score=model(**tok(ex["prompt"]+ex["response"],return_tensors='pt')).item()
        print(ex["prompt"],"->",score)

if __name__=='__main__':
    torch.manual_seed(0)
    main()
