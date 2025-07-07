"""Chain-of-Thought prompting demo.

Shows how adding the phrase "Let's think step by step" before a prompt
elicits more structured reasoning from an off-the-shelf GPT-2 model.

Run:
    python demos/cot_prompt.py
"""
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

import argparse

MODEL_NAME = "distilgpt2"

def generate(model, tok, prompt, max_new=64):
    ids = tok(prompt, return_tensors="pt").to(model.device)
    out = model.generate(**ids, max_new_tokens=max_new, temperature=0.7)
    return tok.decode(out[0], skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser(description="Chain-of-Thought prompting demo")
    parser.add_argument("--no_run", action="store_true", help="Just parse args and exit (used for tests)")
    args = parser.parse_args()
    if args.no_run:
        return

    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

    task = "If there are 12 apples and you eat 5, how many are left?"

    plain = generate(model, tok, task)
    cot_prompt = "Let's think step by step. " + task
    cot = generate(model, tok, cot_prompt)

    print("Plain generation:\n", plain)
    print("\nChain-of-Thought generation:\n", cot)

if __name__ == "__main__":
    torch.manual_seed(0)
    main()
