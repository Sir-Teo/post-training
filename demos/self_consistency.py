"""Self-Consistency decoding demo.

Generates N reasoning chains for a simple arithmetic prompt, then returns
 the majority numeric answer across chains (Demonstrates self-consistency
from the paper *Self-Consistency Improves Chain of Thought*).

Run:
    python demos/self_consistency.py --prompt "What is 13 + 24?" --samples 20
"""
import argparse
from collections import Counter
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL = "distilgpt2"

RE_NUM = re.compile(r"(-?\d+)")

def extract_number(text:str):
    m = RE_NUM.search(text)
    return int(m.group(1)) if m else None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", default="If you have 8 bananas and eat 3, how many remain?", type=str)
    parser.add_argument("--samples", type=int, default=20)
    args = parser.parse_args()

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL)
    prompt = "Let's think step by step. " + args.prompt
    inputs = tok(prompt, return_tensors="pt")

    gens = model.generate(**inputs, do_sample=True, temperature=0.7, max_new_tokens=32, num_return_sequences=args.samples)
    answers = []
    for g in gens:
        text = tok.decode(g, skip_special_tokens=True)
        num = extract_number(text.split("\n")[-1])  # look at last line
        if num is not None:
            answers.append(num)
    if not answers:
        print("No numeric answers parsed.")
        return
    majority = Counter(answers).most_common(1)[0][0]
    print(f"Self-consistency voting among {len(answers)} parses → {majority}")

if __name__ == "__main__":
    torch.manual_seed(0)
    main()
