"""Self-Feedback (Reflexion-style) demo.

Uses the small `distilgpt2` model (safetensors weights) so the output
text is understandable in English while remaining CPU-friendly.


After generating an answer, the model critiques its own output and then
produces a revised answer. Illustrates the "self-feedback" intermediate
concept discussed in the paper.

Run:
    python demos/self_feedback.py --question "Define gravity in simple words."
"""
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL = "distilgpt2"

PROMPT_TEMPLATE = """Q: {question}\nA: {draft}\n\nWas the above answer correct and well-reasoned? If not, briefly say why.\nCritique: """

REVISION_TEMPLATE = """Q: {question}\nCritique: {critique}\n\nPlease provide an improved answer.\nA: """

def generate(model, tok, prompt, max_new=32):
    ids = tok(prompt, return_tensors="pt").to(model.device)
    out = model.generate(**ids, max_new_tokens=max_new)
    return tok.decode(out[0], skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser(description="Self-feedback demo")
    parser.add_argument("--question", type=str, default="What is 2 + 2?")
    parser.add_argument("--no_run", action="store_true", help="Parse args and exit (tests)")
    args = parser.parse_args()
    if args.no_run:
        return

    tok = AutoTokenizer.from_pretrained(MODEL, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL, use_safetensors=True)

    # Draft answer
    draft = generate(model, tok, "Q: " + args.question + "\nA:", 32).split("A:")[-1].strip()
    print("Draft:", draft)

    # Self critique
    critique_prompt = PROMPT_TEMPLATE.format(question=args.question, draft=draft)
    critique = generate(model, tok, critique_prompt, 64).split("Critique:")[-1].strip()
    print("Critique:", critique)

    # Revision
    revision_prompt = REVISION_TEMPLATE.format(question=args.question, critique=critique)
    improved = generate(model, tok, revision_prompt, 64)
    print("\nImproved answer:\n", improved)

if __name__ == "__main__":
    torch.manual_seed(0)
    main()
