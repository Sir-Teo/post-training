"""Episodic Memory Agent demo.

A toy agent that keeps a sliding-window memory of previous Q&A pairs and
includes a summary of that memory in each new prompt—demonstrating the
"episodic memory" intermediate concept.

Run:
    python demos/episodic_memory.py
Then type questions; Ctrl+C to quit.
"""
import readline  # noqa: F401 enables arrow-key history on mac
import textwrap
from collections import deque
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL = "distilgpt2"
MEM_SIZE = 3  # remember last 3 episodes


def summarise(memory):
    """Simple summariser: join last MEM_SIZE Q&A pairs."""
    return "\n".join(memory)


def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL)

    memory = deque(maxlen=MEM_SIZE)
    print("Episodic-memory chatbot – ask something (Ctrl+C to quit)")
    while True:
        try:
            question = input("You: ")
        except KeyboardInterrupt:
            print("\nBye!")
            break
        context = summarise(memory)
        prompt = textwrap.dedent(f"""
        The following is a conversation. Previous context (may be empty):
        {context}
        \nUser: {question}\nAssistant:""")
        ids = tok(prompt, return_tensors="pt").to(model.device)
        out = model.generate(**ids, max_new_tokens=64, temperature=0.7)
        answer = tok.decode(out[0], skip_special_tokens=True).split("Assistant:")[-1].strip()
        print("Agent:", answer)
        memory.append(f"User: {question}\nAssistant: {answer}")

if __name__ == "__main__":
    torch.manual_seed(0)
    main()
