"""Tree-of-Thoughts (ToT) search demo.

Implements a minimal breadth-first reasoning tree using the formulation
from the ToT paper. At each step the model expands K candidate thoughts;
we keep the top-scoring M by a simple heuristic (string length inverse),
iterate for D depth, then return the best final answer.

Because GPT-2 is not a reasoning powerhouse, this example focuses on the
algorithmic skeleton rather than result quality.

Run:
    python demos/tree_of_thoughts.py --question "Multiply 3 by 7 then add 4."
"""
import argparse
import networkx as nx
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL = "distilgpt2"
K = 3  # branching factor
M = 2  # kept per level
D = 2  # tree depth


def score(text: str) -> float:
    """Heuristic: shorter answers are preferred (demo purposes)."""
    return -len(text)


def expand(model, tok, state: str, k: int):
    ids = tok(state, return_tensors="pt").to(model.device)
    outs = model.generate(**ids, do_sample=True, temperature=0.8, max_new_tokens=16, num_return_sequences=k)
    return [tok.decode(o, skip_special_tokens=True) for o in outs]


def tot_search(question: str):
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL)

    root = question + "\nThought: "
    G = nx.DiGraph()
    G.add_node(root, score=score(root))

    frontier = [root]
    for depth in range(D):
        new_frontier = []
        for state in frontier:
            children = expand(model, tok, state, K)
            for child in children:
                G.add_edge(state, child)
                G.nodes[child]["score"] = score(child)
            new_frontier.extend(children)
        # prune to top-M by score
        new_frontier.sort(key=lambda s: G.nodes[s]["score"], reverse=True)
        frontier = new_frontier[:M]
    # select best leaf
    best = max(frontier, key=lambda s: G.nodes[s]["score"])
    return best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--question", default="If you have 2 cookies and get 3 more, how many in total?", type=str)
    args = parser.parse_args()
    answer = tot_search(args.question)
    print("Best answer tree leaf:\n", answer)

if __name__ == "__main__":
    torch.manual_seed(0)
    main()
