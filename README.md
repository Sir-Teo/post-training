# Minimal Post-Training Demos for LLMs

This repo contains **tiny, runnable Python demonstrations** of every post-training method mentioned in *"LLM Post-Training: A Deep Dive into Reasoning Large Language Models"* (arXiv:2502.21321v2).  Each script is designed for educational clarity and executes on macOS CPU in a couple of minutes.

## Quick Start
```bash
# create venv & install deps
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# run all demos side-by-side (shows first 10 lines of each)
python run_all_demos.py --lines 10

# run unit tests (each script loads with --help)
pytest -q
```

> **Note on Torch >= 2.6**
>
> Newer versions of 🤗 Transformers forbid loading *legacy* `.bin` checkpoints with Torch < 2.6 (see CVE-2025-32434).  If you encounter an error when a script loads a model, simply upgrade torch:
>
> ```bash
> pip install --upgrade "torch>=2.6"
> ```

## Directory Layout
| Folder / File                     | Purpose |
|-----------------------------------|---------|
| `data/simple_tasks.jsonl`         | Shared toy prompt-response pairs (translation, arithmetic, capitals). |
| `demos/`                          | **One self-contained script per method**.  Open any file for annotated code. |
| `tests/test_demos.py`             | Pytest smoke test – each demo loads with `--help`. |
| `run_all_demos.py`                | Convenience runner that executes every demo sequentially and prints trimmed output. |
| `methods_list.md`                 | Mapping from paper sections → demo script name. |

## Demo Cheat-Sheet
| Script | Post-training Idea | One-sentence Explanation |
|--------|-------------------|--------------------------|
| `sft.py` | Supervised Fine-Tuning (SFT) | Train on labelled Q-A pairs. |
| `rlhf_ppo.py` | RLHF ‑ PPO | PPO optimises policy with human-written reward. |
| `rlaif_ppo.py` | RLAIF | Same as RLHF but reward ≈ another LLM. |
| `dpo.py` | Direct Preference Optimisation | Convert pairwise prefs → closed-form loss. |
| `trpo_stub.py` | Trust-Region Policy Opt. | Annotated walkthrough only. |
| `oreo_stub.py` | OREO | Stub with algorithmic summary. |
| `grpo_stub.py` | GRPO | Stub with algorithmic summary. |
| `cot_prompt.py` | Chain-of-Thought Prompting | Adds "Let’s think step by step" at inference. |
| `self_consistency.py` | Self-Consistency Decoding | Sample multiple reasoning chains and majority-vote. |
| `tree_of_thoughts.py` | Tree-of-Thoughts | Breadth-first search of reasoning tree. |
| `reward_modeling.py` | Explicit Reward Modelling | Trains a regression head to predict human reward. |
| `self_feedback.py` | Self-Feedback / Reflexion | Model critiques & revises its own answer. |
| `episodic_memory.py` | Agent w/ Episodic Memory | Maintains sliding-window memory of past Q-A. |

Each runnable script prints **before/after metrics** or **intermediate artefacts** (draft → critique → revision, votes, reward values, etc.) so you can see the method’s effect even with a tiny CPU model.

---
Made with 💜 for quick learning.
