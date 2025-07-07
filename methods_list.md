# Post-Training Methods Covered (from "LLM Post-Training" paper)

This repository includes a **minimal, runnable demonstration** for every post-training technique explicitly named in the paper.
Each demo lives in `demos/<method>.py` and operates on the same tiny toy dataset found in `data/simple_tasks.jsonl`.

| Section in Paper | Method / Concept | Demo Script |
|------------------|------------------|-------------|
| 4 Supervised Fine-Tuning | Instruction Tuning (SFT) | `demos/sft.py` |
| | Long-Context / CoT SFT | `demos/sft_cot.py` |
| 3 Reinforced LLMs | RLHF (PPO w/ human prefs) | `demos/rlhf_ppo.py` |
| | RLAIF (AI feedback) | `demos/rlaif_ppo.py` |
| | DPO (Direct Preference Optimisation) | `demos/dpo.py` |
| | TRPO | `demos/trpo_stub.py` |
| | OREO | `demos/oreo_stub.py` |
| | GRPO | `demos/grpo_stub.py` |
| 5 Test-time Scaling | Chain-of-Thought Prompting | `demos/cot_prompt.py` |
| | Self-Consistency Decoding | `demos/self_consistency.py` |
| | Tree-of-Thoughts (ToT) Search | `demos/tree_of_thoughts.py` |
| 3.1 Reward Modeling | Explicit vs Implicit Reward Modeling | `demos/reward_modeling.py` |
| Agentic Concepts | Self-Feedback (Reflexion) | `demos/self_feedback.py` |
| | Episodic Memory Agent | `demos/episodic_memory.py` |

If a method (e.g. TRPO, OREO, GRPO) lacks an open-source implementation, the corresponding script is a clearly documented **stub** that walks through the algorithm with pseudo-updates so learners still grasp the idea.

Run `python demos/<script>.py --help` for options.
