"""GRPO algorithm stub.

Generalized Reward Policy Optimization (GRPO) extends PPO with a
reward-conditioned objective. No reference implementation is public; we
provide an annotated algorithm outline.
"""
import textwrap

def main():
    print(textwrap.dedent("""
    GRPO Overview
    -------------
    1. Maintain two policies: π (student) and ρ (reward model).
    2. Sample trajectories with π; obtain scalar reward r_t from ρ.
    3. Compute advantage Â based on r_t.
    4. Update π with PPO-style clipped objective weighted by Â.
    5. Periodically update ρ on new (state, action, reward) tuples.
    Note: differs from RLHF in jointly optimising reward and policy.
    """))

if __name__ == "__main__":
    main()
