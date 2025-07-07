"""OREO algorithm stub.

Off-policy REinforcement learning with Oracles (OREO) is briefly
mentioned in the post-training survey. No public implementation is
available, so this script prints a concise overview of what the paper
introduces: combining offline trajectories (oracle demonstrations) with
online RL updates.
"""
import textwrap

def main():
    print(textwrap.dedent("""
    OREO Idea
    =========
    1. Gather ORACLE trajectories (expert or synthetic) → D_offline.
    2. Initialise policy π with supervised learning on D_offline.
    3. Interact with environment to collect D_online.
    4. Optimise π with a mixture objective:
         L = L_offline + λ · L_online
       where L_offline is behavioural-cloning loss on oracle data,
       and L_online is RL loss (e.g. PPO) on new rollouts.
    5. Anneal λ to shift from imitation to RL.
    """))

if __name__ == "__main__":
    main()
