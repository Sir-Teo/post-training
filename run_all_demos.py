"""Quick comparison runner.

Executes each runnable demo in `demos/` sequentially, captures stdout,
and prints the first few lines under a header so you can eyeball how
each post-training method behaves on the same machine.

Usage:
    python run_all_demos.py --lines 8
"""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEMOS_DIR = ROOT / "demos"

# Demo scripts to run in order (skip algorithm stubs)
ORDER = [
    "sft.py",
    "rlhf_ppo.py",
    "rlaif_ppo.py",
    "dpo.py",
    "cot_prompt.py",
    "self_consistency.py",
    "tree_of_thoughts.py",
    "reward_modeling.py",
    "self_feedback.py",
]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lines", type=int, default=8, help="Lines of output to show per demo")
    args = parser.parse_args()

    for script in ORDER:
        path = DEMOS_DIR / script
        print("\n" + "=" * 60)
        print(f">>> {script}")
        print("=" * 60)
        result = subprocess.run([sys.executable, str(path)], capture_output=True, text=True, timeout=180)
        if result.returncode != 0:
            print(f"[ERROR] {script} exited with code {result.returncode}\n", result.stderr)
        else:
            out_lines = result.stdout.strip().splitlines()[: args.lines]
            for l in out_lines:
                print(l)

if __name__ == "__main__":
    main()
