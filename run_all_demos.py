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
    parser = argparse.ArgumentParser(description="Run all demo scripts in lightweight mode")
    parser.add_argument("--lines", type=int, default=8, help="Lines of output to show per demo")
    parser.add_argument("--full", action="store_true", help="Run demos fully (may be slow)")
    args = parser.parse_args()

    # Custom quick-run overrides to keep runtimes short even in --full mode
    OVERRIDES = {
        "sft.py": ["--epochs", "1", "--quick"],
        "rlhf_ppo.py": ["--steps", "5"],
        "rlaif_ppo.py": ["--pretrain_steps", "10", "--ppo_steps", "5"],
        "dpo.py": ["--steps", "5"],
    }

    for script in ORDER:
        path = DEMOS_DIR / script
        print("\n" + "=" * 60)
        print(f">>> {script}")
        print("=" * 60)
        cmd = [sys.executable, str(path)]
        if not args.full:
            cmd.append("--no_run")
        else:
            cmd.extend(OVERRIDES.get(script, []))
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            print(f"[ERROR] {script} exited with code {result.returncode}\n", result.stderr)
        else:
            out_lines = result.stdout.strip().splitlines()[: args.lines]
            for l in out_lines:
                print(l)

if __name__ == "__main__":
    main()
