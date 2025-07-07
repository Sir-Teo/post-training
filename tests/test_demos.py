import subprocess, sys, pytest
from pathlib import Path

DEMOS = [
    "sft.py",
    "rlhf_ppo.py",
    "rlaif_ppo.py",
    "dpo.py",
    "cot_prompt.py",
    "self_consistency.py",
    "tree_of_thoughts.py",
    "reward_modeling.py",
    "self_feedback.py",
    "episodic_memory.py",  # run once to ensure startup
]
ROOT = Path(__file__).resolve().parents[1]
DEMOS_DIR = ROOT / "demos"

@pytest.mark.parametrize("script", DEMOS)
def test_demo_runs(script):
    path = DEMOS_DIR / script
    # In CI tests we only check the script parses args and exits.
    cmd = [sys.executable, str(path), "--no_run"]
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    assert res.returncode == 0, f"{script} failed: {res.stderr}"
