"""
Integration test: CLI --ingest flag routes Quest 3 CSV through the loader.
"""
from pathlib import Path
import json
import subprocess
import sys
import pathlib
import os
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

ROOT = Path(__file__).resolve().parent.parent
SAMPLE = ROOT / "samples" / "quest3_tiny.csv"


def test_cli_ingest_shapes():
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "omniintent",
            "demo",
            "--ingest",
            str(SAMPLE),
            "--seq-len",
            "10",
        ],
        capture_output=True,
        text=True,
        check=True,
        env=dict(os.environ, PYTHONPATH=str(ROOT / "src")),
    )
    shapes = json.loads(completed.stdout)
    # Expect batch dim 1, seq dim 10, feature-count as coded in COLUMN_MAP
    assert shapes["gaze"] == [1, 10, 3]
    assert shapes["hand_pose"] == [1, 10, 1]
