"""
quest3_ingest.py
~~~~~~~~~~~~~~~~
Load Quest\u00a03 mixed-reality sensor logs (CSV or Parquet) and convert them into
the batched tensor dict expected by MultiModalTokenizer.

Expected columns
---------------
timestamp_ns | gyro_x | gyro_y | gyro_z | accel_x | ... | left_gaze_yaw | ...
"""
from pathlib import Path
import pandas as pd
import torch
import numpy as np

# Map CSV columns \u2192 (modal, feature_idx)
_COLUMN_MAP = {
    "left_gaze_yaw": ("gaze", 0),
    "left_gaze_pitch": ("gaze", 1),
    "fix_conf": ("gaze", 2),
    "hand_pose_0_x": ("hand_pose", 0),  # etc…
}

def load(path: str | Path, seq_len: int = 60) -> dict[str, torch.Tensor]:
    p = Path(path)
    df = pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)
    # normalise timestamp \u2192 frame index
    df["frame"] = (df["timestamp_ns"] - df["timestamp_ns"].min()) // 8_333_333  # 120 Hz
    grouped = df.groupby("frame").mean().iloc[:seq_len]
    modal = {"gaze": [], "hand_pose": []}
    for col, (m, idx) in _COLUMN_MAP.items():
        modal[m].append(torch.tensor(grouped[col].values, dtype=torch.float32).unsqueeze(-1))
    return {k: torch.cat(v, dim=-1).unsqueeze(0) for k, v in modal.items()}

