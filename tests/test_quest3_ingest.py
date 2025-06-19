import torch, tempfile, pandas as pd, numpy as np, pathlib, json, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from omniintent.ingest.quest3_ingest import load

def test_ingest_shapes(tmp_path):
    # generate fake 60\u2011frame CSV
    rows = []
    for i in range(60):
        rows.append(
            {"timestamp_ns": i * 8_333_333,
             "left_gaze_yaw": np.random.rand(),
             "left_gaze_pitch": np.random.rand(),
             "fix_conf": 0.8,
             "hand_pose_0_x": np.random.rand()}
        )
    csv = tmp_path / "q3.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)
    batch = load(csv, seq_len=60)
    assert batch["gaze"].shape == (1, 60, 3)
    assert batch["hand_pose"].shape[1] == 60
