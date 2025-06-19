import numpy as np, pandas as pd, torch
from omniintent.ingest.quest3_dataset import Quest3Dataset


def _make_csv(path, frames=120):
    rows = [
        {
            "timestamp_ns": i * 8_333_333,
            "left_gaze_yaw": np.random.rand(),
            "left_gaze_pitch": np.random.rand(),
            "fix_conf": 0.8,
            "hand_pose_0_x": np.random.rand(),
        }
        for i in range(frames)
    ]
    pd.DataFrame(rows).to_csv(path, index=False)


def test_dataset_len_and_shapes(tmp_path):
    # two logs, 120 frames each \u2192 with seq_len=60, stride=30:
    # windows per file: 0,30,60 \u2192 3 windows \u00d7 2 files = 6 samples
    paths = []
    for n in range(2):
        p = tmp_path / f"log{n}.csv"
        _make_csv(p, frames=120)
        paths.append(p)

    ds = Quest3Dataset(paths, seq_len=60, stride=30)

    assert len(ds) == 6

    sample = ds[0]
    assert sample["gaze"].shape == (60, 3)
    assert sample["hand_pose"].shape == (60, 1)
    # Enforce Tensor dtype
    assert isinstance(sample["gaze"], torch.Tensor)


def test_dataset_transform(tmp_path):
    p = tmp_path / "log.csv"
    _make_csv(p, frames=60)

    def add_norm(batch):
        batch["norm"] = torch.norm(batch["gaze"], dim=-1, keepdim=True)
        return batch

    ds = Quest3Dataset([p], seq_len=60, transform=add_norm)
    sample = ds[0]
    assert "norm" in sample
    assert sample["norm"].shape == (60, 1)
