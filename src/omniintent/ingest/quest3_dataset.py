"""
quest3_dataset.py
~~~~~~~~~~~~~~~~~
PyTorch `Dataset` that turns a folder (or list) of Quest\u00a03 CSV/Parquet logs
into fixed-length, optionally-overlapping windows of multimodal tensors.

Example
-------
>>> ds = Quest3Dataset("logs/", seq_len=60, stride=30)
>>> batch = torch.utils.data.DataLoader(ds, batch_size=8, shuffle=True)
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable, List, Sequence, Union

import torch
from torch.utils.data import Dataset

from .quest3_ingest import load as _load_q3


class Quest3Dataset(Dataset):  # type: ignore[misc]
    """Create sliding windows over Quest\u00a03 sensor logs."""

    def __init__(
        self,
        files: Union[str, Path, Sequence[Union[str, Path]]],
        seq_len: int = 60,
        stride: int | None = None,
        transform: Callable[[dict[str, torch.Tensor]], dict[str, torch.Tensor]]
        | None = None,
    ):
        """
        Parameters
        ----------
        files
            Directory or list/tuple of CSV/Parquet paths.
        seq_len
            Frames per sample.
        stride
            Overlap between windows.  Defaults to `seq_len` (no overlap).
        transform
            Optional function applied to **each** tensor dict before return.
        """
        self.seq_len = seq_len
        self.stride = stride or seq_len
        self.transform = transform

        # Expand directory into file list
        if isinstance(files, (str, Path)) and Path(files).is_dir():
            self.files: List[Path] = sorted(
                p for p in Path(files).glob("*") if p.suffix in {".csv", ".parquet"}
            )
        else:
            self.files = [Path(f) for f in files]  # type: ignore[arg-type]

        # Pre-compute offsets: (file_idx, start_frame)
        self.index: list[tuple[int, int]] = []
        for f_idx, path in enumerate(self.files):
            # Use loader to infer total frames
            batch = _load_q3(str(path), seq_len=10_000)  # big number \u2192 full file
            # any modal shape is fine (batch=1, frames, feat)
            num_frames = next(iter(batch.values())).shape[1]
            for start in range(0, num_frames - seq_len + 1, self.stride):
                self.index.append((f_idx, start))

    # --------------------------------------------------------------------- #
    def __len__(self) -> int:  # noqa: D401
        """Return number of sliding windows across all logs."""
        return len(self.index)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        f_idx, start = self.index[idx]
        file_path = self.files[f_idx]
        batch = _load_q3(str(file_path), seq_len=start + self.seq_len)

        # cut window & squeeze batch-dim
        window = {
            k: v[0, start : start + self.seq_len] for k, v in batch.items()
        }

        if self.transform:
            window = self.transform(window)
        return window


__all__ = ["Quest3Dataset"]
