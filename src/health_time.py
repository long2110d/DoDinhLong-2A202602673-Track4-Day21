"""Timestamp loading and scene-aware time-gap analysis."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Iterable

import pandas as pd

from starter.datasets import load_frame
from starter.nuscenes_io import is_nuscenes_root


@lru_cache(maxsize=512)
def _lidar_timestamp(data_root: str, frame_id: str) -> float:
    return float(load_frame(data_root, frame_id)["timestamp_lidar_us"]) / 1e6


def load_timestamps(data_root: str | Path,
                    frame_ids: Iterable[str]) -> pd.Series | None:
    """Return seconds indexed by sorted frame IDs, or None if unavailable."""
    root = Path(data_root)
    frames = sorted(frame_ids)
    if is_nuscenes_root(root):
        values = [_lidar_timestamp(str(root.resolve()), fid) for fid in frames]
    else:
        path = root / "training" / "timestamps.txt"
        if not path.exists():
            return None
        lines = path.read_text(encoding="utf-8").splitlines()
        if len(lines) != len(frames):
            raise ValueError(
                f"{path}: timestamp line count ({len(lines)}) does not match "
                f"frame count ({len(frames)})"
            )
        values = [float(line) for line in lines]
    return pd.Series(values, index=frames, dtype=float, name="timestamp_s")


def time_gaps(ts: pd.Series) -> pd.DataFrame:
    """Compute gaps in input order, excluding gaps across scene boundaries."""
    timestamps = ts.astype(float).reset_index(drop=True)
    frames = pd.Series(ts.index, dtype=str).reset_index(drop=True)
    scenes = frames.map(lambda fid: fid.rsplit("_", 1)[0] if "_" in fid else "")
    gaps = timestamps.diff().mask(scenes.ne(scenes.shift()))
    median = gaps.median() if gaps.notna().any() else float("nan")
    return pd.DataFrame({
        "frame_id": frames,
        "timestamp_s": timestamps,
        "dt_s": gaps,
        "dt_ratio": gaps / median,
    })
