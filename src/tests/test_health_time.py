from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from starter.datasets import list_frames
from src.health_time import load_timestamps, time_gaps


DATA = Path(__file__).resolve().parents[2] / "data"


def test_synthetic_timestamps_and_gap():
    root = DATA / "synthetic"
    frames = list_frames(root)
    ts = load_timestamps(root, reversed(frames))
    assert ts.index.tolist() == sorted(frames)
    np.testing.assert_allclose(ts, [0.0, 0.1, 0.2, 0.4, 0.5])
    table = time_gaps(ts)
    assert list(table.columns) == ["frame_id", "timestamp_s", "dt_s", "dt_ratio"]
    assert np.isnan(table.dt_s.iloc[0])
    assert np.isnan(table.dt_ratio.iloc[0])
    assert table.dt_s.median() == pytest.approx(0.1)
    row = table.set_index("frame_id").loc["000003"]
    assert row.dt_s == pytest.approx(0.2)
    assert row.dt_ratio == pytest.approx(2.0)


def test_non_monotonic_timestamps_preserve_order():
    ts = pd.Series([0.0, 0.2, 0.1, 0.1, 0.4],
                   index=["000000", "000002", "000001", "000003", "000004"])
    table = time_gaps(ts)
    assert table.frame_id.tolist() == ts.index.tolist()
    assert table.dt_s.iloc[2] == pytest.approx(-0.1)
    assert table.dt_s.iloc[3] == 0


def test_missing_kitti_timestamps():
    root = DATA / "kitti_mini"
    assert load_timestamps(root, list_frames(root)) is None


def test_timestamp_count_mismatch(tmp_path):
    training = tmp_path / "training"
    training.mkdir()
    (training / "timestamps.txt").write_text("0.0\n0.1\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"line count \(2\).*frame count \(3\)"):
        load_timestamps(tmp_path, ["000000", "000001", "000002"])


def test_nuscenes_timestamps_and_scene_boundaries():
    root = DATA / "nuscenes_mini_subset"
    ts = load_timestamps(root, list_frames(root))
    assert len(ts) == 80
    table = time_gaps(ts)
    assert 0.4 <= table.dt_s.median() <= 0.6
    indexed = table.set_index("frame_id")
    for fid in ["scene-0103_000", "scene-1094_000"]:
        assert np.isnan(indexed.loc[fid, "dt_s"])
        assert np.isnan(indexed.loc[fid, "dt_ratio"])
    pd.testing.assert_series_equal(ts, load_timestamps(root, list_frames(root)))


def test_nuscenes_timestamp_cache(tmp_path, monkeypatch):
    from src import health_time

    metadata = tmp_path / "v1.0-mini"
    metadata.mkdir()
    (metadata / "sample.json").write_text("[]", encoding="utf-8")
    calls = []

    def fake_load_frame(root, fid):
        calls.append((root, fid))
        return {"timestamp_lidar_us": 1_500_000}

    monkeypatch.setattr(health_time, "load_frame", fake_load_frame)
    first = load_timestamps(tmp_path, ["scene-test_000"])
    second = load_timestamps(tmp_path, ["scene-test_000"])
    assert first.iloc[0] == 1.5
    pd.testing.assert_series_equal(first, second)
    assert len(calls) == 1


@pytest.mark.parametrize("values, frames", [([], []), ([1.0], ["000000"])])
def test_empty_and_single_frame(values, frames):
    table = time_gaps(pd.Series(values, index=frames, dtype=float))
    assert len(table) == len(values)
    assert table.dt_s.isna().all()
    assert table.dt_ratio.isna().all()
