from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from src.health_stats import (
    azimuth_hist, dataset_stats, dataset_tag, elevation_hist, frame_stats, raw_arrays,
)


def test_finite_rows_and_3d_statistics():
    points = np.array([[3, 4, 12, 0], [1, 0, 0, 1], [60, 0, 0, 0.5],
                       [np.nan, 0, 0, 0]])
    stats = frame_stats(points)
    assert stats["n_points"] == 4
    assert stats["n_invalid"] == 1
    assert stats["invalid_ratio"] == 0.25
    assert stats["range_p50"] == 13
    assert stats["range_p05"] == pytest.approx(2.2)
    assert stats["range_p95"] == pytest.approx(55.3)
    assert stats["ratio_beyond_50m"] == pytest.approx(1 / 3)
    assert stats["ratio_within_2m"] == pytest.approx(1 / 3)
    assert stats["intensity_sat_ratio"] == pytest.approx(1 / 3)
    assert stats["intensity_std"] == pytest.approx(np.std([0, 1, 0.5]))


def test_quadrant_coverage_and_zero_intensity():
    angles = np.radians(np.arange(5, 90, 10))
    points = np.column_stack((np.cos(angles), np.sin(angles),
                              np.zeros(9), np.zeros(9)))
    stats = frame_stats(points)
    assert stats["empty_az_bins"] == 27
    assert stats["az_max_sector_gap_deg"] == 270
    assert stats["az_min_bin_ratio"] == 0
    assert stats["intensity_zero_ratio"] == 1


def test_hash_uses_original_bytes():
    points = np.array([[1, 2, 3, 0]], dtype=np.float32)
    original = points.copy()
    assert frame_stats(points)["points_hash"] == frame_stats(points.copy())["points_hash"]
    assert frame_stats(points)["points_hash"] == hashlib.md5(points.tobytes()).hexdigest()[:12]
    assert frame_stats(points)["points_hash"] != frame_stats(points + 1)["points_hash"]
    np.testing.assert_array_equal(points, original)


@pytest.mark.parametrize("points", [np.full((3, 4), np.nan), np.empty((0, 4))])
def test_no_finite_rows(points):
    stats = frame_stats(points)
    assert stats["invalid_ratio"] == 1
    assert stats["n_invalid"] == len(points)
    for key, value in stats.items():
        if key not in {"n_points", "n_invalid", "invalid_ratio", "points_hash"}:
            assert np.isnan(value), key
    assert azimuth_hist(points).sum() == elevation_hist(points).sum() == 0


def test_histogram_boundaries_and_invalid_intensity():
    points = np.array([[-1, 0, 0, 0], [-1, -0.0, 0, 0],
                       [1, 0, 1, 0], [1, 0, 0, np.inf]])
    np.testing.assert_array_equal(azimuth_hist(points, 4), [2, 0, 1, 0])
    np.testing.assert_array_equal(elevation_hist(points, 3, (-45, 45)), [0, 2, 1])


@pytest.mark.parametrize("name, tag", [("synthetic", "synthetic"), ("kitti_mini", "kitti"),
                                     ("nuscenes_mini_subset", "nusc"), ("custom", "custom")])
def test_dataset_tag(name, tag):
    assert dataset_tag(Path("data") / name) == tag


def test_synthetic_integration():
    root = Path(__file__).resolve().parents[2] / "data" / "synthetic"
    table = dataset_stats(root)
    assert len(table) == 5
    assert list(table.columns) == ["frame_id", *frame_stats(np.empty((0, 4))).keys()]
    assert table.frame_id.tolist() == sorted(table.frame_id)
    assert table.to_csv(index=False) == dataset_stats(root).to_csv(index=False)
    arrays = raw_arrays(root, table.frame_id.iloc[0])
    assert set(arrays) == {"range", "intensity", "az", "el"}
    assert all(a.ndim == 1 and np.isfinite(a).all() for a in arrays.values())
    assert len(arrays["range"]) == table.n_points.iloc[0] - table.n_invalid.iloc[0]
