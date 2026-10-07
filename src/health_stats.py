"""Extended point-cloud health statistics without output-file side effects."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from starter.datasets import list_frames, load_points


def dataset_tag(data_root: str | Path) -> str:
    name = Path(data_root).name
    return {"synthetic": "synthetic", "kitti_mini": "kitti",
            "nuscenes_mini_subset": "nusc"}.get(name, name)


def _finite_points(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points)
    if points.ndim != 2 or points.shape[1] != 4:
        raise ValueError("points must have shape (N, 4)")
    return points[np.isfinite(points).all(axis=1)].astype(np.float64)


def _arrays(points: np.ndarray) -> dict[str, np.ndarray]:
    p = _finite_points(points)
    return {
        "range": np.linalg.norm(p[:, :3], axis=1),
        "intensity": p[:, 3],
        "az": (np.degrees(np.arctan2(p[:, 1], p[:, 0])) + 180) % 360 - 180,
        "el": np.degrees(np.arctan2(p[:, 2], np.hypot(p[:, 0], p[:, 1]))),
    }


def azimuth_hist(points: np.ndarray, n_bins: int = 36) -> np.ndarray:
    return np.histogram(_arrays(points)["az"], bins=n_bins, range=(-180, 180))[0]


def elevation_hist(points: np.ndarray, n_bins: int = 16,
                   el_range: tuple[float, float] = (-30, 15)) -> np.ndarray:
    return np.histogram(_arrays(points)["el"], bins=n_bins, range=el_range)[0]


def _sector_gap(hist: np.ndarray) -> float:
    longest = run = 0
    for empty in np.tile(hist == 0, 2):
        run = run + 1 if empty else 0
        longest = max(longest, run)
    return min(longest, len(hist)) * 360.0 / len(hist)


def frame_stats(points: np.ndarray, n_az_bins: int = 36, n_el_bins: int = 16,
                el_range: tuple[float, float] = (-30.0, 15.0)) -> dict:
    points = np.asarray(points)
    p = _finite_points(points)
    arrays = _arrays(points)
    rng, intensity = arrays["range"], arrays["intensity"]
    az_hist = azimuth_hist(points, n_az_bins)
    el_hist = elevation_hist(points, n_el_bins, el_range)
    stats = {
        "n_points": len(points),
        "n_invalid": len(points) - len(p),
        "invalid_ratio": (len(points) - len(p)) / len(points) if len(points) else 1.0,
    }
    numeric_keys = (
        "range_min", "range_p05", "range_p50", "range_p95", "range_max",
        "ratio_beyond_50m", "ratio_within_2m", "z_min", "z_p50", "z_max",
        "intensity_mean", "intensity_std", "intensity_zero_ratio", "intensity_sat_ratio",
        "empty_az_bins", "az_min_bin_ratio", "az_max_sector_gap_deg", "empty_el_bins",
        "el_min_deg", "el_max_deg",
    )
    stats.update(dict.fromkeys(numeric_keys, float("nan")))
    if len(p):
        stats.update(zip(numeric_keys[:5], np.percentile(rng, [0, 5, 50, 95, 100])))
        stats.update({
            "ratio_beyond_50m": float((rng > 50).mean()),
            "ratio_within_2m": float((rng < 2).mean()),
            "z_min": float(p[:, 2].min()), "z_p50": float(np.median(p[:, 2])),
            "z_max": float(p[:, 2].max()),
            "intensity_mean": float(intensity.mean()), "intensity_std": float(intensity.std()),
            "intensity_zero_ratio": float((intensity == 0).mean()),
            "intensity_sat_ratio": float((intensity >= 0.99).mean()),
            "empty_az_bins": int((az_hist == 0).sum()),
            "az_min_bin_ratio": float(az_hist.min() / az_hist.mean()),
            "az_max_sector_gap_deg": _sector_gap(az_hist),
            "empty_el_bins": int((el_hist == 0).sum()),
            "el_min_deg": float(arrays["el"].min()), "el_max_deg": float(arrays["el"].max()),
        })
    stats["points_hash"] = hashlib.md5(points.tobytes()).hexdigest()[:12]
    return stats


def dataset_stats(data_root: str | Path) -> pd.DataFrame:
    rows = [{"frame_id": fid, **frame_stats(load_points(data_root, fid))}
            for fid in sorted(list_frames(data_root))]
    columns = ["frame_id", *frame_stats(np.empty((0, 4))).keys()]
    return pd.DataFrame(rows, columns=columns).round(6)


def raw_arrays(data_root: str | Path, frame_id: str) -> dict[str, np.ndarray]:
    return _arrays(load_points(data_root, frame_id))
