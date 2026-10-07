"""Controlled degradation sweeps and CPU frame-statistics benchmarks."""
from __future__ import annotations

import argparse
from pathlib import Path
import platform
from time import perf_counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from starter.datasets import list_frames, load_points
from starter.perturb import gaussian_noise, random_dropout, sector_dropout
from src.health_rules import LOW_POINTS_FACTOR, SECTOR_GAP_DEG, Z_SHIFT_M, evaluate
from src.health_stats import dataset_tag, frame_stats

SWEEPS = {
    "random_dropout": ([1.0, 0.9, 0.7, 0.5, 0.3], "LOW_POINTS", "n_points"),
    "sector_dropout": ([0, 15, 30, 60, 90], "SECTOR_GAP", "az_max_sector_gap_deg"),
    "gaussian_noise": ([0, 0.02, 0.05, 0.1, 0.3], "", "range_p95"),
    "z_offset": ([0, 0.1, 0.2, 0.4, 0.8], "Z_SHIFT", "z_p50"),
}


def degrade(points: np.ndarray, perturbation: str, level: float) -> np.ndarray:
    if perturbation == "random_dropout":
        return random_dropout(points, level, seed=0)
    if level == 0:
        return points.copy()
    if perturbation == "sector_dropout":
        return sector_dropout(points, -level / 2, level / 2)
    if perturbation == "gaussian_noise":
        return gaussian_noise(points, sigma_xyz_m=level, seed=0)
    if perturbation == "z_offset":
        shifted = points.copy()
        shifted[:, 2] += level
        return shifted
    raise ValueError(f"Unknown perturbation: {perturbation}")


def detection_sweep(data_root: Path) -> pd.DataFrame:
    frames = sorted(list_frames(data_root))
    if not frames:
        raise ValueError("The dataset has no frames")
    clouds = [load_points(data_root, fid) for fid in frames]
    clean = pd.DataFrame([{"frame_id": fid, **frame_stats(p)}
                          for fid, p in zip(frames, clouds)])
    # Reference copies are context, not additional captured frames.
    clean["points_hash"] = None
    rows = []
    for name, (levels, expected, metric) in SWEEPS.items():
        for level in levels:
            measured, flags = [], []
            for points in clouds:
                stats = frame_stats(degrade(points, name, level))
                measured.append(stats)
                target = {"frame_id": "~degraded", **stats}
                result = evaluate(pd.concat([clean, pd.DataFrame([target])], ignore_index=True),
                                  reference_stats=clean)
                flags.append(set(filter(None, result.iloc[-1]["flags"].split(";"))))
            rows.append({
                "perturbation": name, "level": level, "n_frames": len(frames),
                "detection_rate": np.mean([expected in f if expected else bool(f) for f in flags]),
                "any_flag_rate": np.mean([bool(f) for f in flags]),
                "mean_n_points": np.mean([s["n_points"] for s in measured]),
                "mean_metric": np.mean([s[metric] for s in measured]),
                "config": f"seed=0;frames={data_root.name}({len(frames)})",
                "expected_rule": expected or "ANY", "metric": metric,
                "fired_rules": ";".join(sorted(set().union(*flags))),
            })
    return pd.DataFrame(rows).round(6)


def latency_benchmark(roots: list[Path], repeats: int) -> pd.DataFrame:
    if repeats < 1:
        raise ValueError("repeats must be positive")
    rows = []
    for root in roots:
        frames = sorted(list_frames(root))
        samples = []
        for fid in frames:
            points = load_points(root, fid)
            frame_stats(points)
            for _ in range(repeats):
                start = perf_counter()
                frame_stats(points)
                samples.append((perf_counter() - start) * 1000)
        rows.append({"dataset": dataset_tag(root), "n_frames": len(frames),
                     "repeats": repeats, "n_samples": len(samples),
                     "p50_ms": np.percentile(samples, 50),
                     "p95_ms": np.percentile(samples, 95),
                     "processor": platform.processor(), "platform": platform.platform()})
    return pd.DataFrame(rows).round(6)


def plot_detection(table: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    thresholds = {"random_dropout": LOW_POINTS_FACTOR,
                  "sector_dropout": SECTOR_GAP_DEG, "z_offset": Z_SHIFT_M}
    units = {"random_dropout": "Keep ratio", "sector_dropout": "Sector width (deg)",
             "gaussian_noise": "XYZ noise sigma (m)", "z_offset": "Z offset (m)"}
    for ax, name in zip(axes.flat, SWEEPS):
        group = table[table.perturbation == name]
        ax.plot(group.level, group.detection_rate, "o-", label=group.expected_rule.iloc[0])
        ax.plot(group.level, group.any_flag_rate, "s--", label="Any flag", alpha=0.6)
        if name in thresholds:
            ax.axvline(thresholds[name], color="red", linestyle=":", label="Rule threshold")
        else:
            ax.text(.04, .08, "No noise-specific rule threshold", transform=ax.transAxes)
        if name == "random_dropout":
            ax.invert_xaxis()
        ax.set(title=name, xlabel=units[name], ylabel="Detection rate", ylim=(-.03, 1.05))
        ax.grid(alpha=.25)
        ax.legend(fontsize=8)
    fig.suptitle("One factor per sweep; seed 0; clean reference context")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/kitti_mini"))
    parser.add_argument("--out-dir", type=Path, default=Path("results"))
    parser.add_argument("--repeats", type=int, default=20)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    figures = args.out_dir / "figures"
    figures.mkdir(exist_ok=True)
    detection = detection_sweep(args.data_root)
    detection.to_csv(args.out_dir / "stress_detection.csv", index=False, float_format="%.6f")
    nusc = Path(__file__).resolve().parents[1] / "data/nuscenes_mini_subset"
    latency_benchmark([args.data_root, nusc], args.repeats).to_csv(
        args.out_dir / "stress_latency.csv", index=False, float_format="%.6f")
    plot_detection(detection, figures / "stress_detection.png")


if __name__ == "__main__":
    main()
