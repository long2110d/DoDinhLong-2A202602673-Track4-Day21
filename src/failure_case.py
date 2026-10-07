"""Demonstrate LOW_POINTS threshold transfer and azimuth convention failures."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from starter.datasets import list_frames, load_points
from starter.perturb import random_dropout
from src.health_rules import LOW_POINTS_FACTOR
from src.health_stats import azimuth_hist


def failure_case(kitti: list[np.ndarray], nusc: list[np.ndarray],
                 out_dir: str | Path, seed: int = 0) -> pd.DataFrame:
    """Compare clean clouds and seeded 50% retention against clean baselines.

    Only LOW_POINTS is evaluated. Clean means unperturbed, not that every
    other dashboard rule considers these frames healthy.
    """
    if not kitti or not nusc:
        raise ValueError("Both datasets must contain frames")
    counts = {"kitti": np.array([len(p) for p in kitti]),
              "nusc": np.array([len(p) for p in nusc]),
              "nusc_dropout_50": np.array([
                  len(random_dropout(p, keep_ratio=0.5, seed=seed + i))
                  for i, p in enumerate(nusc)])}
    absolute = LOW_POINTS_FACTOR * np.median(counts["kitti"])
    relative = {name: LOW_POINTS_FACTOR * np.median(counts[name])
                for name in ("kitti", "nusc")}
    relative["nusc_dropout_50"] = relative["nusc"]
    rows = []
    for variant in ("absolute", "relative"):
        for name, values in counts.items():
            threshold = absolute if variant == "absolute" else relative[name]
            flagged = int((values < threshold).sum())
            rows.append({"rule_variant": variant, "dataset": name,
                         "n_frames": len(values), "n_flagged": flagged,
                         "flagged_ratio": flagged / len(values)})
    table = pd.DataFrame(rows).round(6)
    out_dir = Path(out_dir)
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)
    table.to_csv(out_dir / "failure_case.csv", index=False,
                 float_format="%.6f", lineterminator="\n")

    fig, axes = plt.subplots(1, 3, figsize=(17, 5), layout="constrained")
    colors = {"kitti": "tab:blue", "nusc": "tab:orange",
              "nusc_dropout_50": "tab:red"}
    for ax, variant in zip(axes[:2], ("absolute", "relative")):
        for name, values in counts.items():
            x = np.arange(len(values))
            ax.plot(x, values, ".", color=colors[name], label=name)
        if variant == "absolute":
            ax.axhline(absolute, color="black", linestyle="--",
                       label=f"KITTI threshold: {absolute:,.0f}")
        else:
            for name in ("kitti", "nusc"):
                ax.axhline(relative[name], color=colors[name], linestyle="--",
                           label=f"{name} threshold: {relative[name]:,.0f}")
        ax.set(title=f"LOW_POINTS: {variant} threshold", xlabel="Sorted frame index",
               ylabel="Points per frame")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2)
    centers = np.linspace(-175, 175, 36)
    for name, clouds in (("kitti", kitti), ("nusc", nusc)):
        hist = np.sum([azimuth_hist(p) for p in clouds], axis=0)
        density = hist / hist.sum() if hist.sum() else np.zeros(36)
        axes[2].plot(centers, density, color=colors[name], label=name)
    for angle, name, label in ((0, "kitti", "KITTI front: 0°"),
                               (90, "nusc", "nuScenes front: +90°")):
        axes[2].axvline(angle, color=colors[name], linestyle="--", label=label)
    axes[2].set(title="Azimuth conventions", xlabel="atan2(y, x) (degrees)",
                ylabel="Fraction of returns per 10° bin", xlim=(-180, 180))
    axes[2].text(0.02, 0.97, "nuScenes 0° = right; KITTI front-sector\ncoordinates watch the wrong direction.",
                 transform=axes[2].transAxes, va="top", fontsize=8)
    axes[2].legend(loc="lower right", fontsize=8)
    fig.suptitle("Metric / preprocessing failure: transferring KITTI thresholds to nuScenes\n"
                 "Unperturbed frames vs seeded 50% random retention; strict count < threshold")
    try:
        fig.savefig(out_dir / "figures" / "fail_01_kitti_threshold_on_nuscenes.png", dpi=150)
    finally:
        plt.close(fig)
    return table


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data"),
                        help="Parent directory containing both dataset folders")
    parser.add_argument("--out-dir", type=Path, default=Path("results"))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    clouds = []
    for folder in ("kitti_mini", "nuscenes_mini_subset"):
        root = args.data_root / folder
        clouds.append([load_points(root, fid) for fid in sorted(list_frames(root))])
    print(failure_case(*clouds, args.out_dir, args.seed).to_string(index=False))


if __name__ == "__main__":
    main()
