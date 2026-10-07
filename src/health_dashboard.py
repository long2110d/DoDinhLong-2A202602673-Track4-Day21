"""CPU-only data health dashboard and deterministic CSV exports."""
from __future__ import annotations

import argparse
from pathlib import Path
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

from starter.datasets import load_points
from src.health_stats import (azimuth_hist, dataset_stats, dataset_tag,
                              elevation_hist, raw_arrays)
from src.health_rules import evaluate, rules_table
from src.health_time import load_timestamps, time_gaps


COLORS = {"OK": "#339966", "WARN": "#e6a523", "ERROR": "#d64545"}


def _plot(root: Path, stats: pd.DataFrame, flags: pd.DataFrame,
          path: Path) -> None:
    frames = stats.frame_id.tolist()
    flagged = set(flags.loc[flags.n_flags > 0, "frame_id"])
    arrays = [raw_arrays(root, fid) for fid in frames]
    points = [load_points(root, fid) for fid in frames]
    fig, axes = plt.subplots(3, 3, figsize=(16, 12), dpi=110)
    fig.suptitle(f"Data health: {root.name} | {len(frames)} frames | "
                 f"{len(flagged)} flagged | {int(flags.n_flags.sum())} flags")
    try:
        for ax, key, title in zip(axes[0, :2], ["range", "intensity"],
                                  ["Range histogram", "Intensity histogram"]):
            pooled = np.concatenate([a[key] for a in arrays]) if arrays else np.array([])
            selected = [a[key] for fid, a in zip(frames, arrays) if fid in flagged]
            overlay = np.concatenate(selected) if selected else np.array([])
            bins = np.histogram_bin_edges(pooled, bins=50)
            ax.hist(pooled, bins=bins, color="#507db0", alpha=0.65, label="All frames")
            ax.hist(overlay, bins=bins, color=COLORS["ERROR"], alpha=0.65,
                    label="Flagged frames")
            ax.set(title=title, xlabel="Range (m)" if key == "range" else "Intensity",
                   ylabel="Points")
            ax.legend(fontsize=8)

        x = np.arange(len(frames))
        colors = [COLORS[s] for s in flags.status]
        axes[0, 2].bar(x, stats.n_points, color=colors)
        axes[0, 2].set(title="Points per frame", ylabel="Points")
        axes[0, 2].legend(handles=[Patch(color=c, label=s) for s, c in COLORS.items()],
                          fontsize=8)
        axes[1, 0].bar(x, stats.invalid_ratio, color=colors)
        axes[1, 0].set(title="Invalid ratio per frame", ylabel="Invalid / total")

        for ax, hist_fn, count, title, label in [
            (axes[1, 1], azimuth_hist, 36, "Azimuth density", "Azimuth bin (-180 to 180 deg)"),
            (axes[1, 2], elevation_hist, 16, "Elevation density", "Elevation bin (-30 to 15 deg)"),
        ]:
            hist = np.array([hist_fn(p) for p in points]).reshape(len(frames), count)
            if len(frames):
                im = ax.imshow(np.log1p(hist), aspect="auto", origin="lower",
                               interpolation="nearest", cmap="viridis")
                fig.colorbar(im, ax=ax, label="log(1 + count)", shrink=0.8)
            ax.set(title=title, xlabel=label, ylabel="Frame")

        if "dt_s" in stats:
            axes[2, 0].plot(x, stats.dt_s, "o-", color="#507db0")
            median = stats.dt_s.median()
            if pd.notna(median):
                axes[2, 0].axhline(median, color=COLORS["ERROR"], linestyle="--",
                                   label=f"Median {median:.6g} s")
                axes[2, 0].legend(fontsize=8)
        else:
            axes[2, 0].text(0.5, 0.5, "no timestamps in this dataset",
                            ha="center", va="center", transform=axes[2, 0].transAxes)
        axes[2, 0].set(title="Time gap per frame", ylabel="dt_s (s)")
        axes[2, 1].plot(x, stats.z_p50, "o-", color="#507db0")
        axes[2, 1].set(title="Median height per frame", ylabel="z_p50 (m)")

        step = max(1, int(np.ceil(len(frames) / 30)))
        ticks = x[::step]
        labels = frames[::step]
        for ax in [axes[0, 2], axes[1, 0], axes[2, 0], axes[2, 1]]:
            ax.set_xticks(ticks, labels, rotation=90, fontsize=7)
            ax.set_xlabel("Frame")
        for ax in axes[1, 1:]:
            ax.set_yticks(ticks, labels, fontsize=7)

        panel = axes[2, 2]
        panel.axis("off")
        panel.set_title("Flagged frames and codes")
        rows = flags.loc[flags.n_flags > 0]
        # Use columns for long datasets so every flagged frame remains visible.
        columns = max(1, int(np.ceil(len(rows) / 20)))
        per_column = max(1, int(np.ceil(len(rows) / columns)))
        for i, row in enumerate(rows.itertuples(index=False)):
            codes = textwrap.fill(row.flags.replace(';', ', '), width=max(18, 55 // columns))
            text = f"{row.frame_id} [{row.status}]\n{codes}"
            panel.text((i // per_column) / columns, 0.98 - (i % per_column) / per_column,
                       text, transform=panel.transAxes, va="top",
                       fontsize=max(3, 7 - columns), color=COLORS[row.status], wrap=True)
        if rows.empty:
            panel.text(0.5, 0.5, "No flagged frames", ha="center", va="center")
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        fig.savefig(path, dpi=110)
    finally:
        plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--out-dir", default=Path("results"), type=Path)
    args = parser.parse_args(argv)
    stats = dataset_stats(args.data_root)
    ts = load_timestamps(args.data_root, stats.frame_id)
    gaps = time_gaps(ts) if ts is not None else None
    flags = evaluate(stats, gaps)
    if gaps is not None:
        stats = stats.merge(gaps, on="frame_id", how="left", validate="one_to_one")
    out = args.out_dir
    (out / "figures").mkdir(parents=True, exist_ok=True)
    tag = dataset_tag(args.data_root)
    paths = [out / f"health_stats_{tag}.csv", out / f"health_flags_{tag}.csv",
             out / "health_rules.csv", out / "figures" / f"dashboard_{tag}.png"]
    for table, path in zip([stats, flags, rules_table()], paths):
        table.to_csv(path, index=False, float_format="%.6f", lineterminator="\n")
    _plot(args.data_root, stats, flags, paths[3])
    for row in flags.loc[flags.n_flags > 0].itertuples(index=False):
        print(f"{row.frame_id} {row.status}: {row.flags} -> {row.action}")
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
