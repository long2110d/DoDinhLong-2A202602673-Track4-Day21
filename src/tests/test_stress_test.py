from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import pandas as pd

from src.health_rules import evaluate


def test_frozen_reference_median():
    clean = pd.DataFrame({"frame_id": ["a", "b"], "n_points": [100, 200]})
    target = pd.DataFrame({"frame_id": ["z"], "n_points": [60]})
    combined = pd.concat([clean, target], ignore_index=True)
    assert "LOW_POINTS" in evaluate(combined, reference_stats=clean).iloc[-1]["flags"]
    assert "LOW_POINTS" not in evaluate(combined).iloc[-1]["flags"]


def test_stress_cli(tmp_path):
    root = Path(__file__).resolve().parents[2]
    command = [sys.executable, "-m", "src.stress_test", "--data-root",
               str(root / "data/synthetic"), "--out-dir", str(tmp_path), "--repeats", "3"]
    subprocess.run(command, cwd=root, check=True)
    csv = tmp_path / "stress_detection.csv"
    first = csv.read_bytes()
    table = pd.read_csv(csv)
    assert len(table.groupby("perturbation")) == 4
    assert (table.groupby("perturbation")["level"].nunique() >= 3).all()
    dropout = table[table.perturbation == "random_dropout"]
    assert dropout.loc[dropout.level == 1, "detection_rate"].item() == 0
    assert dropout.loc[dropout.level == .3, "detection_rate"].item() == 1
    assert dropout.detection_rate.is_monotonic_increasing
    latency = pd.read_csv(tmp_path / "stress_latency.csv")
    assert (latency.repeats == 3).all()
    assert (latency.n_samples == latency.n_frames * 3).all()
    assert (latency.p95_ms >= latency.p50_ms).all()
    assert (tmp_path / "figures/stress_detection.png").stat().st_size > 0
    subprocess.run(command, cwd=root, check=True)
    assert csv.read_bytes() == first
