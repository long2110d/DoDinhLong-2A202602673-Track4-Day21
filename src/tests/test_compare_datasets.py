from __future__ import annotations

import pandas as pd
import pytest

from src.compare_datasets import main


def test_comparison_and_fault_evidence(tmp_path):
    for tag, ids in [("synthetic", ["000000", "000001"]),
                     ("kitti", ["000011", "000012"]),
                     ("nusc", ["scene-0103_000", "scene-0103_001", "scene-1094_000"])]:
        n = len(ids)
        stats = pd.DataFrame({"frame_id": ids, "n_points": [10, 30, 90][:n],
                              "range_p95": [20, 40, 80][:n],
                              "intensity_mean": [0.1, 0.3, 0.9][:n],
                              "empty_az_bins": [0, 2, 4][:n],
                              "el_min_deg": [-10, -20, -30][:n],
                              "el_max_deg": [5, 10, 15][:n]})
        if tag != "kitti":
            stats["dt_s"] = [float("nan"), 0.2, 0.4][:n]
        stats.to_csv(tmp_path / f"health_stats_{tag}.csv", index=False)
        flags = pd.DataFrame({"frame_id": ids, "status": ["OK", "WARN", "ERROR"][:n],
                              "flags": ["", "Z_SHIFT;TIME_GAP", "LOW_POINTS"][:n],
                              "reasons": ["", "Z_SHIFT: z_p50=1; abs(value-median(0)) > 0.3 m | TIME_GAP: dt_ratio=2 > 1.5",
                                          "LOW_POINTS: n_points=90 < 100"][:n]})
        flags.iloc[::-1].to_csv(tmp_path / f"health_flags_{tag}.csv", index=False)
    main(["--out-dir", str(tmp_path)])
    paths = [tmp_path / "dataset_comparison.csv", tmp_path / "synthetic_planted_faults.csv"]
    first = [p.read_bytes() for p in paths]
    comparison = pd.read_csv(paths[0]).set_index("dataset")
    assert comparison.index.tolist() == ["synthetic", "kitti", "nusc_day", "nusc_night"]
    assert comparison.loc["nusc_day", "n_points"] == 20
    assert comparison.loc["nusc_night", "n_points"] == 90
    assert comparison.loc["nusc_day", "el_span_deg"] == 22.5
    assert comparison.loc["nusc_day", "dt_s"] == 0.2
    assert pd.isna(comparison.loc["kitti", "dt_s"])
    assert comparison.loc["synthetic", ["ok_pct", "warn_pct", "error_pct"]].tolist() == [50, 50, 0]
    assert comparison.loc["nusc_night", "error_pct"] == 100
    faults = pd.read_csv(paths[1], dtype={"frame_id": str})
    assert faults.columns.tolist() == ["fault", "frame_id", "detected_by_rule", "evidence_value"]
    assert faults.frame_id.tolist() == ["000001", "000001"]
    assert faults.detected_by_rule.tolist() == ["Z_SHIFT", "TIME_GAP"]
    assert faults.evidence_value.tolist() == ["z_p50=1; abs(value-median(0)) > 0.3 m", "dt_ratio=2 > 1.5"]
    main(["--out-dir", str(tmp_path)])
    assert [p.read_bytes() for p in paths] == first


def test_help(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
    assert "--out-dir" in capsys.readouterr().out
