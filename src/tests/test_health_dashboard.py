from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.health_dashboard import main


DATA = Path(__file__).resolve().parents[2] / "data"


def test_synthetic_dashboard_and_determinism(tmp_path, capsys):
    argv = ["--data-root", str(DATA / "synthetic"), "--out-dir", str(tmp_path)]
    main(argv)
    paths = [tmp_path / name for name in
             ["health_stats_synthetic.csv", "health_flags_synthetic.csv", "health_rules.csv"]]
    first = [path.read_bytes() for path in paths]
    assert (tmp_path / "figures/dashboard_synthetic.png").stat().st_size > 20_000
    flags = pd.read_csv(paths[1], dtype={"frame_id": str})
    assert len(flags) == 5
    assert flags.columns.tolist() == ["frame_id", "status", "n_flags", "flags", "reasons", "action"]
    stats = pd.read_csv(paths[0], dtype={"frame_id": str})
    assert {"timestamp_s", "dt_s", "dt_ratio"} <= set(stats.columns)
    assert stats.loc[stats.frame_id == "000003", "dt_s"].iloc[0] == pytest.approx(0.2)
    output = capsys.readouterr().out
    for row in flags.loc[flags.n_flags > 0].itertuples():
        assert f"{row.frame_id} {row.status}: {row.flags}" in output
    main(argv)
    assert [path.read_bytes() for path in paths] == first


def test_dashboard_without_timestamps(tmp_path):
    root = tmp_path / "dataset"
    points = root / "training/velodyne"
    points.mkdir(parents=True)
    source = DATA / "synthetic/training/velodyne/000000.bin"
    (points / source.name).write_bytes(source.read_bytes())
    out = tmp_path / "output"
    main(["--data-root", str(root), "--out-dir", str(out)])
    assert "dt_s" not in pd.read_csv(out / "health_stats_dataset.csv").columns
    assert (out / "figures/dashboard_dataset.png").stat().st_size > 20_000


def test_help(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
    assert "--data-root" in capsys.readouterr().out
