from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from starter.perturb import random_dropout
from src.failure_case import failure_case, main


def test_threshold_transfer_and_clean_reference(tmp_path):
    kitti = [np.tile([1., 0., 0., 0.5], (1000, 1)) for _ in range(2)]
    nusc = [np.tile([0., 1., 0., 0.5], (300, 1)) for _ in range(3)]
    table = failure_case(kitti, nusc, tmp_path)
    path = tmp_path / "failure_case.csv"
    assert pd.read_csv(path).columns.tolist() == [
        "rule_variant", "dataset", "n_frames", "n_flagged", "flagged_ratio"]
    indexed = table.set_index(["rule_variant", "dataset"])
    assert indexed.loc[("absolute", "nusc"), "n_flagged"] == 3
    assert indexed.loc[("absolute", "nusc_dropout_50"), "n_flagged"] == 3
    assert indexed.loc[("relative", "nusc"), "n_flagged"] == 0
    expected = sum(len(random_dropout(p, 0.5, seed=i)) < 150
                   for i, p in enumerate(nusc))
    assert indexed.loc[("relative", "nusc_dropout_50"), "n_flagged"] == expected
    assert indexed.loc[("absolute", "kitti"), "n_flagged"] == 0
    png, = (tmp_path / "figures").glob("*.png")
    assert png.name == "fail_01_kitti_threshold_on_nuscenes.png"
    assert png.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    first = path.read_bytes()
    failure_case(kitti, nusc, tmp_path)
    assert path.read_bytes() == first


def test_empty_dataset(tmp_path):
    with pytest.raises(ValueError, match="Both datasets"):
        failure_case([], [], tmp_path)


def test_help(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
    assert "--data-root" in capsys.readouterr().out
