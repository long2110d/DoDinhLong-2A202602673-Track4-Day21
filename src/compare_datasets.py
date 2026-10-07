"""Compare dashboard CSVs and export measured synthetic fault evidence."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.health_rules import RULES


def compare_datasets(out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    synthetic_flags = None
    for tag in ("synthetic", "kitti", "nusc"):
        stats = pd.read_csv(out_dir / f"health_stats_{tag}.csv", dtype={"frame_id": str})
        flags = pd.read_csv(out_dir / f"health_flags_{tag}.csv", dtype={"frame_id": str}).fillna("")
        if tag == "synthetic":
            synthetic_flags = flags
        table = stats.merge(flags[["frame_id", "status"]], on="frame_id",
                            how="left", validate="one_to_one")
        if table.status.isna().any() or not table.status.isin(["OK", "WARN", "ERROR"]).all():
            raise ValueError(f"Missing or invalid frame status for {tag}")
        groups = [(tag, table)] if tag != "nusc" else [
            ("nusc_day", table.loc[table.frame_id.str.startswith("scene-0103_")]),
            ("nusc_night", table.loc[table.frame_id.str.startswith("scene-1094_")]),
        ]
        for name, group in groups:
            row = {"dataset": name}
            for key in ("n_points", "range_p95", "intensity_mean", "empty_az_bins"):
                row[key] = group[key].median()
            row["el_span_deg"] = (group.el_max_deg - group.el_min_deg).median()
            row["dt_s"] = group.dt_s.median() if "dt_s" in group else float("nan")
            for status in ("OK", "WARN", "ERROR"):
                row[f"{status.lower()}_pct"] = 100 * group.status.eq(status).mean()
            rows.append(row)

    descriptions = {rule.code: rule.description for rule in RULES}
    faults = []
    for row in synthetic_flags.sort_values("frame_id").itertuples(index=False):
        reasons = dict(reason.split(": ", 1) for reason in row.reasons.split(" | ") if ": " in reason)
        for code in filter(None, row.flags.split(";")):
            if code not in reasons:
                raise ValueError(f"Missing evidence for {row.frame_id}: {code}")
            faults.append({"fault": descriptions.get(code, code), "frame_id": row.frame_id,
                           "detected_by_rule": code, "evidence_value": reasons[code]})
    return pd.DataFrame(rows), pd.DataFrame(faults, columns=[
        "fault", "frame_id", "detected_by_rule", "evidence_value"])


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("results"),
                        help="Directory containing dashboard CSVs and receiving comparison CSVs")
    args = parser.parse_args(argv)
    comparison, faults = compare_datasets(args.out_dir)
    for table, name in [(comparison, "dataset_comparison.csv"),
                        (faults, "synthetic_planted_faults.csv")]:
        path = args.out_dir / name
        table.to_csv(path, index=False, float_format="%.6f", lineterminator="\n")
        print(path)


if __name__ == "__main__":
    main()
