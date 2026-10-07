# PLAN — Day 6 Lab, Topic E (Data health dashboard), target level: Good

## Project overview

VinUni AI20K Track 4, Day 6 lab ("3D from Point Clouds & LiDAR-Camera Projection"). We do **topic E — Data health
dashboard** at level **Good** (plus cheap bonus items B4/B5/B6). Everything runs on CPU, no GPU, no model.

**Stack:** Python 3.10+ (machine has 3.11), virtualenv `.venv`, `numpy`, `opencv-python`, `matplotlib`, `pandas`
(already in `requirements.txt`) + `pytest` (added in Task 1). Matplotlib must use the non-interactive backend
(`matplotlib.use("Agg")` at the top of every module that plots).

**Setup (once):**
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    | Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

**Repo layout (relevant parts):**
```
starter/            # instructor code. DO NOT EDIT, except the 2 TODO(CP2) functions in starter/projection.py
  datasets.py       #   list_frames(root), load_points(root, fid) -> (N,4) float32 x,y,z,intensity(0..1), load_frame(...)
  kitti_io.py       #   KittiCalib (P2, R0_rect, Tr_velo_to_cam, property T_cam_velo 4x4)
  nuscenes_io.py    #   load_frame(...) also returns "timestamp_lidar_us", "timestamp_camera_us"
  projection.py     #   velo_to_cam / cam_to_image are TODO(CP2)
  data_health.py    #   baseline per-frame stats -> CSV (reuse point_stats idea, do not edit)
  perturb.py        #   random_dropout, sector_dropout, gaussian_noise, range_dropout, beam_dropout (seeded)
data/               # READ-ONLY. Never modify or add files here.
  synthetic/        #   5 KITTI-format frames 000000..000004 + training/timestamps.txt (has PLANTED faults)
  kitti_mini/       #   20 KITTI frames (64-beam), no timestamps
  nuscenes_mini_subset/  # 80 frames scene-0103_000..039 (day), scene-1094_000..039 (night); 32-beam;
                         # LiDAR x points RIGHT, y FORWARD (azimuth 0° = right side of car)
src/                # ALL our own code goes here (it is a Python package: src/__init__.py)
  tests/            #   pytest tests (our code too, so it lives in src/)
results/            # generated CSVs (committed)
  figures/          # generated PNGs (committed); failure images named fail_<NN>_<desc>.png
report/REPORT.md    # Vietnamese report template with 6 sections; every "[ĐIỀN...]" must be replaced
tools/check_submission.py  # must end with "KẾT QUẢ: SẴN SÀNG NỘP"
```

**Conventions (all tasks):**
- Run modules from the repo root as `python -m src.<module>`; imports look like `from starter.datasets import ...`
  and `from src.health_stats import ...`. `pytest.ini` sets `pythonpath = .`.
- Every CLI uses `argparse` with `--help`, `--data-root`, `--out-dir` (default `results`), and is deterministic
  (fixed `seed`, sorted frame lists). Running twice must produce byte-identical CSVs (round floats to 6 decimals).
- CSV names: lowercase snake_case. Dataset short tags: `synthetic`, `kitti`, `nusc` (helper `dataset_tag(root)` in
  `src/health_stats.py` maps a data root path to the tag by its last path component).
- **Never fabricate numbers or images.** Every number in REPORT.md must be copied from a CSV produced by our code;
  every image must be produced by our code.
- Tests must be fast (< ~60 s total) and must write only to pytest `tmp_path`, never into `results/`
  (except the final submission test in Task 10, which only reads).
- Keep comments sparse and in English; code style like `starter/` (type hints, `from __future__ import annotations`).
- Commit messages follow the lab convention `CPx: <short description>` (mapping given in each task).

TEST_CMD: python -m pytest -q src/tests

---

## Task 1: Project skeleton and test configuration

Goal: make `TEST_CMD` runnable.

Files to create/modify:
- `requirements.txt` — append `pytest>=7.4` (keep existing lines).
- `pytest.ini` (repo root):
  ```ini
  [pytest]
  pythonpath = .
  testpaths = src/tests
  addopts = -q -p no:cacheprovider
  ```
- `src/__init__.py` (empty), `src/tests/__init__.py` (empty). Delete `src/.gitkeep`.
- `src/tests/test_smoke.py` — imports `starter.datasets` and asserts
  `list_frames("data/synthetic") == ["000000","000001","000002","000003","000004"]`,
  `len(list_frames("data/kitti_mini")) == 20`, `len(list_frames("data/nuscenes_mini_subset")) == 80`.
- `results/figures/.gitkeep` (create the directory).

Acceptance criteria:
- `python -m pytest -q src/tests` passes (3 assertions in the smoke test).
- No file under `starter/` or `data/` changed.
- Commit: `CP0: project skeleton, pytest config`.

## Task 2: Implement the two TODO(CP2) projection functions

Only edit the bodies of `velo_to_cam` and `cam_to_image` in `starter/projection.py` (nothing else in that file).

- `velo_to_cam(points_xyz, calib)`: homogeneous (N,4) → `(calib.T_cam_velo @ hom.T).T[:, :3]`.
- `cam_to_image(points_cam, P2, image_shape, min_depth=0.1)`: `finite = np.isfinite(points_cam).all(1)`;
  `valid = finite & (z > min_depth)`; project only valid points with `P2` (3x4) on homogeneous coords; divide by the
  3rd component; then also require `0 <= u < W` and `0 <= v < H` with `(H, W) = image_shape[:2]`.
  Return `(uv[mask], depth[mask], mask)` where `mask` has length N. No warnings on NaN input
  (use `np.errstate` or avoid computing on invalid rows).

Files:
- `starter/projection.py` (2 function bodies only).
- `src/tests/test_projection.py`:
  - synthetic frame `000000` calib (`load_frame("data/synthetic","000000")["calib"]`): velodyne point `(10,0,0)` →
    `z_cam ≈ 9.73` (abs tol 0.05) and pixel `(u,v) ≈ (614,175)` (abs tol 2 px) with image shape `(375,1242,3)`.
  - a point behind the camera (`(-10,0,0)`) and a NaN row are masked out; mask length equals input length.
  - `project_velo_to_image` on real kitti_mini frame `000011` returns > 1000 valid points.

Acceptance criteria:
- Tests pass. `python -m starter.projection --data-root data/synthetic --frame 000000` writes a PNG to
  `results/figures/` without error (agent runs it once to verify; do not commit it in this task).
- `git diff starter/` shows changes only inside the two functions.
- Commit: `CP2: implement velo_to_cam and cam_to_image`.

## Task 3: Extended per-frame statistics (`src/health_stats.py`)

Pure functions, no I/O except loading points. Do not import from `starter.data_health` CLI; reimplementing is fine.

API:
- `dataset_tag(data_root) -> str` (`synthetic` / `kitti` / `nusc`, fallback = last path component).
- `frame_stats(points, n_az_bins=36, n_el_bins=16, el_range=(-30.0, 15.0)) -> dict` with keys:
  `n_points, n_invalid, invalid_ratio, range_min, range_p05, range_p50, range_p95, range_max,
  ratio_beyond_50m, ratio_within_2m, z_min, z_p50, z_max, intensity_mean, intensity_std, intensity_zero_ratio,
  intensity_sat_ratio (>=0.99), empty_az_bins, az_min_bin_ratio (min bin count / mean bin count),
  az_max_sector_gap_deg (longest run of consecutive empty azimuth bins, circular, in degrees),
  empty_el_bins, el_min_deg, el_max_deg, points_hash` (hash = `hashlib.md5(points.tobytes()).hexdigest()[:12]`,
  used for duplicate-frame detection). Range = 3D Euclidean norm of xyz. Stats computed on finite rows only;
  if zero finite rows, numeric stats are `nan` and `invalid_ratio = 1.0`.
- `azimuth_hist(points, n_bins=36)` and `elevation_hist(points, n_bins=16, el_range=(-30,15))` → `np.ndarray` counts
  (finite rows only; azimuth = `degrees(arctan2(y, x))` in [-180,180)).
- `dataset_stats(data_root) -> pandas.DataFrame` — one row per frame (`frame_id` first column), sorted.
- `raw_arrays(data_root, frame_id) -> dict` returning `range`, `intensity`, `az`, `el` 1-D arrays (finite rows) for
  histograms.

Files: `src/health_stats.py`, `src/tests/test_health_stats.py`.

Tests (handcrafted arrays, no dataset needed except one integration test):
- 4 points incl. 1 NaN row → `n_points == 4`, `invalid_ratio == 0.25`.
- Points only in azimuth [0°, 90°) → `empty_az_bins == 27`, `az_max_sector_gap_deg == 270`.
- Intensity all 0 → `intensity_zero_ratio == 1.0`.
- Two identical arrays → same `points_hash`; different arrays → different hash.
- All-NaN input does not raise and gives `invalid_ratio == 1.0`.
- `dataset_stats("data/synthetic")` has 5 rows and all keys above as columns.

Acceptance: tests pass. Commit: `CP2: extended per-frame health stats`.

## Task 4: Timestamps and time gaps (`src/health_time.py`)

API:
- `load_timestamps(data_root, frame_ids) -> pandas.Series | None` (index = frame_id, value = seconds, float):
  - KITTI-format root with `training/timestamps.txt`: one float per line, i-th line ↔ i-th sorted frame.
    If line count ≠ frame count, raise `ValueError` with a clear message.
  - nuScenes root: `starter.datasets.load_frame(root, fid)["timestamp_lidar_us"] / 1e6` (cache to avoid
    reloading; reading `v1.0-mini/sample_data.json` directly is also acceptable if faster and equivalent).
  - otherwise (kitti_mini has no timestamps) → `None`.
- `time_gaps(ts: pandas.Series) -> pandas.DataFrame` columns `frame_id, timestamp_s, dt_s, dt_ratio`
  where `dt_s` = difference to the previous frame (NaN for the first frame **and** for the first frame of each new
  nuScenes scene — scene = frame_id prefix before the last `_`) and `dt_ratio = dt_s / median(dt_s)`.
  Non-monotonic timestamps give `dt_s <= 0` (do not sort them away).

Files: `src/health_time.py`, `src/tests/test_health_time.py`.

Tests:
- synthetic: timestamps are `[0.0, 0.1, 0.2, 0.4, 0.5]`; `dt_s` for `000003` ≈ 0.2 and `dt_ratio` ≈ 2.0; median 0.1.
- handcrafted series with a decreasing timestamp yields a negative `dt_s` on that frame.
- kitti_mini → `None`.
- nuScenes: 80 timestamps; median `dt_s` within [0.4, 0.6]; `dt_s` is NaN for `scene-0103_000` and `scene-1094_000`.

Acceptance: tests pass. Commit: `CP2: timestamps and time-gap analysis`.

## Task 5: Alert rules (`src/health_rules.py`)

Rules are data-driven: each threshold is either absolute (sensor-physical) or **relative to the dataset median**
(robust across 64-beam KITTI and 32-beam nuScenes). Define them in one module-level list `RULES` of a small
dataclass `Rule(code, severity, description, threshold_text, check: Callable[[row, ctx], str | None])`, where the
check returns a human-readable reason with the actual value (e.g. `"n_points=12034 < 0.5*median(120400)"`) or `None`.
`ctx` holds dataset medians and the duplicate-hash map.

Required rules (codes are fixed; thresholds are module constants so the stress test can reuse them):
| code | severity | rule |
|---|---|---|
| `INVALID_POINTS` | error if > 1 %, warn if > 0 | `invalid_ratio` |
| `LOW_POINTS` | error | `n_points < 0.5 * median(n_points)` |
| `HIGH_POINTS` | warn | `n_points > 1.5 * median(n_points)` |
| `SECTOR_GAP` | error | `az_max_sector_gap_deg >= 30` **or** `empty_az_bins > median(empty_az_bins) + 2` (FOV-limited data like KITTI-cropped clouds must not all fire — the relative clause is the main one; document it) |
| `RANGE_ANOMALY` | warn | `range_p95 < 0.5*median` or `> 1.5*median`, or `range_max > 200` m |
| `NEAR_BLOCKAGE` | warn | `ratio_within_2m > max(0.05, 3*median)` |
| `INTENSITY_ANOMALY` | warn | `intensity_zero_ratio > 0.9`, or `intensity_sat_ratio > 0.2`, or `abs(intensity_mean - median)` > 3 × MAD-based sigma (MAD floor 0.01) |
| `Z_SHIFT` | warn | `abs(z_p50 - median(z_p50)) > 0.3` m (ground height / mounting / calibration drift) |
| `DUPLICATE_FRAME` | error | `points_hash` equal to an earlier frame's hash |
| `TIME_GAP` | warn | `dt_ratio > 1.5` (dropped frame) |
| `TIME_NONMONOTONIC` | error | `dt_s <= 0` |

API:
- `evaluate(stats_df, gaps_df | None) -> pandas.DataFrame` with columns
  `frame_id, status (OK|WARN|ERROR), n_flags, flags ("A;B"), reasons ("A: reason | B: reason"), action`
  where `action` = `use` (OK), `review/label` (WARN only), `drop` (any ERROR) — matches topic E goal
  "trusted / needs label-retrain / discard".
- `rules_table() -> pandas.DataFrame` (code, severity, description, threshold_text) for the report.

Files: `src/health_rules.py`, `src/tests/test_health_rules.py`.

Tests (build small stats DataFrames by hand, 5–6 rows):
- a row with half the median points → `LOW_POINTS`, status `ERROR`, action `drop`, reason contains the value.
- a duplicated hash → `DUPLICATE_FRAME` only on the second occurrence.
- `dt_ratio = 2.0` → `TIME_GAP`; `dt_s = -0.1` → `TIME_NONMONOTONIC`.
- a clean row → status `OK`, `flags == ""`, action `use`.
- `gaps_df=None` works (no time rules).
- **Integration on synthetic**: `evaluate(dataset_stats("data/synthetic"), time_gaps(load_timestamps(...)))`
  flags `TIME_GAP` on `000003` and flags at least 2 frames in total. Before writing this test, run the stats on
  synthetic, inspect the numbers, and write down every planted fault you can see; add one assertion per planted
  fault that the rules detect (frame id + code). If a planted fault is not caught by any rule above, add a rule
  (with a physical justification) rather than tuning thresholds to a single frame.
- **No-false-alarm sanity**: on `data/kitti_mini`, fewer than 25 % of frames have status `ERROR`.

Acceptance: tests pass. Commit: `CP3: alert rules with thresholds and reasons`.

## Task 6: Dashboard CLI (`src/health_dashboard.py`)

`python -m src.health_dashboard --data-root data/synthetic [--out-dir results]` does:
1. `stats = dataset_stats(root)`, `gaps = time_gaps(ts)` if timestamps exist, `flags = evaluate(stats, gaps)`.
2. Writes `<out>/health_stats_<tag>.csv` (stats merged with gap columns), `<out>/health_flags_<tag>.csv`
   (frame → status, flags, reasons, action), and once `<out>/health_rules.csv` (`rules_table()`).
3. Writes `<out>/figures/dashboard_<tag>.png`: one matplotlib figure, 3×3 grid, ~16×12 in, dpi 110, title with
   dataset name and frame/flag counts:
   (a) range histogram (all frames pooled, flagged frames overlaid in a contrasting colour),
   (b) intensity histogram, (c) points-per-frame bar chart coloured by status (OK/WARN/ERROR legend),
   (d) invalid ratio per frame bar chart, (e) azimuth density heatmap (frames × 36 bins, log counts),
   (f) elevation density heatmap (frames × 16 bins), (g) time gap per frame (`dt_s` with median line; if no
   timestamps, show text "no timestamps in this dataset"), (h) z_p50 per frame, (i) table/text panel listing
   flagged frames with codes. For > 30 frames, tick labels show every k-th frame id.
4. Prints one compact line per flagged frame and the output paths.

Files: `src/health_dashboard.py`, `src/tests/test_health_dashboard.py`.

Tests (use `tmp_path` as `--out-dir`; call `main([...])` — make `main(argv=None)` accept an argv list):
- synthetic run creates the 3 CSVs and `figures/dashboard_synthetic.png` (size > 20 KB); flags CSV has 5 rows and
  columns `frame_id,status,n_flags,flags,reasons,action`.
- Running twice gives byte-identical CSVs (determinism).
- `--help` exits with code 0.

Acceptance: tests pass. Commit: `CP2: data health dashboard (matplotlib + CSV)`.

## Task 7: Generate dashboard results on the 3 datasets + planted-fault table

Run and commit real outputs (this is the "demo chạy thật" evidence):
```bash
python -m src.health_dashboard --data-root data/synthetic
python -m src.health_dashboard --data-root data/kitti_mini
python -m src.health_dashboard --data-root data/nuscenes_mini_subset
python -m starter.projection --data-root data/kitti_mini --frame 000011
python -m starter.projection --data-root data/nuscenes_mini_subset --frame scene-0103_010
```
Add `src/compare_datasets.py` (CLI, `--out-dir`) that reads the produced `health_stats_*.csv` / `health_flags_*.csv`
and writes:
- `results/dataset_comparison.csv`: one row per dataset/group (`synthetic`, `kitti`, `nusc_day` = scene-0103,
  `nusc_night` = scene-1094) with median `n_points`, `range_p95`, `intensity_mean`, `empty_az_bins`, `el_max_deg -
  el_min_deg`, `median dt_s`, and % frames OK/WARN/ERROR.
- `results/synthetic_planted_faults.csv` with columns `fault, frame_id, detected_by_rule, evidence_value` built
  from `health_flags_synthetic.csv` (one row per (frame, flag)), for RUBRIC bonus B6.

Files: `src/compare_datasets.py`, `src/tests/test_compare_datasets.py` (feeds tiny handmade CSVs from `tmp_path`;
checks the 2 output files and the night/day split), and the generated `results/*.csv`,
`results/figures/dashboard_*.png`, `results/figures/overlay_*.png`.

Acceptance:
- Tests pass; the listed files exist in `results/` and are committed.
- Overlay images visually show LiDAR points on objects, none in the sky (agent opens the PNG to check).
- Commit: `CP3: dashboard results on synthetic, kitti_mini, nuscenes`.

## Task 8: Stress test / benchmark with ≥ 3 levels (`src/stress_test.py`)

Question answered: "How strong must a degradation be before the dashboard flags it?" (rule sensitivity), plus
dashboard cost. Controlled experiment: frames = all 20 `kitti_mini` frames, seed = 0, rules/thresholds unchanged,
**only one factor varies per sweep**. The reference medians for relative rules come from the **clean** dataset
(apply degradation to one frame at a time, evaluate it against clean medians, i.e. pass the clean stats plus the
degraded row into `evaluate`, and read the degraded row's flags).

Sweeps (each ≥ 3 levels, level 0 = clean baseline):
- `random_dropout` keep_ratio ∈ {1.0, 0.9, 0.7, 0.5, 0.3} → expect `LOW_POINTS`.
- `sector_dropout` width ∈ {0, 15, 30, 60, 90}° centred on the front (az 0) → expect `SECTOR_GAP`.
- `gaussian_noise` sigma_xyz ∈ {0, 0.02, 0.05, 0.1, 0.3} m → record which (if any) rule fires.
- `z offset` (add dz to z) ∈ {0, 0.1, 0.2, 0.4, 0.8} m (simulated mounting/calibration drift) → expect `Z_SHIFT`.

Outputs:
- `results/stress_detection.csv`: `perturbation, level, n_frames, detection_rate (expected rule), any_flag_rate,
  mean_n_points, mean_metric` (the stat the expected rule watches), plus a `config` column (e.g.
  `seed=0;frames=kitti_mini(20)`).
- `results/stress_latency.csv`: per-frame `frame_stats` latency on kitti_mini vs nuScenes: drop first run, ≥ 20
  repeats, `p50_ms, p95_ms`, plus `platform.processor()` / `platform.platform()` strings (bonus B3).
- `results/figures/stress_detection.png`: 2×2 subplots, detection rate vs level per perturbation, threshold
  marked.

Files: `src/stress_test.py` (CLI with `--data-root`, `--out-dir`, `--repeats`), `src/tests/test_stress_test.py`
(run on `data/synthetic` with `--repeats 3` into `tmp_path`; asserts each perturbation has ≥ 3 levels,
`detection_rate` at level 0 is 0 for `LOW_POINTS`, is 1.0 at keep_ratio 0.3, and is monotonic non-decreasing for
random_dropout; two runs give identical `stress_detection.csv`).

Acceptance: tests pass; real run on kitti_mini produces the 3 artifacts and they are committed.
Commit: `CP3: stress test detection sweep + latency`.

## Task 9: Failure case figure (`src/failure_case.py`)

Find at least one case where the dashboard is **wrong**, from real runs, and render it. Default candidate (verify
with numbers first; if the data shows a different, clearer failure — e.g. a degradation from Task 8 that no rule
catches, such as gaussian_noise 0.1 m — use that instead and name the file accordingly):

**fail_01 — absolute thresholds tuned on KITTI misfire on nuScenes (Metric / Preprocess class).** Compute
`LOW_POINTS` with an absolute threshold = 0.5 × KITTI median `n_points` and apply it to nuScenes: every 32-beam
nuScenes frame is falsely flagged although it is healthy, while a real 50 % random dropout on a nuScenes frame is
indistinguishable from the healthy ones. The relative (per-dataset median) rule from Task 5 avoids this. Also
show azimuth density for nuScenes vs KITTI, noting that nuScenes azimuth 0° = right side (Geometry convention),
so a "front sector" rule written in KITTI azimuth would watch the wrong sector.

Outputs:
- `results/figures/fail_01_kitti_threshold_on_nuscenes.png`: 1×3 panels — (1) points-per-frame for KITTI vs
  nuScenes with the absolute threshold line, (2) same frames with the relative threshold, (3) azimuth histograms
  KITTI vs nuScenes with the "front" direction annotated for each.
- `results/failure_case.csv`: `rule_variant, dataset, n_frames, n_flagged, flagged_ratio` for absolute vs relative
  on kitti/nusc clean and nusc with 50 % dropout.

Files: `src/failure_case.py`, `src/tests/test_failure_case.py` (runs the core function on small synthetic arrays
or the synthetic dataset into `tmp_path`; asserts PNG name starts with `fail_` and CSV has the columns above).

Acceptance: tests pass; real run produces both files, committed; numbers in the CSV support the stated failure
(if they do not, choose another failure as described above — never adjust the narrative to non-matching numbers).
Commit: `CP4: failure analysis figure`.

## Task 10: Fill report/REPORT.md and pass check_submission

Fill `report/REPORT.md` in **Vietnamese**, keeping the exact headings `## 1. Claim` … `## 6. Khai báo sử dụng AI`,
3–8 lines per section, and remove every `[ĐIỀN...]`:
- Header: Họ tên `Đỗ Đình Long`; MSSV `2A202602673`; Lớp `VinUni AI20K – Track 4`; Link repo
  `https://github.com/long2110d/DoDinhLong-2A202602673-Track4-Day21`; Topic `E — Data health dashboard`;
  Dataset: all three; frames: synthetic 000000–000004, kitti_mini 20 frames, nuScenes scene-0103 + scene-1094.
- §1 Claim: one measurable claim with quantity, condition, threshold — taken from Task 8 numbers (e.g. "rule
  LOW_POINTS (ngưỡng 0.5×median) phát hiện 100 % frame KITTI khi giữ ≤ X % điểm, 0 % báo động giả trên dữ liệu
  sạch"). Fill X from `results/stress_detection.csv`.
- §2 Evidence: table of planted faults (from `synthetic_planted_faults.csv`), summary of flags per dataset and the
  KITTI vs nuScenes day/night comparison (from `dataset_comparison.csv`, explain beam count / night intensity),
  stress-test table, images `../results/figures/dashboard_synthetic.png`, `dashboard_kitti.png`,
  `stress_detection.png`, and the rules table (`health_rules.csv`).
- §3 Failure case: image `../results/figures/fail_01_...png`, when/why it fails, debug class (Metric, plus
  Geometry for the azimuth convention), how to detect in production (per-sensor baselines).
- §4 Recommendations: ADAS/robot use case, trade-offs (relative thresholds need a warm-up baseline; latency p50/p95
  from `stress_latency.csv`), metrics to log online (points/frame, sector gap, invalid ratio, dt between sweeps,
  z_p50).
- §5 Reproduction commands: venv + `pip install -r requirements.txt`, every command from Tasks 7–9 in order,
  `python -m pytest -q src/tests`, `python tools/check_submission.py`.
- §6 AI declaration table: Claude Opus (planning, PLAN.md), Codex (writing code per task), Claude Sonnet (code
  review of each diff + test results); verification: unit tests with handcrafted arrays, CP2 manual check point
  `(10,0,0)` → `z≈9.73, (u,v)≈(614,175)`, rerunning scripts to get identical CSVs, visually inspecting every
  figure, cross-checking report numbers against CSVs.

Every number in the report must appear in a committed CSV under `results/`. Also add
`src/tests/test_submission.py`: runs `python tools/check_submission.py` via `subprocess` (`sys.executable`, cwd =
repo root) and asserts exit code 0 and `"SẴN SÀNG NỘP"` in stdout (decode as UTF-8); also asserts every
`../results/...` image path referenced in REPORT.md exists.

Acceptance:
- `python tools/check_submission.py` prints only `[PASS]` lines and ends with `KẾT QUẢ: SẴN SÀNG NỘP`.
- `python -m pytest -q src/tests` passes (full suite).
- Commit: `CP5: complete report, submission check passes`.
