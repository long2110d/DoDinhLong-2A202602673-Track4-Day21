# Progress Log

## Task 1: Project skeleton and test configuration
- Added pytest>=7.4 while preserving existing requirements.
- Configured pytest imports, test discovery, and disabled its cache provider.
- Created src and test package files, removed src/.gitkeep, and added results/figures/.gitkeep.
- Added a smoke test verifying synthetic frame IDs and KITTI/nuScenes frame counts.
- Ran python -m pytest -q src/tests successfully; starter/ and data/ remain unchanged.

## Task 2: Implement the two TODO(CP2) projection functions
﻿- Implemented homogeneous LiDAR-to-camera transformation in velo_to_cam.
- Implemented camera projection with finite-value, minimum-depth, and image-boundary filtering while preserving the input mask length.
- Added synthetic calibration, invalid-input, boundary, and real KITTI projection regression tests.
- Ran python -m pytest -q src/tests successfully: all 5 tests passed.
- Verified the synthetic projection CLI generated a PNG with 3,910 valid points, then removed the verification image from the task changes.

## Task 3: Extended per-frame statistics (`src/health_stats.py`)
- Implemented Task 3 in src/health_stats.py with finite-row statistics, 3D ranges, intensity metrics, angular coverage, and point hashes.
- Added dataset tags, sorted per-frame DataFrames rounded to six decimals, and raw arrays for histograms.
- Added tests for invalid and empty clouds, circular azimuth gaps, histogram boundaries, hashes, and the synthetic dataset.
- Ran python -m pytest -q src/tests using the repository virtual environment; all 16 tests passed.

## Task 4: Timestamps and time gaps (`src/health_time.py`)
- Implemented timestamp loading for KITTI-format and nuScenes datasets, with cached nuScenes timestamps.
- Added clear timestamp-count validation and support for datasets without timestamps.
- Implemented time gaps and median gap ratios while preserving non-monotonic timestamps and resetting at scene boundaries.
- Added tests for synthetic faults, missing timestamps, count mismatches, nuScenes scenes, caching, and empty inputs.
- Validation: python -m pytest -q src/tests passed all 24 tests using the virtual environment.

## Task 5: Alert rules (`src/health_rules.py`)
- Implemented Task 5 health rules with reusable threshold constants, dataset medians, measured reasons, and triage actions.
- Added duplicate-frame and timestamp alerts, robust intensity checks, and FOV-aware sector-gap detection.
- Inspected synthetic statistics and added a physically motivated sparse-sector warning for partial angular dropout.
- Added rule, synthetic-fault, and KITTI false-error sanity tests; all 44 tests pass with python -m pytest -q src/tests.

## Task 6: Dashboard CLI (`src/health_dashboard.py`)
﻿- Implemented the Task 6 dashboard CLI with deterministic statistics, flags, and rules CSV exports.
- Added a nine-panel Matplotlib dashboard with status colors, flagged-frame overlays, density heatmaps, and timestamp handling.
- Added tests for synthetic outputs, CSV determinism, missing timestamps, and CLI help.
- Validated with python -m pytest -q src/tests using the project virtual environment: all 47 tests passed.

## Task 7: Generate dashboard results on the 3 datasets + planted-fault table
- Implemented the dataset comparison CLI with median health metrics, timing, status percentages, and separate nuScenes day/night groups.
- Exported synthetic fault rows directly from dashboard flags and their measured rule evidence.
- Generated real dashboard CSVs and figures for synthetic, KITTI, and nuScenes, plus both requested projection overlays.
- Visually checked both overlays for points on scene objects and no visible points in the sky.
- Added handmade CSV tests covering aggregation, day/night splitting, fault evidence, missing timestamps, and deterministic exports; python -m pytest -q src/tests passed.

## Task 8: Stress test / benchmark with ≥ 3 levels (`src/stress_test.py`)
- Implemented Task 8 with four deterministic, five-level degradation sweeps and rule detection metrics.
- Added optional clean reference statistics to freeze benchmark medians without changing existing rules or thresholds.
- Generated KITTI detection CSV, KITTI/nuScenes latency CSV with 20 repeats per frame, and a four-panel detection figure.
- Added synthetic CLI, reproducibility, latency, and frozen-reference tests; all 51 tests pass with python -m pytest -q src/tests.
