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
