# Progress Log

## Task 1: Project skeleton and test configuration
- Added pytest>=7.4 while preserving existing requirements.
- Configured pytest imports, test discovery, and disabled its cache provider.
- Created src and test package files, removed src/.gitkeep, and added results/figures/.gitkeep.
- Added a smoke test verifying synthetic frame IDs and KITTI/nuScenes frame counts.
- Ran python -m pytest -q src/tests successfully; starter/ and data/ remain unchanged.
