from __future__ import annotations

import starter.datasets


def test_dataset_frame_lists() -> None:
    assert starter.datasets.list_frames("data/synthetic") == [
        "000000", "000001", "000002", "000003", "000004"
    ]
    assert len(starter.datasets.list_frames("data/kitti_mini")) == 20
    assert len(starter.datasets.list_frames("data/nuscenes_mini_subset")) == 80
