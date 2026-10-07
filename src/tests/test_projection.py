from __future__ import annotations

import warnings

import numpy as np
import pytest

from starter.datasets import load_frame
from starter.projection import cam_to_image, project_velo_to_image, velo_to_cam


def test_synthetic_projection():
    calib = load_frame("data/synthetic", "000000")["calib"]
    points_cam = velo_to_cam(np.array([[10.0, 0.0, 0.0]]), calib)
    assert points_cam[0, 2] == pytest.approx(9.73, abs=0.05)
    uv, depth, mask = cam_to_image(points_cam, calib.P2, (375, 1242, 3))
    np.testing.assert_allclose(uv[0], [614, 175], atol=2)
    np.testing.assert_array_equal(mask, [True])
    np.testing.assert_array_equal(depth, points_cam[:, 2])


def test_invalid_points_are_masked_without_warnings():
    calib = load_frame("data/synthetic", "000000")["calib"]
    points = np.array([[10.0, 0.0, 0.0], [-10.0, 0.0, 0.0]])
    points_cam = np.vstack((velo_to_cam(points, calib), [np.nan, 0, 1]))
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        uv, depth, mask = cam_to_image(points_cam, calib.P2, (375, 1242, 3))
    np.testing.assert_array_equal(mask, [True, False, False])
    assert mask.shape == (len(points_cam),)
    assert uv.shape == (1, 2)
    assert depth.shape == (1,)


def test_depth_and_image_boundaries():
    points = np.array([[0, 0, 1], [9, 7, 1], [10, 0, 1], [0, 8, 1],
                       [-1, 0, 1], [0, -1, 1], [0, 0, 0.1], [np.inf, 0, 1]])
    uv, depth, mask = cam_to_image(points, np.eye(3, 4), (8, 10))
    np.testing.assert_array_equal(mask, [True, True, False, False, False, False, False, False])
    np.testing.assert_array_equal(uv, [[0, 0], [9, 7]])
    np.testing.assert_array_equal(depth, [1, 1])


def test_real_kitti_projection():
    frame = load_frame("data/kitti_mini", "000011")
    uv, depth, mask = project_velo_to_image(frame["points"], frame["calib"], frame["image"].shape)
    assert len(uv) > 1000
    assert len(depth) == mask.sum() == len(uv)
    assert mask.shape == (len(frame["points"]),)
