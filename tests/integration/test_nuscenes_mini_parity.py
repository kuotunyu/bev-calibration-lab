"""Parity against the official devkit on real nuScenes mini samples.

This is the only test in the repository that reads the dataset, and it is the one
that can tell us the adapter agrees with the reference implementation rather than
merely agreeing with its own fixtures. It is skipped unless `NUSCENES_ROOT` points
at a mini installation, and it is marked `slow` so an ordinary run never waits on
it.

Nothing else depends on it. Every rule the adapter implements is covered by unit
tests on synthetic records, so a skip here reduces confidence in one specific
thing, agreement with the devkit, and weakens no coverage.

Set NUSCENES_ROOT to a v1.0-mini installation to run it. Its absence is a skip,
not evidence that the dataset is uninstalled.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.slow

TOLERANCE = 1e-6
# Fixed tokens from the nuScenes mini split, so the comparison is reproducible.
SAMPLE_TOKENS = (
    "ca9a282c9e77460f8360f564131a8af5",
    "39586f9d59004284a7114a68825e8eec",
)


def nuscenes_root() -> Path:
    root = os.environ.get("NUSCENES_ROOT")
    if root is None:
        pytest.skip(
            "NUSCENES_ROOT is not set. Set NUSCENES_ROOT to a v1.0-mini installation to run "
            "devkit parity. Every adapter "
            "rule is covered by unit tests on synthetic records; this test adds agreement with "
            "the official devkit, which cannot be checked without the data."
        )
    return Path(root)


@pytest.fixture(scope="module")
def devkit():  # type: ignore[no-untyped-def]
    root = nuscenes_root()
    nuscenes = pytest.importorskip("nuscenes.nuscenes", reason="nuscenes-devkit is not installed")
    return nuscenes.NuScenes(version="v1.0-mini", dataroot=str(root), verbose=False)


@pytest.mark.parametrize("sample_token", SAMPLE_TOKENS)
def test_transformed_lidar_points_match_the_devkit(devkit, sample_token: str) -> None:  # type: ignore[no-untyped-def]
    """The whole chain and native projection agree point for point."""

    from nuscenes.utils.geometry_utils import view_points

    from bevcalib.geometry.projection import project_camera
    from bevcalib.geometry.se3 import transform_points
    from bevcalib.nuscenes_adapter.frames import lidar_to_camera_chain
    from bevcalib.nuscenes_adapter.samples import read_lidar_points, read_sensor_packet

    def lookup(table: str, token: str) -> dict:
        return devkit.get(table, token)

    sample = devkit.get("sample", sample_token)
    lidar = read_sensor_packet(
        lookup, sample["data"]["LIDAR_TOP"], ego_frame="lidar_ego", sensor_frame="lidar_sensor"
    )
    camera = read_sensor_packet(
        lookup, sample["data"]["CAM_FRONT"], ego_frame="camera_ego", sensor_frame="camera_sensor"
    )

    points = read_lidar_points(nuscenes_root() / lidar.file_relative_path)[:, :3]
    assert points.shape[0] > 0
    ours = transform_points(lidar_to_camera_chain(lidar, camera).value, points)
    theirs = devkit_chain_points(devkit, lidar, camera, points)
    np.testing.assert_allclose(ours, theirs, atol=TOLERANCE, rtol=0)

    camera_data = devkit.get("sample_data", camera.sample_data_token)
    calibrated = devkit.get("calibrated_sensor", camera_data["calibrated_sensor_token"])
    intrinsic = np.asarray(calibrated["camera_intrinsic"], dtype=np.float64)
    image_size = (int(camera_data["width"]), int(camera_data["height"]))
    projected = project_camera(ours, intrinsic, image_size)
    finite = np.all(np.isfinite(theirs), axis=1)
    projectable = finite & (theirs[:, 2] != 0.0)
    assert bool(np.any(projectable))
    reference_uv = np.full_like(projected.uv, np.nan)
    reference_uv[projectable] = view_points(theirs[projectable].T, intrinsic, normalize=True)[:2].T
    reference_depth = theirs[:, 2]
    reference_in_front = finite & (reference_depth > 0.0)
    width, height = image_size
    reference_in_image = (
        (reference_uv[:, 0] >= 0.0)
        & (reference_uv[:, 0] < width)
        & (reference_uv[:, 1] >= 0.0)
        & (reference_uv[:, 1] < height)
    )
    reference_valid = reference_in_front & reference_in_image

    np.testing.assert_array_equal(projected.in_front, reference_in_front)
    np.testing.assert_array_equal(projected.in_image, reference_in_image)
    np.testing.assert_array_equal(projected.valid, reference_valid)
    assert bool(np.any(reference_valid))
    np.testing.assert_allclose(
        projected.uv[reference_valid], reference_uv[reference_valid], atol=TOLERANCE, rtol=0
    )
    np.testing.assert_allclose(
        projected.optical_depth,
        reference_depth,
        atol=TOLERANCE,
        rtol=0,
    )

    point_max_abs_error = float(np.max(np.abs(ours - theirs)))
    uv_max_abs_error = float(
        np.max(np.abs(projected.uv[reference_valid] - reference_uv[reference_valid]))
    )
    depth_max_abs_error = float(np.max(np.abs(projected.optical_depth - reference_depth)))
    max_abs_error = max(point_max_abs_error, uv_max_abs_error, depth_max_abs_error)
    print(
        "H_PARITY "
        f"case=lidar_projection sample_index={SAMPLE_TOKENS.index(sample_token)} "
        f"comparison_count={points.shape[0]} valid_uv_count={int(np.sum(reference_valid))} "
        f"point_max_abs_error={point_max_abs_error:.17g} "
        f"uv_max_abs_error={uv_max_abs_error:.17g} "
        f"depth_max_abs_error={depth_max_abs_error:.17g} "
        f"max_abs_error={max_abs_error:.17g}"
    )


def devkit_chain_points(devkit, lidar, camera, points: np.ndarray) -> np.ndarray:  # type: ignore[no-untyped-def]
    """Apply the same four links using the devkit's own quaternion and pose helpers."""

    from pyquaternion import Quaternion

    def pose(record: dict) -> tuple[np.ndarray, np.ndarray]:
        return Quaternion(record["rotation"]).rotation_matrix, np.array(record["translation"])

    lidar_data = devkit.get("sample_data", lidar.sample_data_token)
    camera_data = devkit.get("sample_data", camera.sample_data_token)
    sensor_rotation, sensor_translation = pose(
        devkit.get("calibrated_sensor", lidar_data["calibrated_sensor_token"])
    )
    ego_rotation, ego_translation = pose(devkit.get("ego_pose", lidar_data["ego_pose_token"]))
    camera_ego_rotation, camera_ego_translation = pose(
        devkit.get("ego_pose", camera_data["ego_pose_token"])
    )
    camera_rotation, camera_translation = pose(
        devkit.get("calibrated_sensor", camera_data["calibrated_sensor_token"])
    )

    moved = points @ sensor_rotation.T + sensor_translation
    moved = moved @ ego_rotation.T + ego_translation
    moved = (moved - camera_ego_translation) @ camera_ego_rotation
    return (moved - camera_translation) @ camera_rotation


@pytest.mark.parametrize("sample_token", SAMPLE_TOKENS)
def test_box_centres_match_the_devkit(devkit, sample_token: str) -> None:  # type: ignore[no-untyped-def]
    """Boxes are natively global, so this checks the origin convention as well as the chain."""

    from bevcalib.nuscenes_adapter.boxes import transform_global_box_to_camera
    from bevcalib.nuscenes_adapter.samples import read_sensor_packet

    def lookup(table: str, token: str) -> dict:
        return devkit.get(table, token)

    sample = devkit.get("sample", sample_token)
    camera = read_sensor_packet(
        lookup, sample["data"]["CAM_FRONT"], ego_frame="camera_ego", sensor_frame="camera_sensor"
    )
    _, reference_boxes, _ = devkit.get_sample_data(sample["data"]["CAM_FRONT"])
    assert reference_boxes

    errors = []
    for reference in reference_boxes:
        global_box = devkit.get_box(reference.token)
        ours, _ = transform_global_box_to_camera(
            np.asarray(global_box.center, dtype=np.float64),
            tuple(float(value) for value in global_box.orientation.elements),
            camera,
        )
        expected = np.asarray(reference.center)
        np.testing.assert_allclose(ours, expected, atol=TOLERANCE, rtol=0)
        errors.append(float(np.max(np.abs(ours - expected))))
    print(
        "H_PARITY "
        f"case=box_centers sample_index={SAMPLE_TOKENS.index(sample_token)} "
        f"comparison_count={len(reference_boxes)} max_abs_error={max(errors):.17g}"
    )
