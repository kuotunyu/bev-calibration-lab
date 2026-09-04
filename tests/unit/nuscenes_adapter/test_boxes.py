"""Contracts for bringing 3D boxes from global coordinates into the camera.

nuScenes stores box centres and orientations in global, not in any sensor frame.
Treating a global centre as if it were already ego-relative is a mistake worth
hundreds of metres, and it does not raise anything: the boxes simply land
somewhere plausible and wrong.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from bevcalib.geometry.frames import FramedTransform, inverse_framed
from bevcalib.geometry.quaternions import normalize_quaternion_wxyz, quaternion_to_matrix
from bevcalib.geometry.se3 import SE3, transform_points

QUARTER_TURN_Z = (math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5))
TILTED = normalize_quaternion_wxyz((0.8, 0.2, 0.3, 0.4))


def camera_packet(
    ego_translation: tuple[float, float, float] = (100.0, 200.0, 0.0),
    sensor_rotation: tuple[float, float, float, float] = QUARTER_TURN_Z,
):  # type: ignore[no-untyped-def]
    from bevcalib.nuscenes_adapter.frames import SensorPacket

    return SensorPacket(
        sample_token="sample-0",
        sample_data_token="camera-data",
        timestamp_us=1_538_984_233_597_259,
        calibrated_sensor=FramedTransform(
            target="camera_ego",
            source="camera_sensor",
            value=SE3(rotation_wxyz=sensor_rotation, translation_xyz_m=(1.7, 0.0, 1.5)),
        ),
        ego_pose=FramedTransform(
            target="global",
            source="camera_ego",
            value=SE3(rotation_wxyz=(1.0, 0.0, 0.0, 0.0), translation_xyz_m=ego_translation),
        ),
        file_relative_path="samples/CAM_FRONT/scene.jpg",
    )


def expected_transform(camera):  # type: ignore[no-untyped-def]
    """The camera-from-global transform, built independently of the code under test."""

    from bevcalib.geometry.frames import compose_framed

    return compose_framed(inverse_framed(camera.calibrated_sensor), inverse_framed(camera.ego_pose))


def test_a_box_centre_is_carried_from_global_into_the_camera() -> None:
    """The centre must move by the whole camera-from-global chain, not part of it."""

    from bevcalib.nuscenes_adapter.boxes import transform_global_box_to_camera

    camera = camera_packet()
    centre_global = np.array([110.0, 205.0, 1.0])

    centre_camera, _ = transform_global_box_to_camera(centre_global, TILTED, camera)

    expected = transform_points(expected_transform(camera).value, centre_global[None, :])[0]
    np.testing.assert_allclose(centre_camera, expected, atol=1e-9)


def test_a_box_at_the_global_origin_lands_where_the_chain_puts_it() -> None:
    """The origin is the case where a forgotten ego translation is most visible."""

    from bevcalib.nuscenes_adapter.boxes import transform_global_box_to_camera

    camera = camera_packet(ego_translation=(100.0, 200.0, 0.0))

    centre_camera, _ = transform_global_box_to_camera(np.zeros(3), (1.0, 0.0, 0.0, 0.0), camera)

    # Far from the origin, because the vehicle is 100 m east and 200 m north of it.
    assert float(np.linalg.norm(centre_camera)) > 100.0
    expected = transform_points(expected_transform(camera).value, np.zeros((1, 3)))[0]
    np.testing.assert_allclose(centre_camera, expected, atol=1e-9)


def test_the_orientation_is_rotated_by_the_same_chain_as_the_centre() -> None:
    """A box whose centre moves but whose heading does not is a box facing the wrong way."""

    from bevcalib.nuscenes_adapter.boxes import transform_global_box_to_camera

    camera = camera_packet()

    _, orientation_camera = transform_global_box_to_camera(np.zeros(3), TILTED, camera)

    expected_matrix = quaternion_to_matrix(
        expected_transform(camera).value.rotation_wxyz
    ) @ quaternion_to_matrix(TILTED)
    np.testing.assert_allclose(quaternion_to_matrix(orientation_camera), expected_matrix, atol=1e-9)


def test_an_unrotated_camera_leaves_a_box_orientation_alone() -> None:
    """A case with an answer that can be read rather than computed."""

    from bevcalib.nuscenes_adapter.boxes import transform_global_box_to_camera

    camera = camera_packet(ego_translation=(0.0, 0.0, 0.0), sensor_rotation=(1.0, 0.0, 0.0, 0.0))

    _, orientation_camera = transform_global_box_to_camera(np.zeros(3), TILTED, camera)

    assert orientation_camera == pytest.approx(TILTED, abs=1e-9)


def test_the_returned_orientation_is_a_canonical_unit_quaternion() -> None:
    """Two records of one box orientation must compare equal, which needs a fixed sign."""

    from bevcalib.nuscenes_adapter.boxes import transform_global_box_to_camera

    _, orientation = transform_global_box_to_camera(np.zeros(3), TILTED, camera_packet())

    assert orientation == pytest.approx(normalize_quaternion_wxyz(orientation), abs=1e-15)
    assert float(np.linalg.norm(orientation)) == pytest.approx(1.0, abs=1e-12)


@pytest.mark.parametrize("shape", [(2,), (4,), (1, 3), (3, 1)])
def test_a_centre_that_is_not_three_numbers_is_rejected(shape: tuple[int, ...]) -> None:
    """A `[1, 3]` centre would broadcast and return an array nobody expected."""

    from bevcalib.nuscenes_adapter.boxes import transform_global_box_to_camera

    with pytest.raises(ValueError, match=r"^a box centre must be three numbers, got shape "):
        transform_global_box_to_camera(np.zeros(shape), TILTED, camera_packet())


def test_a_packet_that_is_not_a_camera_is_refused() -> None:
    """Boxes are projected into a camera; a LiDAR packet here is a wiring mistake."""

    from bevcalib.nuscenes_adapter.boxes import transform_global_box_to_camera

    lidar_shaped = FramedTransform(
        target="lidar_ego",
        source="lidar_sensor",
        value=SE3(rotation_wxyz=(1.0, 0.0, 0.0, 0.0), translation_xyz_m=(0.9, 0.0, 1.8)),
    )
    camera = camera_packet()
    wrong = type(camera)(
        sample_token=camera.sample_token,
        sample_data_token=camera.sample_data_token,
        timestamp_us=camera.timestamp_us,
        calibrated_sensor=lidar_shaped,
        ego_pose=FramedTransform(target="global", source="lidar_ego", value=camera.ego_pose.value),
        file_relative_path=camera.file_relative_path,
    )

    with pytest.raises(
        ValueError,
        match=r"^boxes are projected into a camera, but this packet's sensor frame is .*, not 'camera_sensor'$",
    ):
        transform_global_box_to_camera(np.zeros(3), TILTED, wrong)
