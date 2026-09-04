"""Contracts for turning nuScenes records into frame-safe transforms.

Everything here runs on synthetic records shaped like the devkit's tables. No
nuScenes file is read: the dataset is gated until P1 is released, and more
importantly, logic that only works when the real data is present is logic that
nobody can test.

The chain being built is LiDAR sensor to LiDAR-time ego to global to camera-time
ego to camera sensor. Two ego poses, from two timestamps.
"""

from __future__ import annotations

import math
import struct
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from bevcalib.geometry.quaternions import normalize_quaternion_wxyz, quaternion_to_matrix
from bevcalib.geometry.se3 import SE3, transform_points

# All four components nonzero, so a yaw-only shortcut cannot reproduce it.
TILTED = normalize_quaternion_wxyz((0.8, 0.2, 0.3, 0.4))
IDENTITY = (1.0, 0.0, 0.0, 0.0)
QUARTER_TURN_Z = (math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5))


def records(
    *,
    lidar_sensor_rotation: tuple[float, float, float, float] = IDENTITY,
    lidar_ego_translation: tuple[float, float, float] = (0.0, 0.0, 0.0),
    camera_ego_translation: tuple[float, float, float] = (0.7, 0.0, 0.0),
) -> dict[str, dict[str, dict[str, Any]]]:
    """Build the four tables the adapter reads, with distinguishable values."""

    return {
        "sample_data": {
            "lidar-data": {
                "token": "lidar-data",
                "sample_token": "sample-0",
                "timestamp": 1_538_984_233_547_259,
                "filename": "samples/LIDAR_TOP/scene.pcd.bin",
                "calibrated_sensor_token": "cs-lidar",
                "ego_pose_token": "ego-lidar",
            },
            "camera-data": {
                "token": "camera-data",
                "sample_token": "sample-0",
                "timestamp": 1_538_984_233_597_259,
                "filename": "samples/CAM_FRONT/scene.jpg",
                "calibrated_sensor_token": "cs-camera",
                "ego_pose_token": "ego-camera",
            },
        },
        "calibrated_sensor": {
            "cs-lidar": {
                "translation": [0.9, 0.0, 1.8],
                "rotation": list(lidar_sensor_rotation),
            },
            "cs-camera": {"translation": [1.7, 0.0, 1.5], "rotation": list(QUARTER_TURN_Z)},
        },
        "ego_pose": {
            "ego-lidar": {
                "translation": list(lidar_ego_translation),
                "rotation": list(IDENTITY),
            },
            "ego-camera": {
                "translation": list(camera_ego_translation),
                "rotation": list(IDENTITY),
            },
        },
    }


def lookup_for(tables: dict[str, dict[str, dict[str, Any]]]):  # type: ignore[no-untyped-def]
    def lookup(table: str, token: str) -> dict[str, Any]:
        return tables[table][token]

    return lookup


def packets(**kwargs: Any):  # type: ignore[no-untyped-def]
    from bevcalib.nuscenes_adapter.samples import read_sensor_packet

    lookup = lookup_for(records(**kwargs))
    lidar = read_sensor_packet(
        lookup, "lidar-data", ego_frame="lidar_ego", sensor_frame="lidar_sensor"
    )
    camera = read_sensor_packet(
        lookup, "camera-data", ego_frame="camera_ego", sensor_frame="camera_sensor"
    )
    return lidar, camera


def test_a_packet_carries_its_tokens_timestamp_and_file() -> None:
    """These are what tie a measurement back to one exact sample of one exact scene."""

    lidar, _ = packets()

    assert lidar.sample_token == "sample-0"
    assert lidar.sample_data_token == "lidar-data"
    assert lidar.timestamp_us == 1_538_984_233_547_259
    assert isinstance(lidar.timestamp_us, int)
    assert lidar.file_relative_path == "samples/LIDAR_TOP/scene.pcd.bin"


def test_the_calibrated_sensor_is_read_as_sensor_into_ego() -> None:
    """nuScenes stores the sensor pose in ego coordinates, so the arrow points that way."""

    lidar, camera = packets()

    assert (lidar.calibrated_sensor.target, lidar.calibrated_sensor.source) == (
        "lidar_ego",
        "lidar_sensor",
    )
    assert lidar.calibrated_sensor.value.translation_xyz_m == pytest.approx((0.9, 0.0, 1.8))
    assert (camera.calibrated_sensor.target, camera.calibrated_sensor.source) == (
        "camera_ego",
        "camera_sensor",
    )


def test_the_ego_pose_is_read_as_ego_into_global() -> None:
    """The other direction is the classic mistake, and the labels are what prevent it."""

    lidar, camera = packets(lidar_ego_translation=(100.0, 200.0, 0.0))

    assert (lidar.ego_pose.target, lidar.ego_pose.source) == ("global", "lidar_ego")
    assert lidar.ego_pose.value.translation_xyz_m == pytest.approx((100.0, 200.0, 0.0))
    assert (camera.ego_pose.target, camera.ego_pose.source) == ("global", "camera_ego")


def test_the_whole_quaternion_survives_and_not_only_its_yaw() -> None:
    """Reducing an orientation to yaw is a common shortcut and silently tilts every point."""

    lidar, _ = packets(lidar_sensor_rotation=TILTED)

    assert lidar.calibrated_sensor.value.rotation_wxyz == pytest.approx(TILTED, abs=1e-12)

    # A yaw-only reconstruction of the same orientation moves points elsewhere.
    yaw = math.atan2(
        2 * (TILTED[0] * TILTED[3] + TILTED[1] * TILTED[2]),
        1 - 2 * (TILTED[2] ** 2 + TILTED[3] ** 2),
    )
    yaw_only = SE3(
        rotation_wxyz=(math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)),
        translation_xyz_m=(0.9, 0.0, 1.8),
    )
    probe = np.array([[1.0, 2.0, 3.0]])
    assert not np.allclose(
        transform_points(lidar.calibrated_sensor.value, probe),
        transform_points(yaw_only, probe),
    )


def test_the_chain_runs_from_the_lidar_sensor_to_the_camera_sensor() -> None:
    """The labels state the direction, so a caller cannot mistake which way it goes."""

    from bevcalib.nuscenes_adapter.frames import lidar_to_camera_chain

    chain = lidar_to_camera_chain(*packets())

    assert (chain.target, chain.source) == ("camera_sensor", "lidar_sensor")


def test_the_chain_matches_the_four_links_applied_one_at_a_time() -> None:
    """Composed once or applied four times, the answer must be the same."""

    from bevcalib.geometry.frames import inverse_framed
    from bevcalib.nuscenes_adapter.frames import lidar_to_camera_chain

    lidar, camera = packets(lidar_sensor_rotation=TILTED)
    points = np.array([[10.0, 0.0, 0.0], [0.0, 5.0, -1.0], [0.0, 0.0, 0.0]])

    composed = transform_points(lidar_to_camera_chain(lidar, camera).value, points)

    stepwise = transform_points(lidar.calibrated_sensor.value, points)
    stepwise = transform_points(lidar.ego_pose.value, stepwise)
    stepwise = transform_points(inverse_framed(camera.ego_pose).value, stepwise)
    stepwise = transform_points(inverse_framed(camera.calibrated_sensor).value, stepwise)

    np.testing.assert_allclose(composed, stepwise, atol=1e-9)


def test_the_chain_uses_both_ego_poses_and_not_one_of_them_twice() -> None:
    """The sweep and the exposure are different instants, so they are different poses.

    At 50 km/h the vehicle covers about 0.7 m in the 50 ms between them, which is
    the same size as the translation faults this study injects deliberately. If
    the two poses were collapsed the mistake would be indistinguishable from the
    effect being measured.
    """

    from bevcalib.nuscenes_adapter.frames import lidar_to_camera_chain

    lidar, camera = packets(
        lidar_ego_translation=(0.0, 0.0, 0.0), camera_ego_translation=(0.7, 0.0, 0.0)
    )
    stationary_lidar, stationary_camera = packets(
        lidar_ego_translation=(0.0, 0.0, 0.0), camera_ego_translation=(0.0, 0.0, 0.0)
    )
    origin = np.zeros((1, 3))

    moved = transform_points(lidar_to_camera_chain(lidar, camera).value, origin)
    still = transform_points(
        lidar_to_camera_chain(stationary_lidar, stationary_camera).value, origin
    )

    assert float(np.linalg.norm(moved - still)) == pytest.approx(0.7, abs=1e-9)


def test_giving_the_two_packets_in_the_wrong_order_is_refused() -> None:
    """Swapped arguments still compose into a valid transform, pointing the other way."""

    from bevcalib.nuscenes_adapter.frames import lidar_to_camera_chain

    lidar, camera = packets()

    with pytest.raises(
        ValueError,
        match=r"^the first packet must be a LiDAR reading, but its sensor frame is .*, not 'lidar_sensor'$",
    ):
        lidar_to_camera_chain(camera, lidar)


def test_a_record_missing_a_field_fails_closed() -> None:
    """A silently defaulted pose would be an identity, which looks like a perfect calibration."""

    from bevcalib.nuscenes_adapter.samples import read_sensor_packet

    tables = records()
    del tables["calibrated_sensor"]["cs-lidar"]["rotation"]

    with pytest.raises(KeyError):
        read_sensor_packet(
            lookup_for(tables), "lidar-data", ego_frame="lidar_ego", sensor_frame="lidar_sensor"
        )


def test_the_point_loader_keeps_all_five_lidar_columns(tmp_path: Path) -> None:
    """Intensity and ring are features of the study, not decoration to be dropped."""

    from bevcalib.nuscenes_adapter.samples import LIDAR_COLUMN_NAMES, read_lidar_points

    values = [
        [1.0, 2.0, 3.0, 42.0, 7.0],
        [-4.0, 5.0, -6.0, 13.0, 31.0],
    ]
    path = tmp_path / "scene.pcd.bin"
    path.write_bytes(struct.pack("<10f", *[value for row in values for value in row]))

    points = read_lidar_points(path)

    assert LIDAR_COLUMN_NAMES == ("x", "y", "z", "intensity", "ring")
    assert points.shape == (2, 5)
    assert points.dtype == np.float64
    np.testing.assert_allclose(points, values)


def test_a_point_file_that_is_not_a_whole_number_of_points_is_rejected(tmp_path: Path) -> None:
    """A truncated download would otherwise reshape into silently shifted columns."""

    from bevcalib.nuscenes_adapter.samples import read_lidar_points

    path = tmp_path / "truncated.pcd.bin"
    path.write_bytes(struct.pack("<7f", *range(7)))

    with pytest.raises(
        ValueError, match=r"^a LiDAR sweep must be a whole number of five-column points, got "
    ):
        read_lidar_points(path)


def test_an_empty_point_file_reads_as_no_points(tmp_path: Path) -> None:
    """A sweep with no returns is legitimate and must not need a special case."""

    from bevcalib.nuscenes_adapter.samples import read_lidar_points

    path = tmp_path / "empty.pcd.bin"
    path.write_bytes(b"")

    assert read_lidar_points(path).shape == (0, 5)


def test_the_rotation_matrix_of_a_read_transform_matches_its_quaternion() -> None:
    """One assertion tying the adapter output back to the geometry package."""

    lidar, _ = packets(lidar_sensor_rotation=TILTED)

    np.testing.assert_allclose(
        quaternion_to_matrix(lidar.calibrated_sensor.value.rotation_wxyz),
        quaternion_to_matrix(TILTED),
        atol=1e-12,
    )


def test_passing_the_lidar_packet_for_both_arguments_is_refused() -> None:
    """Both guards must be reachable; the first one alone would hide the second."""

    from bevcalib.nuscenes_adapter.frames import lidar_to_camera_chain

    lidar, _ = packets()

    with pytest.raises(
        ValueError,
        match=r"^the second packet must be a camera reading, but its sensor frame is .*, not 'camera_sensor'$",
    ):
        lidar_to_camera_chain(lidar, lidar)
