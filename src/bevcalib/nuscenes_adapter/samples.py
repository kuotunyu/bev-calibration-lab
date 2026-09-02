"""Read nuScenes records into frame-safe values, through an injected lookup."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np

from bevcalib.geometry.frames import FramedTransform, FrameName
from bevcalib.geometry.quaternions import Float64Array
from bevcalib.geometry.se3 import SE3

from .frames import SensorPacket

# `(table, token) -> record`. Injecting the lookup is what lets every rule in this
# module be tested on synthetic tables, with no dataset and no devkit import.
NuScenesLookup = Callable[[str, str], Mapping[str, Any]]

LIDAR_COLUMN_NAMES: tuple[str, ...] = ("x", "y", "z", "intensity", "ring")


def _as_se3(record: Mapping[str, Any]) -> SE3:
    """Build a rigid transform from a nuScenes pose record.

    nuScenes writes rotations as `[w, x, y, z]`, the same order this package uses,
    and unpacking into named components means a record with the wrong number of
    values fails here rather than broadcasting somewhere later.
    """

    w, x, y, z = (float(value) for value in record["rotation"])
    tx, ty, tz = (float(value) for value in record["translation"])
    return SE3(rotation_wxyz=(w, x, y, z), translation_xyz_m=(tx, ty, tz))


def read_sensor_packet(
    lookup: NuScenesLookup,
    sample_data_token: str,
    *,
    ego_frame: FrameName,
    sensor_frame: FrameName,
) -> SensorPacket:
    """Assemble one `SensorPacket` from the `sample_data` record and its two poses.

    The directions are the ones nuScenes actually stores: `calibrated_sensor` is
    the sensor expressed in ego, and `ego_pose` is the ego expressed in global.
    Labelling them the other way round is the mistake the frame names exist to
    catch, so they are labelled once, here.
    """

    data = lookup("sample_data", sample_data_token)
    calibrated = lookup("calibrated_sensor", data["calibrated_sensor_token"])
    ego = lookup("ego_pose", data["ego_pose_token"])

    return SensorPacket(
        sample_token=str(data["sample_token"]),
        sample_data_token=str(data["token"]),
        timestamp_us=int(data["timestamp"]),
        calibrated_sensor=FramedTransform(
            target=ego_frame, source=sensor_frame, value=_as_se3(calibrated)
        ),
        ego_pose=FramedTransform(target="global", source=ego_frame, value=_as_se3(ego)),
        file_relative_path=str(data["filename"]),
    )


def read_lidar_points(path: Path) -> Float64Array:
    """Read one LiDAR sweep as `[N, 5]`: x, y, z, intensity, ring.

    Intensity and ring are kept because they are inputs to the edge operator, not
    decoration. A file that is not a whole number of five-column points is
    refused rather than reshaped, since a truncated download would otherwise
    become an array whose columns are all shifted by one.
    """

    raw = np.fromfile(path, dtype=np.float32)
    columns = len(LIDAR_COLUMN_NAMES)
    if raw.size % columns != 0:
        raise ValueError(
            f"a LiDAR sweep must be a whole number of five-column points, got {raw.size} values"
        )
    return raw.reshape(-1, columns).astype(np.float64)
