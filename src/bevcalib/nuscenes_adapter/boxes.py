"""Bring 3D boxes from their native global frame into a camera."""

from __future__ import annotations

import numpy as np

from bevcalib.geometry.frames import compose_framed, inverse_framed
from bevcalib.geometry.quaternions import (
    Float64Array,
    Quaternion,
    matrix_to_quaternion,
    quaternion_to_matrix,
)
from bevcalib.geometry.se3 import transform_points

from .frames import SensorPacket


def transform_global_box_to_camera(
    box_center_global: Float64Array,
    box_orientation_wxyz: Quaternion,
    camera: SensorPacket,
) -> tuple[Float64Array, Quaternion]:
    """Return the box centre and orientation expressed in the camera sensor frame.

    nuScenes stores box centres and orientations in global, not relative to any
    sensor. Treating a global centre as if it were already ego-relative is worth
    hundreds of metres and raises nothing: the boxes land somewhere plausible and
    wrong. The orientation travels through the same rotation as the centre,
    because a box whose centre moves and whose heading does not is a box facing
    the wrong way.
    """

    if camera.calibrated_sensor.source != "camera_sensor":
        raise ValueError(
            "boxes are projected into a camera, but this packet's sensor frame is "
            f"{camera.calibrated_sensor.source!r}, not 'camera_sensor'"
        )

    centre = np.asarray(box_center_global, dtype=np.float64)
    if centre.shape != (3,):
        raise ValueError(f"a box centre must be three numbers, got shape {centre.shape}")

    camera_from_global = compose_framed(
        inverse_framed(camera.calibrated_sensor), inverse_framed(camera.ego_pose)
    )
    centre_camera = transform_points(camera_from_global.value, centre[None, :])[0]
    rotation = quaternion_to_matrix(camera_from_global.value.rotation_wxyz) @ quaternion_to_matrix(
        box_orientation_wxyz
    )
    return centre_camera, matrix_to_quaternion(rotation)
