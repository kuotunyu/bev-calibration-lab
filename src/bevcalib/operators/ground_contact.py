"""Where a box touches the ground, reconstructed from one pixel and a calibration."""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass

import numpy as np

from bevcalib.geometry.frames import FramedTransform
from bevcalib.geometry.projection import project_camera, validate_intrinsic
from bevcalib.geometry.quaternions import Float64Array, Quaternion, quaternion_to_matrix
from bevcalib.geometry.se3 import SE3, inverse, transform_points

# Past this range one pixel of reconstruction error is metres of ground position,
# and the study would be reporting the pixel grid rather than the calibration.
MAX_RANGE_M = 80.0
# Below this the ray is parallel to the plane to within any useful precision, and
# the intersection distance is the ratio of two numbers that are both noise.
_PARALLEL_TOLERANCE = 1e-9


@dataclass(frozen=True)
class GroundContactObservation:
    """One box's ground position, as the truth has it and as the calibration says.

    `valid` is the only field that certifies the rest. When it is false the
    reconstructed position is `nan`, deliberately, so that a consumer which
    forgets to check cannot quietly average a placeholder.
    """

    box_token: str
    true_uv: tuple[float, float]
    assumed_ground_xy_m: tuple[float, float]
    oracle_ground_xy_m: tuple[float, float]
    range_m: float
    valid: bool


def ground_plane_z_from_ego(ego_pose: FramedTransform) -> float:
    """Return the ground height in global, taken as the ego origin's own height.

    This is the flat-world simplification, and it lives here so that it is stated
    once rather than assumed at every call site. The ground is treated as a
    horizontal plane through the ego origin, which is wrong on a slope and wrong
    at a kerb. It is defensible because the study measures the CHANGE a
    calibration fault causes, and the same plane is used for the faulted and the
    true reconstruction, so a wrong plane largely cancels. It would not be
    defensible as an absolute claim about where a box is.
    """

    if ego_pose.target != "global":
        raise ValueError(
            f"the ground plane comes from an ego pose in global, not from a "
            f"{ego_pose.target!r} transform"
        )
    return float(ego_pose.value.translation_xyz_m[2])


def bottom_center_global(
    box_center: Float64Array,
    size_wlh: tuple[float, float, float],
    orientation_wxyz: Quaternion,
) -> Float64Array:
    """Return the point at the middle of a box's underside, in global coordinates.

    The offset runs along the box's OWN up axis, not along global z, so a box
    that is tipped over has its underside somewhere else entirely. With only a
    yaw the two agree, which is why a yaw-only test cannot tell a correct
    implementation from one that rotates the wrong vector.
    """

    centre = np.asarray(box_center, dtype=np.float64)
    if centre.shape != (3,):
        raise ValueError(f"a box centre must be three numbers, got shape {centre.shape}")

    width, length, height = (float(value) for value in size_wlh)
    if not all(value > 0.0 for value in (width, length, height)):
        raise ValueError(
            f"a box must have positive width, length and height, got {(width, length, height)}"
        )

    rotation = quaternion_to_matrix(orientation_wxyz)
    return centre + rotation @ np.array([0.0, 0.0, -height / 2.0])


def reconstruct_ground_contact(
    uv: tuple[float, float],
    assumed_camera_from_global: SE3,
    intrinsic: Float64Array,
    ground_z_global: float,
) -> tuple[float, float] | None:
    """Intersect the ray through one pixel with the ground plane, or return `None`.

    `None` means the geometry has no answer: the ray runs parallel to the plane,
    or it meets it behind the camera. Both happen for real pixels above the
    horizon, so they are outcomes rather than errors. A malformed question, a
    pixel or a plane that is not a number, raises instead.
    """

    u, v = (float(value) for value in uv)
    if not (math.isfinite(u) and math.isfinite(v) and math.isfinite(ground_z_global)):
        raise ValueError(
            f"the pixel and the ground height must be finite, got {uv} and {ground_z_global}"
        )
    matrix = validate_intrinsic(intrinsic)

    global_from_camera = inverse(assumed_camera_from_global)
    origin = np.asarray(global_from_camera.translation_xyz_m, dtype=np.float64)
    ray_camera = np.linalg.solve(matrix, np.array([u, v, 1.0]))
    direction = quaternion_to_matrix(global_from_camera.rotation_wxyz) @ ray_camera

    if abs(direction[2]) < _PARALLEL_TOLERANCE:
        return None
    distance = (ground_z_global - origin[2]) / direction[2]
    if distance <= 0.0:
        return None

    contact = origin + distance * direction
    return (float(contact[0]), float(contact[1]))


def observe_ground_contact(
    *,
    box_token: str,
    box_center_global: Float64Array,
    size_wlh: tuple[float, float, float],
    orientation_wxyz: Quaternion,
    true_camera_from_global: SE3,
    assumed_camera_from_global: SE3,
    intrinsic: Float64Array,
    image_size_wh: tuple[int, int],
    ground_z_global: float,
) -> GroundContactObservation:
    """Measure what a calibration fault does to one box's position in bird's-eye view.

    The observation is oracle-controlled on purpose. The contact point comes from
    the ground-truth box and is projected with the TRUE calibration, so the
    detection is perfect by construction and contributes no error of its own. Only
    the back-projection uses the assumed calibration and the horizontal ego plane.
    The oracle XY is the ground-truth box bottom, which can lie off that plane.
    Absolute XY residual therefore includes plane-model error even at zero fault;
    compare against the zero-fault baseline before attributing a change to calibration.
    """

    contact = bottom_center_global(box_center_global, size_wlh, orientation_wxyz)
    oracle = (float(contact[0]), float(contact[1]))
    camera_origin = np.asarray(inverse(true_camera_from_global).translation_xyz_m, dtype=np.float64)
    range_m = float(np.linalg.norm(contact - camera_origin))

    unusable = GroundContactObservation(
        box_token=box_token,
        true_uv=(math.nan, math.nan),
        assumed_ground_xy_m=(math.nan, math.nan),
        oracle_ground_xy_m=oracle,
        range_m=range_m,
        valid=False,
    )

    projection = project_camera(
        transform_points(true_camera_from_global, contact[None, :]), intrinsic, image_size_wh
    )
    if not bool(projection.valid[0]) or range_m > MAX_RANGE_M:
        return unusable

    true_uv = (float(projection.uv[0, 0]), float(projection.uv[0, 1]))
    assumed = reconstruct_ground_contact(
        true_uv, assumed_camera_from_global, intrinsic, ground_z_global
    )
    if assumed is None:
        # The oracle observation exists, so it is recorded; only the faulty
        # calibration failed to turn it back into a position.
        return dataclasses.replace(unusable, true_uv=true_uv)

    return GroundContactObservation(
        box_token=box_token,
        true_uv=true_uv,
        assumed_ground_xy_m=assumed,
        oracle_ground_xy_m=oracle,
        range_m=range_m,
        valid=True,
    )
