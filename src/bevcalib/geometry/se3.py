"""Rigid transforms under the `T_target_source` naming rule."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .quaternions import (
    Float64Array,
    Quaternion,
    matrix_to_quaternion,
    quaternion_to_matrix,
)


@dataclass(frozen=True)
class SE3:
    """A rigid transform, canonical and rigid by construction.

    The quaternion is normalised and the translation is checked when the value is
    made, so no function that receives an `SE3` has to ask whether it is really a
    rotation. A non-unit quaternion would scale every point it touched, which
    looks exactly like a calibration error in the results.
    """

    rotation_wxyz: Quaternion
    translation_xyz_m: tuple[float, float, float]

    def __post_init__(self) -> None:
        from .quaternions import normalize_quaternion_wxyz

        object.__setattr__(self, "rotation_wxyz", normalize_quaternion_wxyz(self.rotation_wxyz))

        translation = tuple(float(value) for value in self.translation_xyz_m)
        if len(translation) != 3:
            raise ValueError(f"translation must have three components, got {len(translation)}")
        if not all(math.isfinite(value) for value in translation):
            raise ValueError(f"translation components must be finite, got {translation}")
        object.__setattr__(self, "translation_xyz_m", translation)


def compose(target_from_mid: SE3, mid_from_source: SE3) -> SE3:
    """Return `T_target_source` from `T_target_mid` and `T_mid_source`.

    The right-hand transform is applied first. Reversing the two arguments still
    returns a valid rigid transform, which is why the order is stated here and
    pinned by a test rather than left to the reader.
    """

    left_rotation = quaternion_to_matrix(target_from_mid.rotation_wxyz)
    rotation = left_rotation @ quaternion_to_matrix(mid_from_source.rotation_wxyz)
    translation = left_rotation @ np.asarray(
        mid_from_source.translation_xyz_m, dtype=np.float64
    ) + np.asarray(target_from_mid.translation_xyz_m, dtype=np.float64)
    return SE3(
        rotation_wxyz=matrix_to_quaternion(rotation),
        translation_xyz_m=(float(translation[0]), float(translation[1]), float(translation[2])),
    )


def inverse(target_from_source: SE3) -> SE3:
    """Return `T_source_target`, the transform that puts the points back."""

    rotation = quaternion_to_matrix(target_from_source.rotation_wxyz).T
    translation = -rotation @ np.asarray(target_from_source.translation_xyz_m, dtype=np.float64)
    return SE3(
        rotation_wxyz=matrix_to_quaternion(rotation),
        translation_xyz_m=(float(translation[0]), float(translation[1]), float(translation[2])),
    )


def transform_points(target_from_source: SE3, points_n3: Float64Array) -> Float64Array:
    """Map `[N, 3]` row-vector points from `source` into `target`.

    Row vectors are the public convention, so the rotation is applied on the
    right as `points @ R.T`. A `[3, N]` array would broadcast against a 3x3
    rotation without complaint and return confident nonsense, which is why the
    shape is checked rather than inferred.
    """

    points = np.asarray(points_n3, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points must have shape [N, 3], got {points.shape}")

    rotation = quaternion_to_matrix(target_from_source.rotation_wxyz)
    translation = np.asarray(target_from_source.translation_xyz_m, dtype=np.float64)
    return points @ rotation.T + translation
