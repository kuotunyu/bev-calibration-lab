"""Frame-safe rigid-body geometry under the `T_target_source` naming rule."""

from .frames import FRAME_NAMES, FramedTransform, FrameName, compose_framed
from .quaternions import (
    matrix_to_quaternion,
    normalize_quaternion_wxyz,
    quaternion_to_matrix,
)
from .se3 import SE3, compose, inverse, transform_points

__all__ = [
    "FRAME_NAMES",
    "SE3",
    "FrameName",
    "FramedTransform",
    "compose",
    "compose_framed",
    "inverse",
    "matrix_to_quaternion",
    "normalize_quaternion_wxyz",
    "quaternion_to_matrix",
    "transform_points",
]
