"""Pinhole projection that preserves every point and records why each one is invalid."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from .quaternions import Float64Array

BoolArray = npt.NDArray[np.bool_]


@dataclass(frozen=True)
class ProjectionResult:
    """One row per input point, with the reasons a row cannot be scored.

    `in_front` and `in_image` are separate on purpose. Negating a point negates
    both the numerator and the denominator of the projection, so a point behind
    the camera lands on exactly the same pixel as its mirror image in front of
    it. Code that checks only the image bounds draws it, and the resulting error
    looks like a calibration fault rather than a missing sign test.
    """

    uv: Float64Array
    optical_depth: Float64Array
    in_front: BoolArray
    in_image: BoolArray
    valid: BoolArray


def validate_image_size(image_size_wh: tuple[int, int]) -> tuple[int, int]:
    """Return a validated `(width, height)` in pixels.

    Shared with the rasteriser so the two cannot disagree about the canvas.
    """

    width, height = (int(value) for value in image_size_wh)
    if width <= 0 or height <= 0:
        raise ValueError(f"image size must be positive, got {(width, height)}")
    return width, height


def _validated_intrinsic(intrinsic_3x3: Float64Array) -> Float64Array:
    intrinsic = np.asarray(intrinsic_3x3, dtype=np.float64)
    if intrinsic.shape != (3, 3):
        raise ValueError(f"intrinsic must have shape 3x3, got {intrinsic.shape}")
    if not np.all(np.isfinite(intrinsic)):
        raise ValueError("intrinsic entries must be finite")
    if not np.array_equal(intrinsic[2], [0.0, 0.0, 1.0]) or intrinsic[1, 0] != 0.0:
        # A bottom row other than [0, 0, 1] rescales the homogeneous divisor, which
        # rescales every depth by a constant nobody would think to look for.
        raise ValueError(
            "intrinsic must be an upper-triangular pinhole matrix with bottom row [0, 0, 1]"
        )
    if intrinsic[0, 0] <= 0.0 or intrinsic[1, 1] <= 0.0:
        raise ValueError("intrinsic focal lengths must be positive")
    return intrinsic


def project_camera(
    points_camera_n3: Float64Array,
    intrinsic_3x3: Float64Array,
    image_size_wh: tuple[int, int],
) -> ProjectionResult:
    """Project `[N, 3]` camera-frame points, returning one row per input point.

    Nothing is filtered. A projection that drops points makes per-sample errors
    incomparable across faults, because each fault would then be scored over a
    different set of points, and a fault that pushes points out of frame would
    read as an improvement.
    """

    points = np.asarray(points_camera_n3, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points must have shape [N, 3], got {points.shape}")
    intrinsic = _validated_intrinsic(intrinsic_3x3)
    width, height = validate_image_size(image_size_wh)

    optical_depth = np.array(points[:, 2], dtype=np.float64)
    finite = np.all(np.isfinite(points), axis=1)
    in_front = finite & (optical_depth > 0.0)
    # Behind-camera points are projected too, precisely so that `in_image` can
    # disagree with `in_front`. Only zero depth has no projection at all.
    projectable = finite & (optical_depth != 0.0)

    uv = np.full((points.shape[0], 2), np.nan, dtype=np.float64)
    scaled = points[projectable] @ intrinsic.T
    uv[projectable] = scaled[:, :2] / scaled[:, 2:3]

    in_image = (uv[:, 0] >= 0.0) & (uv[:, 0] < width) & (uv[:, 1] >= 0.0) & (uv[:, 1] < height)
    return ProjectionResult(
        uv=uv,
        optical_depth=optical_depth,
        in_front=in_front,
        in_image=in_image,
        valid=in_front & in_image,
    )
