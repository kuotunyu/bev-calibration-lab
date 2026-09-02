"""Quaternion rotations with a canonical sign and a validated matrix conversion."""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt

Float64Array = npt.NDArray[np.float64]
Quaternion = tuple[float, float, float, float]

# A unit quaternion always has one component of magnitude at least 0.5, so this
# only ever classifies genuine zeros.
_SIGN_TOLERANCE = 1e-12
# Below this the direction is numerical noise rather than a rotation axis.
_MINIMUM_NORM = 1e-12
# The plan fixes this: singular values of a rotation are all exactly one.
ORTHONORMALITY_TOLERANCE = 1e-9


def normalize_quaternion_wxyz(q: Quaternion) -> Quaternion:
    """Return the unit quaternion for `q` with its first nonzero component positive.

    A quaternion and its negation are the same rotation, so without a fixed sign
    two records of one rotation compare unequal and no test of a stored value
    means anything. The rule is the cheapest one that is total: normalise, then
    make the first component that is not zero positive.
    """

    values = np.asarray(q, dtype=np.float64)
    if values.shape != (4,):
        raise ValueError(f"quaternion must have four components, got shape {values.shape}")
    if not np.all(np.isfinite(values)):
        raise ValueError("quaternion components must be finite")

    norm = float(np.linalg.norm(values))
    if norm < _MINIMUM_NORM:
        raise ValueError(f"quaternion norm {norm!r} is too small to define a rotation")

    # A unit quaternion always has a component of magnitude at least 0.5, so this
    # search cannot come up empty and there is no third case to write.
    unit = values / norm
    significant = np.flatnonzero(np.abs(unit) > _SIGN_TOLERANCE)
    if unit[significant[0]] < 0.0:
        unit = -unit
    return (float(unit[0]), float(unit[1]), float(unit[2]), float(unit[3]))


def quaternion_to_matrix(q: Quaternion) -> Float64Array:
    """Return the 3x3 right-handed rotation matrix for `q`, normalising it first."""

    w, x, y, z = normalize_quaternion_wxyz(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def matrix_to_quaternion(matrix: Float64Array) -> Quaternion:
    """Return the canonical quaternion for a validated right-handed rotation matrix.

    Which of the four branches below is used is decided by the largest diagonal
    term rather than by the trace alone. The textbook single-branch formula
    divides by a quantity that vanishes as the rotation approaches half a turn,
    and a calibration chain reaches that region as soon as two sensors face
    roughly opposite directions.
    """

    values = np.asarray(matrix, dtype=np.float64)
    if values.shape != (3, 3):
        raise ValueError(f"rotation matrix must have shape 3x3, got {values.shape}")
    if not np.all(np.isfinite(values)):
        raise ValueError("rotation matrix entries must be finite")

    singular_values = np.linalg.svd(values, compute_uv=False)
    if not np.allclose(singular_values, 1.0, atol=ORTHONORMALITY_TOLERANCE, rtol=0.0):
        raise ValueError(
            f"matrix is not orthonormal within {ORTHONORMALITY_TOLERANCE}: "
            f"singular values {singular_values.tolist()}"
        )
    determinant = float(np.linalg.det(values))
    if abs(determinant - 1.0) > ORTHONORMALITY_TOLERANCE:
        raise ValueError(f"rotation must be right-handed, but the determinant is {determinant!r}")

    trace = float(values[0, 0] + values[1, 1] + values[2, 2])
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        candidate = (
            0.25 * scale,
            float(values[2, 1] - values[1, 2]) / scale,
            float(values[0, 2] - values[2, 0]) / scale,
            float(values[1, 0] - values[0, 1]) / scale,
        )
    elif values[0, 0] > values[1, 1] and values[0, 0] > values[2, 2]:
        scale = math.sqrt(1.0 + float(values[0, 0] - values[1, 1] - values[2, 2])) * 2.0
        candidate = (
            float(values[2, 1] - values[1, 2]) / scale,
            0.25 * scale,
            float(values[0, 1] + values[1, 0]) / scale,
            float(values[0, 2] + values[2, 0]) / scale,
        )
    elif values[1, 1] > values[2, 2]:
        scale = math.sqrt(1.0 + float(values[1, 1] - values[0, 0] - values[2, 2])) * 2.0
        candidate = (
            float(values[0, 2] - values[2, 0]) / scale,
            float(values[0, 1] + values[1, 0]) / scale,
            0.25 * scale,
            float(values[1, 2] + values[2, 1]) / scale,
        )
    else:
        scale = math.sqrt(1.0 + float(values[2, 2] - values[0, 0] - values[1, 1])) * 2.0
        candidate = (
            float(values[1, 0] - values[0, 1]) / scale,
            float(values[0, 2] + values[2, 0]) / scale,
            float(values[1, 2] + values[2, 1]) / scale,
            0.25 * scale,
        )
    return normalize_quaternion_wxyz(candidate)
