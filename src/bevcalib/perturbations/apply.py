"""Apply a metadata calibration fault to a transform, on one fixed side."""

from __future__ import annotations

import math

import numpy as np

from bevcalib.artifacts.results import CalibrationFaultModel
from bevcalib.geometry.quaternions import Float64Array, matrix_to_quaternion, quaternion_to_matrix
from bevcalib.geometry.se3 import SE3, compose, inverse


def _about_x(radians: float) -> Float64Array:
    cos, sin = math.cos(radians), math.sin(radians)
    return np.array([[1.0, 0.0, 0.0], [0.0, cos, -sin], [0.0, sin, cos]])


def _about_y(radians: float) -> Float64Array:
    cos, sin = math.cos(radians), math.sin(radians)
    return np.array([[cos, 0.0, sin], [0.0, 1.0, 0.0], [-sin, 0.0, cos]])


def _about_z(radians: float) -> Float64Array:
    cos, sin = math.cos(radians), math.sin(radians)
    return np.array([[cos, -sin, 0.0], [sin, cos, 0.0], [0.0, 0.0, 1.0]])


def fault_to_se3(fault: CalibrationFaultModel) -> SE3:
    """Turn a fault's roll, pitch, yaw and translation into a rigid transform.

    The Euler convention is fixed as `R = Rz(yaw) · Ry(pitch) · Rx(roll)`: roll
    about x first, then pitch about y, then yaw about z, all about fixed axes.
    With one angle nonzero every convention agrees, so only a test with all three
    nonzero can tell them apart, and there is one.

    The requested timing offset is deliberately ignored here. Timing changes which
    LiDAR sweep is paired with the fixed camera, never the calibration metadata.
    """

    roll, pitch, yaw = (math.radians(value) for value in fault.rotation_rpy_deg)
    rotation = _about_z(yaw) @ _about_y(pitch) @ _about_x(roll)
    return SE3(
        rotation_wxyz=matrix_to_quaternion(rotation),
        translation_xyz_m=fault.translation_xyz_m,
    )


def apply_metadata_fault(true_target_from_source: SE3, fault: CalibrationFaultModel) -> SE3:
    """Return the calibration a faulty run believes it has: `assumed = true ∘ fault`.

    The fault is composed on the SOURCE side, which is the sensor's own frame.
    That is what a miscalibrated extrinsic physically is: the sensor is believed
    to sit slightly rotated or shifted from where it really does, measured in its
    own axes. Composing on the other side is equally valid arithmetic and answers
    a different question, so the choice is stated here, in the coordinate contract
    and in a test that shows the two differ.

    Only metadata moves. No point and no pixel is touched, because an error that
    mixed a sensing change with a calibration change could not be attributed to
    either.
    """

    return compose(true_target_from_source, fault_to_se3(fault))


def se3_to_fault(value: SE3, requested_time_offset_ms: int = 0) -> CalibrationFaultModel:
    """Recover the roll, pitch, yaw and translation of a rigid transform.

    The exact inverse of `fault_to_se3`, decomposing `R = Rz(yaw) . Ry(pitch) .
    Rx(roll)` back into its three angles. It refuses a rotation near a quarter
    turn in pitch, where roll and yaw become the same degree of freedom and the
    decomposition would silently return zeros for both. No fault in this study
    comes anywhere near that, so the guard is about protecting the next caller
    rather than this one.
    """

    matrix = quaternion_to_matrix(value.rotation_wxyz)
    if abs(matrix[2, 0]) > 1.0 - 1e-9:
        raise ValueError(
            "cannot decompose a rotation with a pitch of a quarter turn: roll and yaw "
            "are not separable there"
        )
    return CalibrationFaultModel(
        rotation_rpy_deg=(
            math.degrees(math.atan2(matrix[2, 1], matrix[2, 2])),
            math.degrees(math.asin(-matrix[2, 0])),
            math.degrees(math.atan2(matrix[1, 0], matrix[0, 0])),
        ),
        translation_xyz_m=value.translation_xyz_m,
        requested_time_offset_ms=requested_time_offset_ms,
    )


def inverse_fault(fault: CalibrationFaultModel) -> CalibrationFaultModel:
    """Return the correction that exactly undoes `fault`.

    This is NOT the fault with its signs flipped. Inverting a rigid transform
    rotates the translation as well as negating it, so for a fault with both a
    rotation and an offset the naive negation is wrong by the sine of the angle
    times the offset. That is small, systematic, and exactly the kind of error a
    learned corrector would faithfully reproduce.

    A timing fault is refused: no 6DoF pose expresses one, so it has no inverse
    in this representation.
    """

    if fault.requested_time_offset_ms != 0:
        raise ValueError(
            "a timing fault has no 6DoF inverse: requested_time_offset_ms is "
            f"{fault.requested_time_offset_ms}, not 0"
        )
    return se3_to_fault(inverse(fault_to_se3(fault)))
