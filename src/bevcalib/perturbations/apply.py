"""Apply a metadata calibration fault to a transform, on one fixed side."""

from __future__ import annotations

import math

import numpy as np

from bevcalib.artifacts.results import CalibrationFaultModel
from bevcalib.geometry.quaternions import Float64Array, matrix_to_quaternion
from bevcalib.geometry.se3 import SE3, compose


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
    camera frame is paired with the sweep, never the geometry of the pairing.
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
