"""How wrong a calibration is, and whether that counts as recovered."""

from __future__ import annotations

import math

import numpy as np

from bevcalib.artifacts.results import CalibrationFaultModel
from bevcalib.geometry.se3 import SE3, compose, inverse
from bevcalib.perturbations.apply import fault_to_se3

# From the perturbation matrix. Both must hold at once: a corrector that fixes the
# rotation and leaves the offset has recovered half a calibration, and the half it
# left is the one that moves a box in bird's-eye view.
RECOVERY_ROTATION_THRESHOLD_DEG = 0.25
RECOVERY_TRANSLATION_THRESHOLD_M = 0.05


def rotation_geodesic_error_deg(estimate: SE3, truth: SE3) -> float:
    """Return the single angle of the rotation that takes `estimate` to `truth`.

    A geodesic, not a sum of per-axis differences. The per-axis version
    over-counts a rotation split across axes, and both answers look plausible on
    a chart: one degree of roll and one of pitch are about 1.414 degrees apart,
    not two.
    """

    relative = compose(inverse(estimate), truth)
    # Clamped because a unit quaternion's w can land a few ulps past one.
    return math.degrees(2.0 * math.acos(min(1.0, abs(relative.rotation_wxyz[0]))))


def calibration_errors(
    estimate: CalibrationFaultModel, truth: CalibrationFaultModel
) -> dict[str, float]:
    """Return every error worth reporting about one estimate: the geodesic and the parts.

    Both are needed. The geodesic is what a recovery is judged on; the signed
    per-axis components are what tell you a corrector is consistently short on one
    axis rather than noisy on all of them, which is a different problem with a
    different fix.
    """

    for name, value in (("estimate", estimate), ("truth", truth)):
        if value.requested_time_offset_ms != 0:
            raise ValueError(
                f"the {name} carries a timing offset of {value.requested_time_offset_ms} ms, "
                "and there is no pose difference between two faults that differ only in time"
            )

    difference = np.asarray(estimate.translation_xyz_m, dtype=np.float64) - np.asarray(
        truth.translation_xyz_m, dtype=np.float64
    )
    norm = float(np.linalg.norm(difference))
    angles = [
        left - right
        for left, right in zip(estimate.rotation_rpy_deg, truth.rotation_rpy_deg, strict=True)
    ]
    return {
        "rotation_geodesic_error_deg": rotation_geodesic_error_deg(
            fault_to_se3(estimate), fault_to_se3(truth)
        ),
        "translation_error_m": norm,
        "translation_error_cm": norm * 100.0,
        "roll_error_deg": angles[0],
        "pitch_error_deg": angles[1],
        "yaw_error_deg": angles[2],
        "x_error_m": float(difference[0]),
        "y_error_m": float(difference[1]),
        "z_error_m": float(difference[2]),
    }


def recovered(rotation_error_deg: float, translation_error_m: float) -> bool:
    """Say whether both errors are inside their thresholds, inclusive of the edge.

    Either threshold alone would report roughly twice the recovery rate. A
    non-finite or negative error raises instead of answering false, because
    counting a broken measurement as a failed recovery would hide the break.
    """

    if not (math.isfinite(rotation_error_deg) and math.isfinite(translation_error_m)):
        raise ValueError(
            f"errors must be finite to be judged, got {rotation_error_deg} and "
            f"{translation_error_m}"
        )
    if rotation_error_deg < 0.0 or translation_error_m < 0.0:
        raise ValueError(
            f"errors are magnitudes and cannot be negative, got {rotation_error_deg} and "
            f"{translation_error_m}"
        )
    return (
        rotation_error_deg <= RECOVERY_ROTATION_THRESHOLD_DEG
        and translation_error_m <= RECOVERY_TRANSLATION_THRESHOLD_M
    )
