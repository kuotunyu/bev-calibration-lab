"""Contracts for measuring how wrong a calibration is, and whether it counts as recovered.

The rotation error is a geodesic: the single angle of the rotation that takes the
estimate to the truth. Summing per-axis differences instead would over-count a
rotation split across axes and under-count one near a wrap, and both errors look
plausible.
"""

from __future__ import annotations

import math

import pytest

from bevcalib.artifacts.results import CalibrationFaultModel
from bevcalib.geometry.se3 import SE3
from bevcalib.perturbations.apply import fault_to_se3

IDENTITY = SE3(rotation_wxyz=(1.0, 0.0, 0.0, 0.0), translation_xyz_m=(0.0, 0.0, 0.0))


def fault(
    rotation: tuple[float, float, float] = (0.0, 0.0, 0.0),
    translation: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> CalibrationFaultModel:
    return CalibrationFaultModel(
        rotation_rpy_deg=rotation,
        translation_xyz_m=translation,
        requested_time_offset_ms=0,
    )


@pytest.mark.parametrize("axis", [(2.0, 0.0, 0.0), (0.0, 2.0, 0.0), (0.0, 0.0, 2.0)])
def test_a_two_degree_turn_is_two_degrees_whichever_axis_it_is_about(
    axis: tuple[float, float, float],
) -> None:
    """The geodesic angle is a property of the rotation, not of how it was written down."""

    from bevcalib.metrics.calibration import rotation_geodesic_error_deg

    assert rotation_geodesic_error_deg(IDENTITY, fault_to_se3(fault(rotation=axis))) == (
        pytest.approx(2.0, abs=1e-9)
    )


def test_a_rotation_split_across_two_axes_is_not_the_sum_of_its_parts() -> None:
    """One degree of roll and one of pitch is about 1.414 degrees apart, not two.

    This is the case that separates a real geodesic from a per-axis sum, and the
    per-axis sum is the version somebody writes when in a hurry.
    """

    from bevcalib.metrics.calibration import rotation_geodesic_error_deg

    error = rotation_geodesic_error_deg(
        fault_to_se3(fault(rotation=(1.0, 0.0, 0.0))),
        fault_to_se3(fault(rotation=(0.0, 1.0, 0.0))),
    )

    assert error == pytest.approx(math.sqrt(2.0), rel=1e-3)
    assert error < 2.0


def test_a_perfect_estimate_has_no_rotation_error() -> None:
    """The floor has to be reachable, and it has to be exactly zero."""

    from bevcalib.metrics.calibration import rotation_geodesic_error_deg

    turned = fault_to_se3(fault(rotation=(1.0, -0.5, 2.0)))

    assert rotation_geodesic_error_deg(turned, turned) == pytest.approx(0.0, abs=1e-9)


def test_the_rotation_error_does_not_depend_on_which_way_round_it_is_asked() -> None:
    """It is a distance between two rotations, so it must be symmetric."""

    from bevcalib.metrics.calibration import rotation_geodesic_error_deg

    left = fault_to_se3(fault(rotation=(1.0, 0.0, 0.0)))
    right = fault_to_se3(fault(rotation=(0.0, 0.0, 2.0)))

    assert rotation_geodesic_error_deg(left, right) == pytest.approx(
        rotation_geodesic_error_deg(right, left), abs=1e-9
    )


def test_the_translation_error_is_reported_as_a_norm_and_in_centimetres() -> None:
    """A three-four-five triangle, so the norm is arithmetic anyone can check."""

    from bevcalib.metrics.calibration import calibration_errors

    errors = calibration_errors(
        fault(translation=(0.03, 0.04, 0.0)), fault(translation=(0.0, 0.0, 0.0))
    )

    assert errors["translation_error_m"] == pytest.approx(0.05)
    assert errors["translation_error_cm"] == pytest.approx(5.0)


def test_the_per_axis_errors_keep_their_sign() -> None:
    """A corrector that is consistently short on one axis is a different fault than noise."""

    from bevcalib.metrics.calibration import calibration_errors

    errors = calibration_errors(
        fault(rotation=(1.0, -2.0, 0.0), translation=(0.10, -0.05, 0.0)), fault()
    )

    assert errors["roll_error_deg"] == pytest.approx(1.0)
    assert errors["pitch_error_deg"] == pytest.approx(-2.0)
    assert errors["x_error_m"] == pytest.approx(0.10)
    assert errors["y_error_m"] == pytest.approx(-0.05)


def test_the_error_table_reports_the_geodesic_alongside_the_components() -> None:
    """Both are needed: the geodesic to judge, the components to diagnose."""

    from bevcalib.metrics.calibration import calibration_errors

    errors = calibration_errors(fault(rotation=(0.0, 0.0, 2.0)), fault())

    assert errors["rotation_geodesic_error_deg"] == pytest.approx(2.0, abs=1e-9)
    assert set(errors) == {
        "rotation_geodesic_error_deg",
        "translation_error_m",
        "translation_error_cm",
        "roll_error_deg",
        "pitch_error_deg",
        "yaw_error_deg",
        "x_error_m",
        "y_error_m",
        "z_error_m",
    }


@pytest.mark.parametrize(
    ("rotation_deg", "translation_m", "expected"),
    [
        (0.0, 0.0, True),
        (0.25, 0.05, True),  # both exactly on the threshold, which is inclusive
        (0.25, 0.0500001, False),
        (0.2500001, 0.05, False),
        (0.1, 0.2, False),  # rotation fine, translation not
        (1.0, 0.01, False),  # translation fine, rotation not
    ],
)
def test_a_recovery_needs_both_thresholds_at_once(
    rotation_deg: float, translation_m: float, expected: bool
) -> None:
    """Either threshold alone would report roughly twice the recovery rate.

    A corrector that fixes the rotation and leaves the offset has not recovered
    the calibration; it has recovered half of it, and the half it left is the one
    that moves a box in bird's-eye view.
    """

    from bevcalib.metrics.calibration import recovered

    assert recovered(rotation_deg, translation_m) is expected


def test_the_recovery_thresholds_are_the_ones_the_protocol_names() -> None:
    """Stated as constants so the report and the metric cannot disagree."""

    from bevcalib.metrics.calibration import (
        RECOVERY_ROTATION_THRESHOLD_DEG,
        RECOVERY_TRANSLATION_THRESHOLD_M,
    )

    assert RECOVERY_ROTATION_THRESHOLD_DEG == 0.25
    assert RECOVERY_TRANSLATION_THRESHOLD_M == 0.05


@pytest.mark.parametrize(
    ("rotation_deg", "translation_m"),
    [(float("nan"), 0.0), (0.0, float("nan")), (float("inf"), 0.0), (-1.0, 0.0), (0.0, -1.0)],
)
def test_an_error_that_is_not_a_distance_cannot_be_judged(
    rotation_deg: float, translation_m: float
) -> None:
    """Answering False for a NaN would count a broken measurement as a failed recovery."""

    from bevcalib.metrics.calibration import recovered

    with pytest.raises(ValueError):
        recovered(rotation_deg, translation_m)


@pytest.mark.parametrize(
    "role",
    ["estimate", "truth"],
    ids=["the estimate carries the offset", "the truth carries the offset"],
)
def test_a_timing_fault_cannot_be_scored_as_a_pose_error(role: str) -> None:
    """There is no pose difference between two faults that differ only in time.

    Both arguments are checked, and the message names WHICH of them carried the
    offset. A caller comparing a recovered pose against a reference needs that:
    an offset on the estimate means the corrector was handed a timing fault it
    cannot express, while an offset on the truth means the experiment matrix
    paired the wrong reference. The message is asserted in full because the
    role name is the whole diagnosis.
    """

    from bevcalib.metrics.calibration import calibration_errors

    timed = CalibrationFaultModel(
        rotation_rpy_deg=(0.0, 0.0, 0.0),
        translation_xyz_m=(0.0, 0.0, 0.0),
        requested_time_offset_ms=50,
    )
    arguments = (timed, fault()) if role == "estimate" else (fault(), timed)

    with pytest.raises(
        ValueError,
        match=(
            rf"^the {role} carries a timing offset of 50 ms, and there is no pose "
            r"difference between two faults that differ only in time$"
        ),
    ):
        calibration_errors(*arguments)


def test_the_error_is_the_estimate_minus_the_truth_in_every_component() -> None:
    """A signed error says which way the corrector was wrong, and the sign is the point.

    A residual of +5 cm and one of -5 cm are different findings: one says the
    corrector over-corrects and the other that it under-corrects, and the study
    reports per-axis errors precisely so that bias can be seen. Adding instead
    of subtracting leaves every magnitude plausible and every sign meaningless,
    and the summed magnitude would still look like a small number.

    Each component is asymmetric here, so no accidental cancellation can hide a
    swapped operand: estimate minus truth is `(0.5, -0.25, 0.75)` in rotation
    and `(0.03, -0.01, 0.05)` in translation, and none of those equals the sum.
    """

    from bevcalib.metrics.calibration import calibration_errors

    estimate = CalibrationFaultModel(
        rotation_rpy_deg=(1.5, 0.25, 1.25),
        translation_xyz_m=(0.08, 0.01, 0.09),
        requested_time_offset_ms=0,
    )
    truth = CalibrationFaultModel(
        rotation_rpy_deg=(1.0, 0.5, 0.5),
        translation_xyz_m=(0.05, 0.02, 0.04),
        requested_time_offset_ms=0,
    )

    errors = calibration_errors(estimate, truth)

    assert errors["roll_error_deg"] == pytest.approx(0.5)
    assert errors["pitch_error_deg"] == pytest.approx(-0.25)
    assert errors["yaw_error_deg"] == pytest.approx(0.75)
    assert errors["x_error_m"] == pytest.approx(0.03)
    assert errors["y_error_m"] == pytest.approx(-0.01)
    assert errors["z_error_m"] == pytest.approx(0.05)
    assert errors["translation_error_m"] == pytest.approx(math.sqrt(0.03**2 + 0.01**2 + 0.05**2))
    assert errors["translation_error_cm"] == pytest.approx(100.0 * errors["translation_error_m"])
