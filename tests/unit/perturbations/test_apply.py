"""Contracts for how a metadata fault perturbs a transform.

There are two ways to compose a fault with a true transform and both produce a
valid rigid transform, so the choice cannot be left implicit. The convention is
fixed here and stated in the coordinate contract:

    assumed = true ∘ fault

The fault is applied on the source side, which is the sensor's own frame. That is
what a miscalibrated extrinsic physically is: you believe the sensor sits a little
rotated or shifted from where it really does, measured in its own axes.
"""

from __future__ import annotations

import hashlib
import math

import numpy as np
import pytest

from bevcalib.artifacts.results import CalibrationFaultModel
from bevcalib.geometry.quaternions import quaternion_to_matrix
from bevcalib.geometry.se3 import SE3, compose, transform_points

TRUE = SE3(
    rotation_wxyz=(math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5)),
    translation_xyz_m=(1.7, 0.0, 1.5),
)


def fault(
    rotation: tuple[float, float, float] = (0.0, 0.0, 0.0),
    translation: tuple[float, float, float] = (0.0, 0.0, 0.0),
    time_ms: int = 0,
) -> CalibrationFaultModel:
    return CalibrationFaultModel(
        rotation_rpy_deg=rotation,
        translation_xyz_m=translation,
        requested_time_offset_ms=time_ms,
    )


def test_a_zero_fault_leaves_the_transform_exactly_as_it_was() -> None:
    """The baseline condition of the study, and the seven zero entries of the schedule."""

    from bevcalib.perturbations.apply import apply_metadata_fault

    assumed = apply_metadata_fault(TRUE, fault())

    assert assumed.rotation_wxyz == pytest.approx(TRUE.rotation_wxyz, abs=1e-12)
    assert assumed.translation_xyz_m == pytest.approx(TRUE.translation_xyz_m, abs=1e-12)


def test_the_fault_is_composed_on_the_source_side() -> None:
    """`assumed = true ∘ fault`, not `fault ∘ true`; both are rigid and only one is meant."""

    from bevcalib.perturbations.apply import apply_metadata_fault, fault_to_se3

    injected = fault(rotation=(0.0, 0.0, 2.0), translation=(0.2, 0.0, 0.0))

    assumed = apply_metadata_fault(TRUE, injected)
    expected = compose(TRUE, fault_to_se3(injected))

    assert assumed.rotation_wxyz == pytest.approx(expected.rotation_wxyz, abs=1e-12)
    assert assumed.translation_xyz_m == pytest.approx(expected.translation_xyz_m, abs=1e-12)


def test_composing_on_the_other_side_would_give_a_different_answer() -> None:
    """Proof the convention test can fail, rather than holding for a symmetric example."""

    from bevcalib.perturbations.apply import apply_metadata_fault, fault_to_se3

    injected = fault(rotation=(0.0, 0.0, 2.0), translation=(0.2, 0.0, 0.0))

    source_side = apply_metadata_fault(TRUE, injected)
    target_side = compose(fault_to_se3(injected), TRUE)

    assert source_side.translation_xyz_m != pytest.approx(target_side.translation_xyz_m, abs=1e-6)


def test_a_pure_translation_fault_moves_the_sensor_along_its_own_axes() -> None:
    """Source-side composition means the shift is rotated by the true rotation."""

    from bevcalib.perturbations.apply import apply_metadata_fault

    # TRUE is a quarter turn about z, so 0.2 m along the sensor x appears along global y.
    assumed = apply_metadata_fault(TRUE, fault(translation=(0.2, 0.0, 0.0)))

    assert assumed.translation_xyz_m == pytest.approx((1.7, 0.2, 1.5), abs=1e-12)


@pytest.mark.parametrize(
    ("axis_index", "expected_axis"),
    [(0, (1.0, 0.0, 0.0)), (1, (0.0, 1.0, 0.0)), (2, (0.0, 0.0, 1.0))],
)
def test_roll_pitch_and_yaw_turn_about_x_then_y_then_z(
    axis_index: int, expected_axis: tuple[float, float, float]
) -> None:
    """The Euler convention is fixed: R = Rz(yaw) · Ry(pitch) · Rx(roll)."""

    from bevcalib.perturbations.apply import fault_to_se3

    angle_deg = 30.0
    rotation = [0.0, 0.0, 0.0]
    rotation[axis_index] = angle_deg

    matrix = quaternion_to_matrix(
        fault_to_se3(fault(rotation=(rotation[0], rotation[1], rotation[2]))).rotation_wxyz
    )

    # A rotation about an axis leaves that axis fixed and turns by the right angle.
    np.testing.assert_allclose(matrix @ np.array(expected_axis), expected_axis, atol=1e-12)
    assert math.degrees(math.acos((np.trace(matrix) - 1.0) / 2.0)) == pytest.approx(angle_deg)


def test_the_three_euler_angles_compose_in_the_stated_order() -> None:
    """With all three nonzero the order is what distinguishes one convention from another."""

    from bevcalib.perturbations.apply import fault_to_se3

    def about(axis: int, degrees: float) -> np.ndarray:
        angle = math.radians(degrees)
        cos, sin = math.cos(angle), math.sin(angle)
        if axis == 0:
            return np.array([[1, 0, 0], [0, cos, -sin], [0, sin, cos]], dtype=float)
        if axis == 1:
            return np.array([[cos, 0, sin], [0, 1, 0], [-sin, 0, cos]], dtype=float)
        return np.array([[cos, -sin, 0], [sin, cos, 0], [0, 0, 1]], dtype=float)

    built = quaternion_to_matrix(fault_to_se3(fault(rotation=(10.0, 20.0, 30.0))).rotation_wxyz)
    expected = about(2, 30.0) @ about(1, 20.0) @ about(0, 10.0)

    np.testing.assert_allclose(built, expected, atol=1e-12)
    assert not np.allclose(built, about(0, 10.0) @ about(1, 20.0) @ about(2, 30.0))


def test_opposite_signs_give_opposite_rotations() -> None:
    """A sweep from -2 to +2 degrees only means something if the sign reaches the transform."""

    from bevcalib.perturbations.apply import apply_metadata_fault, fault_to_se3

    positive = fault_to_se3(fault(rotation=(0.0, 0.0, 1.0)))
    negative = fault_to_se3(fault(rotation=(0.0, 0.0, -1.0)))

    product = quaternion_to_matrix(positive.rotation_wxyz) @ quaternion_to_matrix(
        negative.rotation_wxyz
    )
    np.testing.assert_allclose(product, np.eye(3), atol=1e-12)
    assert apply_metadata_fault(
        TRUE, fault(rotation=(0.0, 0.0, 1.0))
    ).rotation_wxyz != pytest.approx(
        apply_metadata_fault(TRUE, fault(rotation=(0.0, 0.0, -1.0))).rotation_wxyz, abs=1e-6
    )


def test_a_timing_fault_alone_does_not_move_the_transform() -> None:
    """Timing perturbs which frame is paired, never the geometry of the pairing."""

    from bevcalib.perturbations.apply import apply_metadata_fault

    assumed = apply_metadata_fault(TRUE, fault(time_ms=200))

    assert assumed.rotation_wxyz == pytest.approx(TRUE.rotation_wxyz, abs=1e-12)
    assert assumed.translation_xyz_m == pytest.approx(TRUE.translation_xyz_m, abs=1e-12)


def test_a_formal_fault_changes_no_observation_byte() -> None:
    """The study perturbs metadata only, so the pixels and the returns must be untouched.

    If a fault altered the observations, the measured error would mix a sensing
    change with a calibration change and neither could be attributed.
    """

    from bevcalib.perturbations.apply import apply_metadata_fault

    points = np.arange(30, dtype=np.float64).reshape(6, 5)
    image_bytes = bytes(range(256))
    before = (hashlib.sha256(points.tobytes()).hexdigest(), hashlib.sha256(image_bytes).hexdigest())

    apply_metadata_fault(TRUE, fault(rotation=(2.0, -1.0, 0.5), translation=(0.2, -0.1, 0.05)))

    after = (hashlib.sha256(points.tobytes()).hexdigest(), hashlib.sha256(image_bytes).hexdigest())
    assert before == after


def test_the_perturbed_transform_is_still_rigid() -> None:
    """A fault that stretched space would not be a calibration error at all."""

    from bevcalib.perturbations.apply import apply_metadata_fault

    assumed = apply_metadata_fault(
        TRUE, fault(rotation=(2.0, -2.0, 2.0), translation=(0.2, 0.2, -0.2))
    )
    matrix = quaternion_to_matrix(assumed.rotation_wxyz)

    np.testing.assert_allclose(matrix @ matrix.T, np.eye(3), atol=1e-12)
    probe = np.array([[3.0, -4.0, 12.0]])
    moved = transform_points(assumed, probe) - np.asarray(assumed.translation_xyz_m)
    assert float(np.linalg.norm(moved)) == pytest.approx(13.0, abs=1e-9)
