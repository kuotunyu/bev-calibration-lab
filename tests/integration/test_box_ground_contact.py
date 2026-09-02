"""Calibration faults driven through the ground-contact operator, end to end.

The unit tests pin the arithmetic. This one asks the question the study asks: does
a controlled metadata fault produce a bird's-eye-view error that grows with the
fault, and is it the size one would expect from the geometry? If the answer were
no, every number the study later reports would be measuring something else.
"""

from __future__ import annotations

import hashlib
import math

import numpy as np
import pytest

from bevcalib.artifacts.results import CalibrationFaultModel
from bevcalib.geometry.quaternions import matrix_to_quaternion
from bevcalib.geometry.se3 import SE3
from bevcalib.operators.ground_contact import observe_ground_contact
from bevcalib.perturbations.apply import apply_metadata_fault

INTRINSIC = np.array([[800.0, 0.0, 320.0], [0.0, 800.0, 240.0], [0.0, 0.0, 1.0]])
IMAGE_SIZE = (640, 480)
CAMERA_FROM_GLOBAL = SE3(
    rotation_wxyz=matrix_to_quaternion(
        np.array([[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]])
    ),
    translation_xyz_m=(0.0, 1.5, 0.0),
)
# Three cars ahead of the camera, at ranges the study actually reports on.
BOXES = (
    ("box-near", np.array([8.0, -1.5, 0.75])),
    ("box-mid", np.array([20.0, 1.0, 0.75])),
    ("box-far", np.array([45.0, -4.0, 0.75])),
)


def fault(
    rotation: tuple[float, float, float] = (0.0, 0.0, 0.0),
    translation: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> CalibrationFaultModel:
    return CalibrationFaultModel(
        rotation_rpy_deg=rotation,
        translation_xyz_m=translation,
        requested_time_offset_ms=0,
    )


def bev_errors(injected: CalibrationFaultModel) -> list[float]:
    """The bird's-eye-view error each box picks up under one fault."""

    assumed = apply_metadata_fault(CAMERA_FROM_GLOBAL, injected)
    errors: list[float] = []
    for token, centre in BOXES:
        observation = observe_ground_contact(
            box_token=token,
            box_center_global=centre,
            size_wlh=(1.8, 4.2, 1.5),
            orientation_wxyz=(1.0, 0.0, 0.0, 0.0),
            true_camera_from_global=CAMERA_FROM_GLOBAL,
            assumed_camera_from_global=assumed,
            intrinsic=INTRINSIC,
            image_size_wh=IMAGE_SIZE,
            ground_z_global=0.0,
        )
        assert observation.valid, f"{token} must be observable for this test to mean anything"
        errors.append(
            float(
                np.linalg.norm(
                    np.array(observation.assumed_ground_xy_m)
                    - np.array(observation.oracle_ground_xy_m)
                )
            )
        )
    return errors


def test_a_true_calibration_puts_every_box_exactly_where_it_is() -> None:
    """Zero fault, zero error. Any bias here would be added to every reported number."""

    assert bev_errors(fault()) == pytest.approx([0.0, 0.0, 0.0], abs=1e-9)


@pytest.mark.parametrize("yaw_deg", [0.1, 0.5, 2.0])
def test_a_rotation_fault_moves_every_box_in_the_ground_plane(yaw_deg: float) -> None:
    """A fraction of a degree at the camera is centimetres to metres on the ground."""

    assert all(error > 0.0 for error in bev_errors(fault(rotation=(0.0, 0.0, yaw_deg))))


def test_the_error_grows_with_the_size_of_the_rotation_fault() -> None:
    """Monotonicity is what lets the study report a dose-response curve at all."""

    small = bev_errors(fault(rotation=(0.0, 0.0, 0.1)))
    medium = bev_errors(fault(rotation=(0.0, 0.0, 0.5)))
    large = bev_errors(fault(rotation=(0.0, 0.0, 2.0)))

    for near, mid, far in zip(small, medium, large, strict=True):
        assert near < mid < far


def test_a_rotation_fault_hurts_distant_boxes_far_more_than_near_ones() -> None:
    """The lever arm is the range, which is why the study reports error against range.

    A fixed angular error at the camera becomes a ground displacement roughly
    proportional to distance, so a tenth of a degree that is harmless at eight
    metres is not harmless at forty-five.
    """

    near, mid, far = bev_errors(fault(rotation=(0.0, 0.0, 0.5)))

    assert near < mid < far
    # Roughly linear in range: 45 m is about five and a half times 8 m.
    assert far / near == pytest.approx(45.0 / 8.0, rel=0.25)


def test_a_pure_yaw_fault_of_one_degree_moves_a_forty_metre_box_by_about_its_arc() -> None:
    """An order-of-magnitude check against geometry done on paper: error ≈ range · θ."""

    observation_error = bev_errors(fault(rotation=(0.0, 0.0, 1.0)))[2]
    range_m = float(np.linalg.norm(BOXES[2][1][:2]))

    assert observation_error == pytest.approx(range_m * math.radians(1.0), rel=0.15)


def test_a_translation_fault_shifts_every_box_by_the_same_amount() -> None:
    """Unlike a rotation, a sideways shift is range-independent, which is a real distinction."""

    errors = bev_errors(fault(translation=(0.0, 0.05, 0.0)))

    assert errors == pytest.approx([0.05, 0.05, 0.05], abs=1e-6)


def test_the_boxes_themselves_are_never_touched_by_a_fault() -> None:
    """Metadata only, still true once the operator and the fault are used together."""

    before = [hashlib.sha256(centre.tobytes()).hexdigest() for _, centre in BOXES]

    bev_errors(fault(rotation=(2.0, -1.0, 0.5), translation=(0.2, -0.1, 0.05)))

    assert [hashlib.sha256(centre.tobytes()).hexdigest() for _, centre in BOXES] == before
