"""Contracts for timing faults: which camera frame a requested offset actually gets.

A timing fault is the one perturbation that changes nothing about the geometry.
It changes which camera frame is paired with the sweep, and the pairing is served
by whatever frame really exists, so what was asked for and what was obtained are
recorded separately.
"""

from __future__ import annotations

import pytest

from bevcalib.artifacts.results import CalibrationFaultModel
from bevcalib.geometry.frames import FramedTransform
from bevcalib.geometry.se3 import SE3
from bevcalib.nuscenes_adapter.frames import SensorPacket

IDENTITY = SE3(rotation_wxyz=(1.0, 0.0, 0.0, 0.0), translation_xyz_m=(0.0, 0.0, 0.0))
REFERENCE_US = 1_538_984_233_547_259


def sweep(token: str, offset_ms: float) -> SensorPacket:
    return SensorPacket(
        sample_token="sample-0",
        sample_data_token=token,
        timestamp_us=REFERENCE_US + round(offset_ms * 1000),
        calibrated_sensor=FramedTransform(
            target="camera_ego", source="camera_sensor", value=IDENTITY
        ),
        ego_pose=FramedTransform(target="global", source="camera_ego", value=IDENTITY),
        file_relative_path=f"samples/CAM_FRONT/{token}.jpg",
    )


def fault(
    time_ms: int, rotation: tuple[float, float, float] = (0.0, 0.0, 0.0)
) -> CalibrationFaultModel:
    return CalibrationFaultModel(
        rotation_rpy_deg=rotation,
        translation_xyz_m=(0.0, 0.0, 0.0),
        requested_time_offset_ms=time_ms,
    )


def test_the_requested_offset_becomes_a_timestamp_in_microseconds() -> None:
    """Milliseconds in the protocol, microseconds in the data; the conversion happens once."""

    from bevcalib.perturbations.timing import timing_target_timestamp_us

    assert timing_target_timestamp_us(REFERENCE_US, 50) == REFERENCE_US + 50_000
    assert timing_target_timestamp_us(REFERENCE_US, -200) == REFERENCE_US - 200_000
    assert timing_target_timestamp_us(REFERENCE_US, 0) == REFERENCE_US


def test_a_timing_fault_selects_the_camera_frame_nearest_the_request() -> None:
    """The whole point: a 100 ms request is served by whatever frame is actually there."""

    from bevcalib.perturbations.timing import select_camera_for_timing_fault

    selection = select_camera_for_timing_fault(
        (sweep("cam-0", 0.0), sweep("cam-1", 95.0), sweep("cam-2", 150.0)),
        REFERENCE_US,
        fault(time_ms=100),
    )

    assert selection.selected_sample_data_token == "cam-1"
    assert selection.requested_offset_ms == 100
    assert selection.realized_offset_ms == pytest.approx(95.0)
    assert selection.valid


def test_a_request_no_frame_can_serve_is_recorded_as_unmet() -> None:
    """Past 25 ms the pairing is not the one that was asked for, and saying so is the point."""

    from bevcalib.perturbations.timing import select_camera_for_timing_fault

    selection = select_camera_for_timing_fault(
        (sweep("cam-0", 0.0),), REFERENCE_US, fault(time_ms=200)
    )

    assert not selection.valid
    assert selection.absolute_error_ms == pytest.approx(200.0)
    assert selection.selected_sample_data_token == "cam-0"


@pytest.mark.parametrize(("error_ms", "expected"), [(25.0, True), (25.001, False)])
def test_the_selection_limit_is_the_protocol_constant(error_ms: float, expected: bool) -> None:
    """25 ms inclusive, taken from the perturbation matrix rather than retyped here."""

    from bevcalib.perturbations.timing import select_camera_for_timing_fault

    selection = select_camera_for_timing_fault(
        (sweep("cam-0", 50.0 + error_ms),), REFERENCE_US, fault(time_ms=50)
    )

    assert selection.valid is expected


def test_the_learned_target_is_the_six_degrees_of_freedom_and_nothing_else() -> None:
    """A corrector predicts a pose, so the target is exactly the rotation and translation."""

    from bevcalib.perturbations.timing import learned_six_dof_target

    injected = CalibrationFaultModel(
        rotation_rpy_deg=(1.0, -0.5, 0.25),
        translation_xyz_m=(0.1, 0.0, -0.05),
        requested_time_offset_ms=0,
    )

    assert learned_six_dof_target(injected) == ((1.0, -0.5, 0.25), (0.1, 0.0, -0.05))


def test_a_fault_carrying_a_timing_offset_cannot_be_a_learned_target() -> None:
    """Timing is a stress condition. No 6DoF pose can express it, so training on it is a lie.

    A corrector handed a timing fault would be scored on how well it predicted a
    rotation and translation that do not encode the error it was shown, and it
    would learn to predict the mean of everything.
    """

    from bevcalib.perturbations.timing import learned_six_dof_target

    with pytest.raises(
        ValueError,
        match=(
            r"^a timing fault cannot be a learned 6DoF target: "
            r"requested_time_offset_ms is \d+, not 0$"
        ),
    ):
        learned_six_dof_target(fault(time_ms=50, rotation=(1.0, 0.0, 0.0)))


def test_asking_with_no_camera_frames_fails_closed() -> None:
    """A sample with no camera sweeps cannot serve any timing request, met or unmet."""

    from bevcalib.perturbations.timing import select_camera_for_timing_fault

    with pytest.raises(ValueError, match=r"^cannot choose a sweep: no sweeps were offered$"):
        select_camera_for_timing_fault((), REFERENCE_US, fault(time_ms=0))
