"""Contracts for choosing the sweep that realises a requested timing offset.

A timing fault asks for the LiDAR sweep a fixed number of milliseconds away from
the fixed camera. No such frame usually exists: the LiDAR runs at its own rate, so
the request is served by the nearest actual sweep and the difference between what
was asked for and what was obtained has to be recorded rather than assumed away.
"""

from __future__ import annotations

from typing import Any

import pytest

from bevcalib.geometry.frames import FramedTransform
from bevcalib.geometry.se3 import SE3

IDENTITY = SE3(rotation_wxyz=(1.0, 0.0, 0.0, 0.0), translation_xyz_m=(0.0, 0.0, 0.0))
REFERENCE_US = 1_538_984_233_547_259


def packet(token: str, timestamp_us: int) -> Any:
    from bevcalib.nuscenes_adapter.frames import SensorPacket

    return SensorPacket(
        sample_token="sample-0",
        sample_data_token=token,
        timestamp_us=timestamp_us,
        calibrated_sensor=FramedTransform(
            target="lidar_ego", source="lidar_sensor", value=IDENTITY
        ),
        ego_pose=FramedTransform(target="global", source="lidar_ego", value=IDENTITY),
        file_relative_path=f"samples/LIDAR_TOP/{token}.bin",
    )


def sweeps_at(*offsets_ms: float) -> tuple[Any, ...]:
    """Build one LiDAR sweep per offset in milliseconds from the camera reference."""

    return tuple(
        packet(f"lidar-{index}", REFERENCE_US + round(offset * 1000))
        for index, offset in enumerate(offsets_ms)
    )


def test_the_sweep_closest_to_the_requested_time_is_chosen() -> None:
    """The base case: 50 ms was asked for, and 48 ms is nearer than 60 ms."""

    from bevcalib.nuscenes_adapter.sweeps import choose_nearest_sweep

    selection = choose_nearest_sweep(
        sweeps_at(0.0, 48.0, 60.0), REFERENCE_US + 50_000, requested_offset_ms=50
    )

    assert selection.selected_sample_data_token == "lidar-1"
    assert selection.valid


def test_the_realised_offset_is_measured_from_the_reference_not_the_request() -> None:
    """ "Realised offset" is the physical gap between the two sensors, which is what matters."""

    from bevcalib.nuscenes_adapter.sweeps import choose_nearest_sweep

    selection = choose_nearest_sweep(sweeps_at(48.0), REFERENCE_US + 50_000, requested_offset_ms=50)

    assert selection.requested_offset_ms == 50
    assert selection.realized_offset_ms == pytest.approx(48.0)
    assert selection.absolute_error_ms == pytest.approx(2.0)


def test_a_negative_request_keeps_its_sign_in_the_realised_offset() -> None:
    """A LiDAR sweep before the camera is a real fault, and a sign flip would hide it."""

    from bevcalib.nuscenes_adapter.sweeps import choose_nearest_sweep

    selection = choose_nearest_sweep(
        sweeps_at(-98.5, 0.0), REFERENCE_US - 100_000, requested_offset_ms=-100
    )

    assert selection.realized_offset_ms == pytest.approx(-98.5)
    assert selection.absolute_error_ms == pytest.approx(1.5)
    assert selection.valid


def test_a_tie_is_broken_by_token_so_the_choice_is_reproducible() -> None:
    """Two sweeps equidistant from the request must not be resolved by table order."""

    from bevcalib.nuscenes_adapter.sweeps import choose_nearest_sweep

    early = packet("lidar-zulu", REFERENCE_US + 40_000)
    late = packet("lidar-alpha", REFERENCE_US + 60_000)

    forward = choose_nearest_sweep((early, late), REFERENCE_US + 50_000, requested_offset_ms=50)
    backward = choose_nearest_sweep((late, early), REFERENCE_US + 50_000, requested_offset_ms=50)

    assert forward.selected_sample_data_token == "lidar-alpha"
    assert backward.selected_sample_data_token == "lidar-alpha"


@pytest.mark.parametrize(
    ("error_ms", "expected"),
    [(24.999, True), (25.0, True), (25.001, False), (200.0, False)],
)
def test_the_twenty_five_millisecond_limit_includes_its_own_boundary(
    error_ms: float, expected: bool
) -> None:
    """Inclusive at exactly 25 ms, stated once here so no caller has to guess."""

    from bevcalib.nuscenes_adapter.sweeps import choose_nearest_sweep

    selection = choose_nearest_sweep(
        sweeps_at(50.0 + error_ms), REFERENCE_US + 50_000, requested_offset_ms=50
    )

    assert selection.valid is expected
    assert selection.absolute_error_ms == pytest.approx(error_ms, abs=1e-6)


def test_an_unmet_request_still_reports_which_sweep_was_nearest() -> None:
    """An invalid selection is evidence about the data, not a reason to return nothing."""

    from bevcalib.nuscenes_adapter.sweeps import choose_nearest_sweep

    selection = choose_nearest_sweep(
        sweeps_at(0.0), REFERENCE_US + 200_000, requested_offset_ms=200
    )

    assert not selection.valid
    assert selection.selected_sample_data_token == "lidar-0"
    assert selection.realized_offset_ms == pytest.approx(0.0)


def test_a_custom_tolerance_is_honoured() -> None:
    """The 25 ms figure is a protocol constant, but the function must not hard-code it."""

    from bevcalib.nuscenes_adapter.sweeps import choose_nearest_sweep

    selection = choose_nearest_sweep(
        sweeps_at(80.0), REFERENCE_US + 50_000, max_error_ms=40.0, requested_offset_ms=50
    )

    assert selection.valid


def test_asking_with_no_sweeps_available_fails_closed() -> None:
    """There is no nearest element of an empty set, and inventing one would be worse."""

    from bevcalib.nuscenes_adapter.sweeps import choose_nearest_sweep

    with pytest.raises(ValueError, match=r"^cannot choose a sweep: no sweeps were offered$"):
        choose_nearest_sweep((), REFERENCE_US, requested_offset_ms=0)


def test_the_default_request_is_no_offset_at_all() -> None:
    """The common case is the keyframe itself, and it should need no extra argument."""

    from bevcalib.nuscenes_adapter.sweeps import choose_nearest_sweep

    selection = choose_nearest_sweep(sweeps_at(0.0, 100.0), REFERENCE_US)

    assert selection.requested_offset_ms == 0
    assert selection.realized_offset_ms == pytest.approx(0.0)
    assert selection.selected_sample_data_token == "lidar-0"


def test_the_selection_cannot_be_edited_after_the_fact() -> None:
    """It is provenance for a timing fault, so it is frozen like every other record."""

    import dataclasses

    from bevcalib.nuscenes_adapter.sweeps import choose_nearest_sweep

    selection = choose_nearest_sweep(sweeps_at(0.0), REFERENCE_US)

    with pytest.raises(dataclasses.FrozenInstanceError):
        selection.valid = False  # type: ignore[misc]
