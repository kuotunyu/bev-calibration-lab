"""Timing faults: which camera frame a requested offset actually gets."""

from __future__ import annotations

from bevcalib.artifacts.results import CalibrationFaultModel
from bevcalib.nuscenes_adapter.frames import SensorPacket
from bevcalib.nuscenes_adapter.sweeps import TimingSelection, choose_nearest_sweep

from .schedule import TIMING_MAX_SELECTION_ERROR_MS

MICROSECONDS_PER_MILLISECOND = 1000


def timing_target_timestamp_us(reference_timestamp_us: int, requested_offset_ms: int) -> int:
    """Convert a requested offset in milliseconds into an absolute microsecond stamp.

    The protocol speaks in milliseconds and the data in microseconds. The
    conversion happens here, once, so no other function has to remember which
    unit it is holding.
    """

    return reference_timestamp_us + requested_offset_ms * MICROSECONDS_PER_MILLISECOND


def select_camera_for_timing_fault(
    sweeps: tuple[SensorPacket, ...],
    lidar_timestamp_us: int,
    fault: CalibrationFaultModel,
    max_error_ms: float = TIMING_MAX_SELECTION_ERROR_MS,
) -> TimingSelection:
    """Serve a fault's timing request from the camera frames that actually exist.

    A request of 100 ms is served by whatever frame is nearest to that instant,
    and the gap between the two is recorded rather than assumed away. Beyond the
    tolerance the selection is still returned and marked invalid, because which
    frame was nearest is evidence about the data.
    """

    return choose_nearest_sweep(
        sweeps,
        timing_target_timestamp_us(lidar_timestamp_us, fault.requested_time_offset_ms),
        max_error_ms,
        requested_offset_ms=fault.requested_time_offset_ms,
    )


def learned_six_dof_target(
    fault: CalibrationFaultModel,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Return the rotation and translation a learned corrector must predict.

    A timing fault is refused rather than silently reduced to its zero pose. No
    6DoF pose expresses a timing error, so a corrector shown one would be scored
    against a target that does not encode what it was shown, and would learn to
    predict the mean of everything.
    """

    if fault.requested_time_offset_ms != 0:
        raise ValueError(
            "a timing fault cannot be a learned 6DoF target: "
            f"requested_time_offset_ms is {fault.requested_time_offset_ms}, not 0"
        )
    return fault.rotation_rpy_deg, fault.translation_xyz_m
