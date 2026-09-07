"""Timing faults: select LiDAR relative to a fixed camera exposure."""

from __future__ import annotations

from bevcalib.artifacts.results import CalibrationFaultModel
from bevcalib.nuscenes_adapter.frames import SensorPacket
from bevcalib.nuscenes_adapter.sweeps import TimingSelection, choose_nearest_sweep

from .apply import inverse_fault
from .schedule import TIMING_MAX_SELECTION_ERROR_MS

MICROSECONDS_PER_MILLISECOND = 1000


def timing_target_timestamp_us(reference_timestamp_us: int, requested_offset_ms: int) -> int:
    """Convert a requested offset in milliseconds into an absolute microsecond stamp.

    The protocol speaks in milliseconds and the data in microseconds. The
    conversion happens here, once, so no other function has to remember which
    unit it is holding.
    """

    return reference_timestamp_us + requested_offset_ms * MICROSECONDS_PER_MILLISECOND


def select_lidar_for_timing_fault(
    sweeps: tuple[SensorPacket, ...],
    camera_timestamp_us: int,
    fault: CalibrationFaultModel,
) -> TimingSelection:
    """Serve the request using available LiDAR; the camera exposure stays fixed.

    A request of 100 ms is served by whatever frame is nearest to that instant,
    and the gap between the two is recorded rather than assumed away. Beyond the
    tolerance the selection is still returned and marked invalid, because which
    sweep was nearest is evidence about the data. The installation service must
    first filter candidates by LIDAR_TOP channel, scene/log and actual payload.
    """

    if any(sweep.calibrated_sensor.source != "lidar_sensor" for sweep in sweeps):
        raise ValueError("timing candidates must be LiDAR readings")
    if not sweeps:
        return TimingSelection(
            requested_offset_ms=fault.requested_time_offset_ms,
            realized_offset_ms=None,
            selected_sample_data_token=None,
            absolute_error_ms=None,
            valid=False,
            selected_timestamp_us=None,
            reason="no_available_lidar",
        )
    return choose_nearest_sweep(
        sweeps,
        timing_target_timestamp_us(camera_timestamp_us, fault.requested_time_offset_ms),
        TIMING_MAX_SELECTION_ERROR_MS,
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
    correction = inverse_fault(fault)
    return correction.rotation_rpy_deg, correction.translation_xyz_m
