"""Choose the sweep that realises a requested timing offset, and record the gap."""

from __future__ import annotations

from dataclasses import dataclass

from .frames import SensorPacket

MICROSECONDS_PER_MILLISECOND = 1000
# The protocol's default tolerance. A selection outside it is still returned, and
# still marked invalid, because which sweep was nearest is evidence about the data.
DEFAULT_MAX_ERROR_MS = 25.0


@dataclass(frozen=True)
class TimingSelection:
    """What was asked for, what was available, and how far apart the two were."""

    requested_offset_ms: int
    realized_offset_ms: float | None
    selected_sample_data_token: str | None
    absolute_error_ms: float | None
    valid: bool
    selected_timestamp_us: int | None
    reason: str


def choose_nearest_sweep(
    sweeps: tuple[SensorPacket, ...],
    target_timestamp_us: int,
    max_error_ms: float = DEFAULT_MAX_ERROR_MS,
    *,
    requested_offset_ms: int = 0,
) -> TimingSelection:
    """Pick the sweep nearest `target_timestamp_us` and report the timing actually achieved.

    This is a sensor-agnostic nearest selector. The timing service supplies only
    available LIDAR_TOP packets from the fixed camera's scene and log.

    `realized_offset_ms` is measured from the camera reference, not from the
    request, because it is the physical gap between the two sensors and that is
    what a result depends on. The reference is recovered from the request:
    `target = reference + requested_offset`.

    Ties are broken by token so that two runs over the same data, in whatever
    order the tables happen to yield, select the same sweep.
    """

    if not sweeps:
        raise ValueError("cannot choose a sweep: no sweeps were offered")

    chosen = min(
        sweeps,
        key=lambda sweep: (
            abs(sweep.timestamp_us - target_timestamp_us),
            sweep.sample_data_token,
        ),
    )
    reference_us = target_timestamp_us - requested_offset_ms * MICROSECONDS_PER_MILLISECOND
    absolute_error_ms = (
        abs(chosen.timestamp_us - target_timestamp_us) / MICROSECONDS_PER_MILLISECOND
    )
    return TimingSelection(
        requested_offset_ms=requested_offset_ms,
        realized_offset_ms=(chosen.timestamp_us - reference_us) / MICROSECONDS_PER_MILLISECOND,
        selected_sample_data_token=chosen.sample_data_token,
        absolute_error_ms=absolute_error_ms,
        valid=absolute_error_ms <= max_error_ms,
        selected_timestamp_us=chosen.timestamp_us,
        reason="valid" if absolute_error_ms <= max_error_ms else "outside_tolerance",
    )
