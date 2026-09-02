"""Summarising reprojection error, and saying plainly what was left out."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np

from bevcalib.artifacts.results import CalibrationResultV1
from bevcalib.geometry.projection import BoolArray, Float64Array

RangeBin = Literal["0-10", "10-20", "20-40", "40-80", "80+"]

RANGE_BINS: tuple[RangeBin, ...] = ("0-10", "10-20", "20-40", "40-80", "80+")
# Half-open on the right, everywhere. A box at exactly 40 m belongs to one bin and
# only one; landing in two, or in none, shows up as a strange dip in a chart and
# gets explained away as physics.
_UPPER_EDGES: tuple[tuple[float, RangeBin], ...] = (
    (10.0, "0-10"),
    (20.0, "10-20"),
    (40.0, "20-40"),
    (80.0, "40-80"),
)


@dataclass(frozen=True)
class ValiditySummary:
    """How many samples were measurable, how many were not, and why not.

    This travels with every aggregate. A rate computed over an unknown denominator
    is not a rate, and the reasons are what turn "thirty per cent were dropped"
    into something anyone can act on.
    """

    valid: int
    invalid: int
    reasons: tuple[tuple[str, int], ...]


def range_bin(range_m: float) -> RangeBin:
    """Place a range into one of the five reported bins."""

    if not math.isfinite(range_m) or range_m < 0.0:
        raise ValueError(f"a range must be a non-negative finite distance, got {range_m}")
    for edge, name in _UPPER_EDGES:
        if range_m < edge:
            return name
    return "80+"


def pixel_error_percentiles(
    errors_px: Float64Array, valid: BoolArray | None = None
) -> tuple[float | None, float | None]:
    """Return the median and ninetieth percentile of the valid errors.

    Both are `None` when nothing is measurable, rather than zero. Zero pixels of
    error is the best possible score, so returning it for an empty set would make
    the worst outcome indistinguishable from the best.
    """

    errors = np.asarray(errors_px, dtype=np.float64)
    if errors.ndim != 1:
        raise ValueError(f"pixel errors must be a flat array, got shape {errors.shape}")

    if valid is None:
        selected = errors
    else:
        mask = np.asarray(valid, dtype=bool)
        if mask.shape != errors.shape:
            raise ValueError(
                f"the validity mask {mask.shape} must have the same shape as the errors "
                f"{errors.shape}"
            )
        selected = errors[mask]

    if not np.all(np.isfinite(selected)):
        raise ValueError("pixel errors must be finite to be summarised")
    if selected.size == 0:
        return (None, None)
    return (float(np.median(selected)), float(np.quantile(selected, 0.9)))


def summarize_validity(results: Sequence[CalibrationResultV1]) -> ValiditySummary:
    """Count the measurable and unmeasurable results, worst reason first."""

    counter: Counter[str] = Counter()
    valid = 0
    for item in results:
        if item.valid:
            valid += 1
        else:
            # The result model refuses an invalid row without a non-empty reason,
            # so this is always a real string.
            counter[str(item.invalid_reason)] += 1

    return ValiditySummary(
        valid=valid,
        invalid=sum(counter.values()),
        reasons=tuple(sorted(counter.items(), key=lambda pair: (-pair[1], pair[0]))),
    )
