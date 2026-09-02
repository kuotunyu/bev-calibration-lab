"""A deterministic coarse-to-fine coordinate search over the six calibration degrees."""

from __future__ import annotations

import math
from typing import Protocol

import numpy as np

from bevcalib.geometry.quaternions import Float64Array

from .identity import CorrectionEstimate

# The same bounds the study evaluates on, so a correction can never be proposed
# outside the range the results cover.
ROTATION_BOUND_DEG = 2.0
TRANSLATION_BOUND_M = 0.20

# Three passes at shrinking steps. The coarse pass crosses the whole range in two
# moves; the fine pass resolves to the recovery threshold the protocol reports at.
ROTATION_STEPS_DEG: tuple[float, ...] = (1.0, 0.25, 0.1)
TRANSLATION_STEPS_M: tuple[float, ...] = (0.10, 0.025, 0.01)
COORDINATE_NAMES: tuple[str, ...] = ("roll", "pitch", "yaw", "x", "y", "z")

MAX_EVALUATIONS = 600

_BOUNDS = np.array([ROTATION_BOUND_DEG] * 3 + [TRANSLATION_BOUND_M] * 3)
_BOUND_TOLERANCE = 1e-12


class AlignmentObjective(Protocol):
    """Scores a candidate correction. Larger is better, matching the alignment score."""

    def __call__(self, correction_rpy_xyz: Float64Array) -> float: ...


class _BudgetExhausted(Exception):
    """Raised internally when the evaluation budget runs out mid-search."""


def _validated_initial(initial: Float64Array | None) -> Float64Array:
    if initial is None:
        return np.zeros(6)

    point = np.asarray(initial, dtype=np.float64)
    if point.shape != (6,) or not np.all(np.isfinite(point)):
        raise ValueError(
            f"a starting guess must be six finite numbers in the order {COORDINATE_NAMES}, "
            f"got shape {point.shape}"
        )
    if np.any(np.abs(point) > _BOUNDS + _BOUND_TOLERANCE):
        raise ValueError(
            f"a starting guess must be inside the bounds of {ROTATION_BOUND_DEG} degrees and "
            f"{TRANSLATION_BOUND_M} metres, got {point.tolist()}"
        )
    return point.copy()


def coarse_to_fine_correct(
    objective: AlignmentObjective,
    initial: Float64Array | None = None,
    *,
    max_evaluations: int = MAX_EVALUATIONS,
) -> CorrectionEstimate:
    """Maximise `objective` over the six correction degrees on a fixed shrinking grid.

    Coordinates are visited in the declared order and each is offered its negative
    offset before its positive one. A move happens only on a STRICT improvement,
    which has two consequences worth stating. A flat objective never moves the
    search, instead of walking it to a bound on no evidence. And when both offsets
    improve by exactly the same amount, the negative one was offered first and
    wins, so the answer does not depend on anything but the objective.

    The zero offset is the incumbent rather than a re-evaluated candidate. The
    objective is deterministic by construction in this study, so scoring the
    current point again would return the number already held and spend a third of
    the budget doing it.

    `converged` is true only when the search finished inside its budget AND the
    objective actually distinguished something. A flat objective returns zero,
    which may even be the right answer, but the search has no evidence for it and
    reporting convergence would present an uninformative run as a confident one.
    """

    if max_evaluations < 1:
        raise ValueError(f"the search needs at least one evaluation, got {max_evaluations}")

    current = _validated_initial(initial)
    evaluations = 0
    best_seen = -math.inf
    worst_seen = math.inf

    def score(point: Float64Array) -> float:
        nonlocal evaluations, best_seen, worst_seen
        if evaluations >= max_evaluations:
            raise _BudgetExhausted
        value = float(objective(point))
        evaluations += 1
        if not math.isfinite(value):
            raise ValueError(f"the objective must return a finite value, got {value}")
        best_seen, worst_seen = max(best_seen, value), min(worst_seen, value)
        return value

    exhausted = False
    value = math.nan
    try:
        value = score(current)
        for rotation_step, translation_step in zip(
            ROTATION_STEPS_DEG, TRANSLATION_STEPS_M, strict=True
        ):
            steps = np.array([rotation_step] * 3 + [translation_step] * 3)
            improved = True
            while improved:
                improved = False
                for coordinate in range(len(COORDINATE_NAMES)):
                    for offset in (-steps[coordinate], steps[coordinate]):
                        candidate = current.copy()
                        candidate[coordinate] += offset
                        if abs(candidate[coordinate]) > _BOUNDS[coordinate] + _BOUND_TOLERANCE:
                            continue
                        candidate_value = score(candidate)
                        if candidate_value > value:
                            current, value = candidate, candidate_value
                            improved = True
                            break
    except _BudgetExhausted:
        exhausted = True

    return CorrectionEstimate(
        rotation_rpy_deg=(float(current[0]), float(current[1]), float(current[2])),
        translation_xyz_m=(float(current[3]), float(current[4]), float(current[5])),
        objective_value=value,
        evaluations=evaluations,
        converged=not exhausted and best_seen > worst_seen,
    )
