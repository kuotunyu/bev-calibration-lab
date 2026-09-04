"""Contracts for the deterministic coarse-to-fine classical corrector.

The search is a coordinate sweep on a fixed grid, three times, at shrinking steps.
Nothing about it is stochastic: the same objective and the same starting point
produce the same estimate and the same number of evaluations, on any machine. A
corrector whose answer moved between runs could not be compared with the learned
one, which is the comparison the whole study exists to make.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

BOUND_DEG = 2.0
BOUND_M = 0.20


def paraboloid(target: tuple[float, ...]):  # type: ignore[no-untyped-def]
    """A convex objective whose single maximum sits exactly at `target`."""

    centre = np.asarray(target, dtype=np.float64)

    def objective(correction_rpy_xyz: np.ndarray) -> float:
        return -float(np.sum((np.asarray(correction_rpy_xyz) - centre) ** 2))

    return objective


def test_a_convex_objective_is_recovered_exactly_on_the_grid() -> None:
    """Every component of this target is a sum of the fixed steps, so it is reachable.

    Rotation steps are 1.0, 0.25 and 0.1 degrees, so 1.25 is 1.0 then 0.25.
    Translation steps are 0.10, 0.025 and 0.01 metres, so 0.125 is 0.10 then 0.025.
    """

    from bevcalib.correctors.classical import coarse_to_fine_correct

    target = (1.25, -0.25, 0.0, 0.125, -0.10, 0.0)

    estimate = coarse_to_fine_correct(paraboloid(target))

    assert estimate.rotation_rpy_deg == pytest.approx(target[:3], abs=1e-9)
    assert estimate.translation_xyz_m == pytest.approx(target[3:], abs=1e-9)
    assert estimate.converged
    assert estimate.objective_value == pytest.approx(0.0, abs=1e-15)


def test_the_same_problem_gives_the_same_answer_and_the_same_cost_every_time() -> None:
    """Determinism is what makes two runs of this corrector comparable at all."""

    from bevcalib.correctors.classical import coarse_to_fine_correct

    target = (0.5, 0.0, -1.0, 0.0, 0.05, -0.02)

    first = coarse_to_fine_correct(paraboloid(target))
    second = coarse_to_fine_correct(paraboloid(target))

    assert first == second


def test_the_search_starts_from_a_given_guess_when_one_is_offered() -> None:
    """A warm start from a previous frame is the obvious use, so it must actually be used."""

    from bevcalib.correctors.classical import coarse_to_fine_correct

    target = (1.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    warm = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    cold_start = coarse_to_fine_correct(paraboloid(target))
    warm_start = coarse_to_fine_correct(paraboloid(target), warm)

    assert warm_start.rotation_rpy_deg == pytest.approx(cold_start.rotation_rpy_deg, abs=1e-9)
    assert warm_start.evaluations < cold_start.evaluations


def test_the_estimate_never_leaves_the_bounds_the_study_evaluates() -> None:
    """An objective that rewards going further must still stop at the perturbation matrix."""

    from bevcalib.correctors.classical import coarse_to_fine_correct

    def always_further(correction_rpy_xyz: np.ndarray) -> float:
        return float(np.sum(correction_rpy_xyz))

    estimate = coarse_to_fine_correct(always_further)

    assert estimate.rotation_rpy_deg == pytest.approx((BOUND_DEG,) * 3, abs=1e-9)
    assert estimate.translation_xyz_m == pytest.approx((BOUND_M,) * 3, abs=1e-9)


def test_no_candidate_outside_the_bounds_is_ever_evaluated() -> None:
    """Scoring an out-of-range calibration would be measuring outside the study."""

    from bevcalib.correctors.classical import coarse_to_fine_correct

    seen: list[np.ndarray] = []

    def recording(correction_rpy_xyz: np.ndarray) -> float:
        seen.append(np.array(correction_rpy_xyz))
        return float(np.sum(correction_rpy_xyz))

    coarse_to_fine_correct(recording)

    visited = np.array(seen)
    assert np.all(np.abs(visited[:, :3]) <= BOUND_DEG + 1e-12)
    assert np.all(np.abs(visited[:, 3:]) <= BOUND_M + 1e-12)


def test_the_coordinates_are_visited_in_the_declared_order() -> None:
    """Roll, pitch, yaw, x, y, z. A different order finds a different local optimum."""

    from bevcalib.correctors.classical import COORDINATE_NAMES, coarse_to_fine_correct

    assert COORDINATE_NAMES == ("roll", "pitch", "yaw", "x", "y", "z")

    moved: list[int] = []

    def objective(correction_rpy_xyz: np.ndarray) -> float:
        nonzero = np.flatnonzero(np.asarray(correction_rpy_xyz))
        if nonzero.size:
            moved.append(int(nonzero[0]))
        return -float(np.sum(np.abs(correction_rpy_xyz)))

    coarse_to_fine_correct(objective)

    # Whatever is probed first within a sweep is the first coordinate.
    assert moved[0] == 0


def test_the_negative_offset_wins_a_tie_between_two_equal_improvements() -> None:
    """Two directions can improve by the same amount; the choice must not be arbitrary."""

    from bevcalib.correctors.classical import coarse_to_fine_correct

    def symmetric_well(correction_rpy_xyz: np.ndarray) -> float:
        # A double well in roll: +1 and -1 are equally good, and both beat 0.
        roll = float(np.asarray(correction_rpy_xyz)[0])
        return -abs(abs(roll) - 1.0)

    estimate = coarse_to_fine_correct(symmetric_well)

    assert estimate.rotation_rpy_deg[0] == pytest.approx(-1.0, abs=1e-9)


def test_an_objective_that_says_nothing_is_reported_as_not_converged() -> None:
    """A flat objective gives a correction of zero, and calling that converged would lie.

    Zero happens to be the right answer here, but the search has no evidence for
    it: every candidate scored the same. Reporting convergence would present an
    uninformative run as a confident one.
    """

    from bevcalib.correctors.classical import coarse_to_fine_correct

    estimate = coarse_to_fine_correct(lambda _: 0.5)

    assert estimate.rotation_rpy_deg == (0.0, 0.0, 0.0)
    assert estimate.translation_xyz_m == (0.0, 0.0, 0.0)
    assert not estimate.converged
    assert estimate.evaluations > 0


def test_starting_at_the_optimum_is_convergence_and_not_a_flat_objective() -> None:
    """Nothing improves in either case, so the two are told apart by whether anything differed."""

    from bevcalib.correctors.classical import coarse_to_fine_correct

    estimate = coarse_to_fine_correct(paraboloid((0.0,) * 6))

    assert estimate.converged
    assert estimate.rotation_rpy_deg == (0.0, 0.0, 0.0)


def test_running_out_of_evaluations_is_reported_rather_than_hidden() -> None:
    """A truncated search returns its best so far, clearly labelled as unfinished."""

    from bevcalib.correctors.classical import coarse_to_fine_correct

    estimate = coarse_to_fine_correct(paraboloid((1.25,) * 3 + (0.125,) * 3), max_evaluations=5)

    assert not estimate.converged
    assert estimate.evaluations <= 5


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_an_objective_that_is_not_a_number_fails_closed(value: float) -> None:
    """Treating a NaN score as "very bad" would hide a broken objective for a whole run."""

    from bevcalib.correctors.classical import coarse_to_fine_correct

    with pytest.raises(ValueError, match="finite"):
        coarse_to_fine_correct(lambda _: value)


@pytest.mark.parametrize(
    "initial",
    [
        np.array([3.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 0.0, 0.5, 0.0, 0.0]),
    ],
)
def test_a_starting_guess_outside_the_bounds_is_refused(initial: np.ndarray) -> None:
    """Silently clipping it would answer a different question than the caller asked."""

    from bevcalib.correctors.classical import coarse_to_fine_correct

    with pytest.raises(ValueError, match="bounds"):
        coarse_to_fine_correct(paraboloid((0.0,) * 6), initial)


@pytest.mark.parametrize(
    "initial",
    [np.zeros(5), np.zeros(7), np.zeros((2, 3)), np.array([float("nan")] + [0.0] * 5)],
)
def test_a_starting_guess_that_is_not_six_finite_numbers_is_refused(
    initial: np.ndarray,
) -> None:
    """Six degrees of freedom, in the declared order; anything else is a wiring mistake."""

    from bevcalib.correctors.classical import coarse_to_fine_correct

    with pytest.raises(ValueError, match=r"six|finite"):
        coarse_to_fine_correct(paraboloid((0.0,) * 6), initial)


def test_the_reported_objective_value_belongs_to_the_reported_estimate() -> None:
    """A value from some other point would make the estimate impossible to audit."""

    from bevcalib.correctors.classical import coarse_to_fine_correct

    target = (1.0, 0.0, 0.0, 0.10, 0.0, 0.0)
    objective = paraboloid(target)

    estimate = coarse_to_fine_correct(objective)

    recomputed = objective(np.array([*estimate.rotation_rpy_deg, *estimate.translation_xyz_m]))
    assert estimate.objective_value == pytest.approx(recomputed, abs=1e-15)


def test_correcting_a_known_fault_leaves_far_less_of_it_behind() -> None:
    """The claim a corrector exists to support, on the real composition rather than a proxy."""

    from bevcalib.artifacts.results import CalibrationFaultModel
    from bevcalib.correctors.classical import coarse_to_fine_correct
    from bevcalib.geometry.se3 import SE3, compose, inverse
    from bevcalib.perturbations.apply import apply_metadata_fault, fault_to_se3

    true = SE3(rotation_wxyz=(1.0, 0.0, 0.0, 0.0), translation_xyz_m=(1.7, 0.0, 1.5))
    injected = CalibrationFaultModel(
        rotation_rpy_deg=(0.0, 0.0, 1.0),
        translation_xyz_m=(0.10, 0.0, 0.0),
        requested_time_offset_ms=0,
    )
    assumed = apply_metadata_fault(true, injected)

    def gap(candidate: SE3) -> float:
        relative = compose(inverse(candidate), true)
        angle = 2.0 * math.degrees(math.acos(min(1.0, abs(relative.rotation_wxyz[0]))))
        return angle + float(np.linalg.norm(relative.translation_xyz_m))

    def objective(correction_rpy_xyz: np.ndarray) -> float:
        values = np.asarray(correction_rpy_xyz, dtype=np.float64)
        correction = CalibrationFaultModel(
            rotation_rpy_deg=(values[0], values[1], values[2]),
            translation_xyz_m=(values[3], values[4], values[5]),
            requested_time_offset_ms=0,
        )
        return -gap(compose(assumed, fault_to_se3(correction)))

    estimate = coarse_to_fine_correct(objective)
    corrected = compose(
        assumed,
        fault_to_se3(
            CalibrationFaultModel(
                rotation_rpy_deg=estimate.rotation_rpy_deg,
                translation_xyz_m=estimate.translation_xyz_m,
                requested_time_offset_ms=0,
            )
        ),
    )

    assert gap(corrected) < 0.1 * gap(assumed)


def test_a_search_with_no_evaluations_at_all_is_refused() -> None:
    """A budget of zero cannot even score the starting point, so there is nothing to report."""

    from bevcalib.correctors.classical import coarse_to_fine_correct

    with pytest.raises(ValueError, match="at least one evaluation"):
        coarse_to_fine_correct(paraboloid((0.0,) * 6), max_evaluations=0)


def test_a_budget_of_one_evaluation_is_a_valid_search() -> None:
    """One evaluation is the smallest honest search: score the start and stop.

    The bound is at-least-one, so one must pass it. Written `<= 1` or `< 2` the
    validator would refuse the cheapest possible call, which is exactly what a
    caller probing the objective once would ask for, and the refusal would read
    as a malformed request rather than as an off-by-one bound.

    The single evaluation is spent on the incumbent, so the search cannot move
    and cannot claim convergence: it has no second value to compare against.
    """

    from bevcalib.correctors.classical import coarse_to_fine_correct

    calls: list[int] = []

    def objective(point: np.ndarray) -> float:
        calls.append(1)
        return -float(np.abs(point).sum())

    result = coarse_to_fine_correct(objective, np.zeros(6), max_evaluations=1)

    assert len(calls) == 1
    assert not result.converged
    assert result.rotation_rpy_deg == (0.0, 0.0, 0.0)
    assert result.translation_xyz_m == (0.0, 0.0, 0.0)


def test_the_budget_counts_every_objective_call_including_the_first() -> None:
    """The incumbent's own score is an evaluation, and the count starts at zero.

    Starting the counter at one would spend a call the objective never made, so
    a caller who budgeted 600 would get 599 and the recorded evaluation count
    would disagree with the number of times their own function ran. That
    discrepancy is invisible in the result and would quietly change what a
    fixed-budget comparison between two correctors means.
    """

    from bevcalib.correctors.classical import coarse_to_fine_correct

    calls: list[int] = []

    def objective(point: np.ndarray) -> float:
        calls.append(1)
        return -float(np.abs(point).sum())

    result = coarse_to_fine_correct(objective, np.zeros(6), max_evaluations=5)

    assert len(calls) == 5
    assert result.evaluations == 5


@pytest.mark.parametrize(
    "coordinate",
    [0, 3],
    ids=["a rotation at its bound", "a translation at its bound"],
)
def test_a_starting_point_exactly_on_its_bound_is_accepted(coordinate: int) -> None:
    """The bound is inclusive to within a tolerance, so the bound itself is inside.

    A search is often started from the previous frame's answer, and that answer
    can sit exactly on a bound. Written `>=` the validator would refuse it and
    the caller would be told their starting point is out of range when it is
    precisely in range; widening the comparison the other way, by subtracting
    the tolerance instead of adding it, refuses everything within a hair of the
    bound for the same reason.
    """

    from bevcalib.correctors.classical import (
        ROTATION_BOUND_DEG,
        TRANSLATION_BOUND_M,
        coarse_to_fine_correct,
    )

    start = np.zeros(6)
    start[coordinate] = ROTATION_BOUND_DEG if coordinate < 3 else TRANSLATION_BOUND_M

    result = coarse_to_fine_correct(lambda point: 0.0, start, max_evaluations=50)

    recovered = np.array([*result.rotation_rpy_deg, *result.translation_xyz_m])
    np.testing.assert_allclose(recovered, start, atol=0.0)
