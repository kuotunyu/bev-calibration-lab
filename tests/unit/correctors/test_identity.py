"""Contracts for the corrector that does nothing.

Every recovery number the study reports is measured against this baseline, so an
identity that quietly improved anything would flatter every other corrector by
exactly that much. Its whole job is to be exactly zero, and to say plainly that it
never looked at an objective.
"""

from __future__ import annotations

import math


def test_the_identity_correction_is_exactly_zero_and_not_nearly_zero() -> None:
    """Exact, because a baseline with a small bias shifts every comparison against it."""

    from bevcalib.correctors.identity import identity_correction

    estimate = identity_correction()

    assert estimate.rotation_rpy_deg == (0.0, 0.0, 0.0)
    assert estimate.translation_xyz_m == (0.0, 0.0, 0.0)


def test_the_identity_never_evaluated_an_objective_and_says_so() -> None:
    """Reporting an objective value of 0.0 would read as a perfect alignment score.

    The alignment score is a negated distance, so zero is its ceiling. A baseline
    that never measured anything must not report the best possible number; `nan`
    with zero evaluations is the truthful pair.
    """

    from bevcalib.correctors.identity import identity_correction

    estimate = identity_correction()

    assert estimate.evaluations == 0
    assert math.isnan(estimate.objective_value)


def test_the_identity_counts_as_converged() -> None:
    """It ran no search, so it cannot have failed one, and a filter on `converged`
    must not silently drop the baseline every other corrector is compared against."""

    from bevcalib.correctors.identity import identity_correction

    assert identity_correction().converged


def test_two_identity_corrections_are_the_same_value() -> None:
    """A frozen record with no inputs should compare equal, run after run."""

    from bevcalib.correctors.identity import identity_correction

    first, second = identity_correction(), identity_correction()

    assert first.rotation_rpy_deg == second.rotation_rpy_deg
    assert first.translation_xyz_m == second.translation_xyz_m
    assert first.evaluations == second.evaluations


def test_an_estimate_cannot_be_edited_after_the_fact() -> None:
    """Estimates are evidence about a corrector, like every other record here."""

    import dataclasses

    import pytest

    from bevcalib.correctors.identity import identity_correction

    with pytest.raises(dataclasses.FrozenInstanceError):
        identity_correction().converged = False  # type: ignore[misc]
