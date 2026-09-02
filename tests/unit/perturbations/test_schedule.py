"""Contracts for which faults the study runs, and for the keyed training draws.

The schedule is the experiment. If it drifts, two runs are not comparable and no
aggregate over them means anything, so it is pinned here by exact values rather
than by a rule that regenerates it.

The training draws are keyed rather than random: a corrector trained on faults
that change between runs cannot be compared with one that was not.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG = REPO_ROOT / "configs" / "perturbations" / "formal_v1.yaml"

ROTATIONS = (-2.0, -1.0, -0.5, -0.25, -0.1, 0.0, 0.1, 0.25, 0.5, 1.0, 2.0)
TRANSLATIONS = (-0.20, -0.10, -0.05, -0.02, 0.0, 0.02, 0.05, 0.10, 0.20)
TIMINGS = (-200.0, -100.0, -50.0, 0.0, 50.0, 100.0, 200.0)


def test_the_schedule_is_every_axis_crossed_with_its_own_value_list() -> None:
    """Three rotation axes at eleven values, three translation axes at nine, time at seven."""

    from bevcalib.perturbations.schedule import formal_single_axis_faults

    schedule = formal_single_axis_faults()

    assert len(schedule) == 3 * 11 + 3 * 9 + 7 == 67
    for axis in ("roll", "pitch", "yaw"):
        assert tuple(value for name, value in schedule if name == axis) == ROTATIONS
    for axis in ("x", "y", "z"):
        assert tuple(value for name, value in schedule if name == axis) == TRANSLATIONS
    assert tuple(value for name, value in schedule if name == "time") == TIMINGS


def test_the_axes_appear_in_a_fixed_order() -> None:
    """Run order is part of the record; a set would make two runs incomparable to read."""

    from bevcalib.perturbations.schedule import formal_single_axis_faults

    axes = tuple(dict.fromkeys(axis for axis, _ in formal_single_axis_faults()))

    assert axes == ("roll", "pitch", "yaw", "x", "y", "z", "time")


def test_every_axis_carries_its_own_baseline() -> None:
    """Each single-axis sweep needs its own zero, or its curve has no origin to pass through.

    The seven zero entries are one physical condition, so a runner may compute it
    once and reuse it. That is a runner's optimisation and not the schedule's
    business: the schedule states the experimental design.
    """

    from bevcalib.perturbations.schedule import formal_single_axis_faults

    zeros = [axis for axis, value in formal_single_axis_faults() if value == 0.0]

    assert zeros == ["roll", "pitch", "yaw", "x", "y", "z", "time"]


def test_the_magnitudes_are_symmetric_about_zero() -> None:
    """An asymmetric sweep cannot separate a sign error from a real directional effect."""

    from bevcalib.perturbations.schedule import formal_single_axis_faults

    for axis in ("roll", "x", "time"):
        values = [value for name, value in formal_single_axis_faults() if name == axis]
        assert sorted(values) == sorted(-value for value in values)


def test_the_committed_config_and_the_code_state_the_same_matrix() -> None:
    """The config is the readable statement of the protocol; drift between them is silent."""

    from bevcalib.perturbations.schedule import load_perturbation_matrix

    matrix = load_perturbation_matrix(CONFIG)

    assert matrix.rotation_single_axis_deg == ROTATIONS
    assert matrix.translation_single_axis_m == TRANSLATIONS
    assert matrix.timing_offset_ms == tuple(int(value) for value in TIMINGS)
    assert matrix.timing_max_selection_error_ms == 25
    assert matrix.learned_training_rotation_bound_deg == 2.0
    assert matrix.learned_training_translation_bound_m == 0.20
    assert matrix.recovery_rotation_threshold_deg == 0.25
    assert matrix.recovery_translation_threshold_m == 0.05


def test_the_schedule_is_built_from_that_same_matrix() -> None:
    """One source of truth, asserted rather than assumed."""

    from bevcalib.perturbations.schedule import formal_single_axis_faults, load_perturbation_matrix

    matrix = load_perturbation_matrix(CONFIG)
    schedule = formal_single_axis_faults()

    assert tuple(v for a, v in schedule if a == "roll") == matrix.rotation_single_axis_deg
    assert tuple(v for a, v in schedule if a == "z") == matrix.translation_single_axis_m


def test_a_config_that_does_not_describe_the_matrix_is_rejected(tmp_path: Path) -> None:
    """A widened bound or a stray key would change the experiment without saying so."""

    from pydantic import ValidationError

    from bevcalib.perturbations.schedule import load_perturbation_matrix

    document = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    path = tmp_path / "broken.yaml"
    path.write_text(yaml.safe_dump(document | {"unexpected": 1}), encoding="utf-8")

    with pytest.raises(ValidationError):
        load_perturbation_matrix(path)


def test_the_same_key_always_draws_the_same_training_fault() -> None:
    """A corrector trained on faults that move between runs cannot be compared with one that was not."""

    from bevcalib.perturbations.schedule import sample_training_fault

    first = sample_training_fault("scene-token-a", epoch=3, global_seed=20260902)
    again = sample_training_fault("scene-token-a", epoch=3, global_seed=20260902)

    assert first == again


@pytest.mark.parametrize(
    ("token", "epoch", "seed"),
    [("scene-token-b", 3, 20260902), ("scene-token-a", 4, 20260902), ("scene-token-a", 3, 1)],
)
def test_changing_any_part_of_the_key_changes_the_draw(token: str, epoch: int, seed: int) -> None:
    """Sample, epoch and seed must each move the fault, or one of them is being ignored."""

    from bevcalib.perturbations.schedule import sample_training_fault

    baseline = sample_training_fault("scene-token-a", epoch=3, global_seed=20260902)

    assert sample_training_fault(token, epoch=epoch, global_seed=seed) != baseline


def test_every_training_draw_stays_inside_the_bounds_it_will_be_evaluated_on() -> None:
    """Training outside the evaluated range would report recovery the study never measured."""

    from bevcalib.perturbations.schedule import load_perturbation_matrix, sample_training_fault

    matrix = load_perturbation_matrix(CONFIG)

    for index in range(400):
        fault = sample_training_fault(f"scene-{index}", epoch=index % 7, global_seed=20260902)
        assert all(
            abs(value) <= matrix.learned_training_rotation_bound_deg
            for value in fault.rotation_rpy_deg
        )
        assert all(
            abs(value) <= matrix.learned_training_translation_bound_m
            for value in fault.translation_xyz_m
        )


def test_a_training_draw_never_carries_a_timing_fault() -> None:
    """Timing is a stress condition, not something a 6DoF corrector can predict."""

    from bevcalib.perturbations.schedule import sample_training_fault

    for index in range(50):
        assert (
            sample_training_fault(f"scene-{index}", epoch=1, global_seed=7).requested_time_offset_ms
            == 0
        )


def test_the_draws_spread_over_the_range_rather_than_repeating_one_value() -> None:
    """A key that collapses to one value would be deterministic and useless."""

    from bevcalib.perturbations.schedule import sample_training_fault

    rolls = {
        sample_training_fault(f"scene-{index}", epoch=0, global_seed=11).rotation_rpy_deg[0]
        for index in range(200)
    }

    assert len(rolls) > 190
    assert min(rolls) < -1.0
    assert max(rolls) > 1.0
