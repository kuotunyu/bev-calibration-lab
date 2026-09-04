"""Contracts for what the learned corrector is trained on, and on which scenes.

Two things decide whether the learned result means anything. The target has to be
the correction that actually undoes the injected fault, which is not the fault
negated; and the scenes have to come from the development pool only, because a
corrector that has seen the evaluation scenes is not being evaluated on anything.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

DEVELOPMENT = ("scene-dev-1", "scene-dev-2")
CALIBRATION = ("scene-cal-1",)
EVALUATION = ("scene-eval-1",)


def example(**overrides: object):  # type: ignore[no-untyped-def]
    from bevcalib.training.dataset import build_training_example

    arguments: dict[str, object] = {
        "sample_token": "sample-0",
        "scene_token": "scene-dev-1",
        "epoch": 3,
        "global_seed": 20260902,
        "development_scenes": DEVELOPMENT,
        "calibration_scenes": CALIBRATION,
        "evaluation_scenes": EVALUATION,
    }
    return build_training_example(**(arguments | overrides))  # type: ignore[arg-type]


def test_an_example_carries_the_fault_it_was_built_from() -> None:
    """The fault is evidence: the target can be rechecked against it after the fact."""

    from bevcalib.perturbations.schedule import sample_training_fault

    built = example()

    assert built.sample_token == "sample-0"
    assert built.scene_token == "scene-dev-1"
    assert built.epoch == 3
    assert built.fault == sample_training_fault("sample-0", epoch=3, global_seed=20260902)


def test_the_same_key_builds_the_same_example_every_time() -> None:
    """Training on faults that drift between runs makes two trainings incomparable."""

    assert example() == example()


def test_the_target_is_the_correction_that_undoes_the_fault() -> None:
    """Composing the fault with the target must land back on the identity."""

    from bevcalib.artifacts.results import CalibrationFaultModel
    from bevcalib.geometry.se3 import compose
    from bevcalib.perturbations.apply import fault_to_se3

    built = example()
    target = CalibrationFaultModel(
        rotation_rpy_deg=(
            built.target_rpy_xyz[0],
            built.target_rpy_xyz[1],
            built.target_rpy_xyz[2],
        ),
        translation_xyz_m=(
            built.target_rpy_xyz[3],
            built.target_rpy_xyz[4],
            built.target_rpy_xyz[5],
        ),
        requested_time_offset_ms=0,
    )

    residual = compose(fault_to_se3(built.fault), fault_to_se3(target))

    assert residual.rotation_wxyz == pytest.approx((1.0, 0.0, 0.0, 0.0), abs=1e-9)
    assert residual.translation_xyz_m == pytest.approx((0.0, 0.0, 0.0), abs=1e-9)


def test_the_target_is_not_simply_the_fault_with_its_signs_flipped() -> None:
    """Negating a fault is the obvious wrong answer, and it is wrong by a real amount.

    Inverting a rigid transform rotates the translation as well as negating it, so
    for a yawed fault the translation target differs from the naive negation by
    the sine of the angle times the offset.
    """

    from bevcalib.artifacts.results import CalibrationFaultModel
    from bevcalib.perturbations.apply import inverse_fault

    injected = CalibrationFaultModel(
        rotation_rpy_deg=(0.0, 0.0, 1.0),
        translation_xyz_m=(0.10, 0.0, 0.0),
        requested_time_offset_ms=0,
    )

    correction = inverse_fault(injected)

    assert correction.rotation_rpy_deg == pytest.approx((0.0, 0.0, -1.0), abs=1e-9)
    # The naive negation would be exactly (-0.10, 0.0, 0.0).
    assert correction.translation_xyz_m[0] == pytest.approx(-0.10 * math.cos(math.radians(1.0)))
    assert correction.translation_xyz_m[1] == pytest.approx(0.10 * math.sin(math.radians(1.0)))
    assert abs(correction.translation_xyz_m[1]) > 1e-4


def test_the_target_is_six_finite_numbers_in_physical_units() -> None:
    """Degrees then metres, so a reader can check a target against the fault by eye."""

    built = example()

    assert len(built.target_rpy_xyz) == 6
    assert all(math.isfinite(value) for value in built.target_rpy_xyz)
    assert np.all(np.abs(built.target_rpy_xyz[:3]) <= 2.0 + 1e-9)
    assert np.all(np.abs(built.target_rpy_xyz[3:]) <= 0.2 + 1e-9)


def test_a_calibration_scene_is_refused_and_told_what_it_is_for() -> None:
    """Calibration scenes pick the checkpoint; training on them would pick it on itself."""

    from bevcalib.training.dataset import build_training_example

    with pytest.raises(
        ValueError, match=r"^scene .* is a calibration scene, reserved for checkpoint selection$"
    ):
        example(scene_token="scene-cal-1")

    assert build_training_example is not None


def test_an_evaluation_scene_is_refused_in_the_strongest_terms() -> None:
    """This is the one that would invalidate the whole study rather than degrade it."""

    with pytest.raises(
        ValueError, match=r"^scene .* is an evaluation scene and must never be trained on$"
    ):
        example(scene_token="scene-eval-1")


def test_a_scene_in_no_cohort_at_all_is_refused() -> None:
    """An unknown scene means the cohort and the loader disagree, which is worth stopping for."""

    with pytest.raises(ValueError, match=r"^scene .* is not in any cohort$"):
        example(scene_token="scene-who-knows")


def test_an_example_cannot_be_edited_after_the_fact() -> None:
    """What a model was trained on is provenance, and provenance is frozen here."""

    import dataclasses

    with pytest.raises(dataclasses.FrozenInstanceError):
        example().epoch = 4  # type: ignore[misc]


def test_a_fault_carrying_a_timing_offset_cannot_become_a_training_target() -> None:
    """Timing is a stress condition; no 6DoF pose expresses it, so it has no target."""

    from bevcalib.artifacts.results import CalibrationFaultModel
    from bevcalib.training.dataset import target_for_fault

    with pytest.raises(
        ValueError, match=r"^a timing fault has no 6DoF inverse: requested_time_offset_ms is "
    ):
        target_for_fault(
            CalibrationFaultModel(
                rotation_rpy_deg=(1.0, 0.0, 0.0),
                translation_xyz_m=(0.0, 0.0, 0.0),
                requested_time_offset_ms=50,
            )
        )
