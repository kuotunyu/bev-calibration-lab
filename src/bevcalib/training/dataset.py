"""What the learned corrector is trained on, and which scenes it may come from."""

from __future__ import annotations

from dataclasses import dataclass

from bevcalib.artifacts.results import CalibrationFaultModel
from bevcalib.perturbations.apply import inverse_fault
from bevcalib.perturbations.schedule import sample_training_fault

Target6 = tuple[float, float, float, float, float, float]


@dataclass(frozen=True)
class TrainingExample:
    """One training item: which sample, which fault, and what should be predicted."""

    sample_token: str
    scene_token: str
    epoch: int
    fault: CalibrationFaultModel
    target_rpy_xyz: Target6


def target_for_fault(fault: CalibrationFaultModel) -> Target6:
    """Return the correction a model should predict, in degrees then metres.

    The target is the INVERSE of the injected fault, not the fault with its signs
    flipped. Inverting a rigid transform rotates the translation as well as
    negating it, and a model trained on the naive negation would faithfully learn
    a small systematic error.

    Physical units on purpose: a reader can check a target against the fault it
    came from without knowing any scale factors. The loss is the single place that
    normalises.
    """

    correction = inverse_fault(fault)
    return (*correction.rotation_rpy_deg, *correction.translation_xyz_m)


def build_training_example(
    *,
    sample_token: str,
    scene_token: str,
    epoch: int,
    global_seed: int,
    development_scenes: tuple[str, ...],
    calibration_scenes: tuple[str, ...],
    evaluation_scenes: tuple[str, ...],
) -> TrainingExample:
    """Build one keyed training item, refusing every scene that is not for training.

    The cohort rule is enforced here rather than trusted to the caller because it
    is the one mistake that cannot be detected afterwards from the artifacts: a
    corrector that has seen the evaluation scenes still produces a number, and the
    number is simply not a measurement of anything.
    """

    if scene_token in evaluation_scenes:
        raise ValueError(
            f"scene {scene_token!r} is an evaluation scene and must never be trained on"
        )
    if scene_token in calibration_scenes:
        raise ValueError(
            f"scene {scene_token!r} is a calibration scene, reserved for checkpoint selection"
        )
    if scene_token not in development_scenes:
        raise ValueError(f"scene {scene_token!r} is not in any cohort")

    fault = sample_training_fault(sample_token, epoch=epoch, global_seed=global_seed)
    return TrainingExample(
        sample_token=sample_token,
        scene_token=scene_token,
        epoch=epoch,
        fault=fault,
        target_rpy_xyz=target_for_fault(fault),
    )
