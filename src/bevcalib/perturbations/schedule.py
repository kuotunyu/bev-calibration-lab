"""The formal fault schedule, and the keyed faults used to train a corrector."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict

from bevcalib.artifacts.results import CalibrationFaultModel

FaultAxis = Literal["roll", "pitch", "yaw", "x", "y", "z", "time"]

# The matrix is stated twice on purpose: here, where the code uses it, and in
# `configs/perturbations/formal_v1.yaml`, where a reader can see it. A test binds
# the two together, so neither can move without the other going red. Loading the
# YAML at import time instead would make the installed package depend on a file
# that is not part of the wheel.
ROTATION_SINGLE_AXIS_DEG: tuple[float, ...] = (
    -2.0,
    -1.0,
    -0.5,
    -0.25,
    -0.1,
    0.0,
    0.1,
    0.25,
    0.5,
    1.0,
    2.0,
)
TRANSLATION_SINGLE_AXIS_M: tuple[float, ...] = (
    -0.20,
    -0.10,
    -0.05,
    -0.02,
    0.0,
    0.02,
    0.05,
    0.10,
    0.20,
)
TIMING_OFFSET_MS: tuple[int, ...] = (-200, -100, -50, 0, 50, 100, 200)

ROTATION_AXES: tuple[FaultAxis, ...] = ("roll", "pitch", "yaw")
TRANSLATION_AXES: tuple[FaultAxis, ...] = ("x", "y", "z")

TIMING_MAX_SELECTION_ERROR_MS = 25.0
LEARNED_TRAINING_ROTATION_BOUND_DEG = 2.0
LEARNED_TRAINING_TRANSLATION_BOUND_M = 0.20

_UINT32_MAX = 0xFFFFFFFF


class PerturbationMatrixV1(BaseModel):
    """The protocol's perturbation matrix, validated strictly on load."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["bev-perturbation-matrix/v1"]
    rotation_single_axis_deg: tuple[float, ...]
    translation_single_axis_m: tuple[float, ...]
    timing_offset_ms: tuple[int, ...]
    timing_max_selection_error_ms: float
    learned_training_rotation_bound_deg: float
    learned_training_translation_bound_m: float
    recovery_rotation_threshold_deg: float
    recovery_translation_threshold_m: float


def load_perturbation_matrix(path: Path) -> PerturbationMatrixV1:
    """Load and strictly validate one perturbation matrix document."""

    return parse_perturbation_matrix(path.read_bytes())


def parse_perturbation_matrix(raw: bytes) -> PerturbationMatrixV1:
    """Validate the exact bytes the caller binds into a protocol identity."""
    from bevcalib.metrics.calibration import (
        RECOVERY_ROTATION_THRESHOLD_DEG,
        RECOVERY_TRANSLATION_THRESHOLD_M,
    )

    matrix = PerturbationMatrixV1.model_validate(yaml.safe_load(raw))
    supported = PerturbationMatrixV1(
        schema_version="bev-perturbation-matrix/v1",
        rotation_single_axis_deg=ROTATION_SINGLE_AXIS_DEG,
        translation_single_axis_m=TRANSLATION_SINGLE_AXIS_M,
        timing_offset_ms=TIMING_OFFSET_MS,
        timing_max_selection_error_ms=TIMING_MAX_SELECTION_ERROR_MS,
        learned_training_rotation_bound_deg=LEARNED_TRAINING_ROTATION_BOUND_DEG,
        learned_training_translation_bound_m=LEARNED_TRAINING_TRANSLATION_BOUND_M,
        recovery_rotation_threshold_deg=RECOVERY_ROTATION_THRESHOLD_DEG,
        recovery_translation_threshold_m=RECOVERY_TRANSLATION_THRESHOLD_M,
    )
    if matrix != supported:
        raise ValueError("referenced perturbation matrix differs from supported compiled V1 values")
    return matrix


def formal_single_axis_faults() -> tuple[tuple[FaultAxis, float], ...]:
    """Return every formal condition, in run order: axis then magnitude.

    Zero appears on all seven axes. They are one physical condition, so a runner
    may compute the baseline once and reuse it; that is an optimisation and not
    the schedule's business. Each single-axis sweep needs its own zero, or its
    curve has no origin to pass through.
    """

    entries: list[tuple[FaultAxis, float]] = []
    for axis in ROTATION_AXES:
        entries.extend((axis, value) for value in ROTATION_SINGLE_AXIS_DEG)
    for axis in TRANSLATION_AXES:
        entries.extend((axis, value) for value in TRANSLATION_SINGLE_AXIS_M)
    entries.extend(("time", float(value)) for value in TIMING_OFFSET_MS)
    return tuple(entries)


def sample_training_fault(sample_token: str, epoch: int, global_seed: int) -> CalibrationFaultModel:
    """Draw the fault this sample is trained on, from the key alone.

    The six components are read straight out of a SHA-256 digest rather than from
    a seeded generator. A digest is identical on every machine, every Python build
    and every library version, for as long as the study exists; a generator stream
    is only guaranteed while the library keeps its promise. For a result that has
    to be reproducible years later, that difference is the whole point.

    Timing is never drawn: it is a stress condition that no 6DoF pose can express.
    """

    key = f"{global_seed}|{epoch}|{sample_token}".encode()
    digest = hashlib.sha256(key).digest()
    draws = [
        int.from_bytes(digest[index * 4 : index * 4 + 4], "big") / _UINT32_MAX for index in range(6)
    ]
    rotation = tuple((2.0 * draw - 1.0) * LEARNED_TRAINING_ROTATION_BOUND_DEG for draw in draws[:3])
    translation = tuple(
        (2.0 * draw - 1.0) * LEARNED_TRAINING_TRANSLATION_BOUND_M for draw in draws[3:]
    )
    return CalibrationFaultModel(
        rotation_rpy_deg=(rotation[0], rotation[1], rotation[2]),
        translation_xyz_m=(translation[0], translation[1], translation[2]),
        requested_time_offset_ms=0,
    )
