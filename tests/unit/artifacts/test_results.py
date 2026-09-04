"""Contracts for one calibration result: the row every figure in this study reduces to.

A result is per sample, and it carries both the fault that was injected and the
estimate a corrector produced, so recovery can be recomputed from the artifact
rather than trusted. The invariants here exist because a NaN, a negative error or
an unexplained invalid row silently poisons every aggregate downstream.
"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

FAULT: dict[str, Any] = {
    "rotation_rpy_deg": [1.0, 0.0, 0.0],
    "translation_xyz_m": [0.0, 0.05, 0.0],
    "requested_time_offset_ms": 50,
}
VALID: dict[str, Any] = {
    "schema_version": "bev-calibration-result/v1",
    "sample_token": "0" * 32,
    "scene_token": "1" * 32,
    "fault": FAULT,
    "estimate": FAULT | {"rotation_rpy_deg": [0.9, 0.0, 0.0]},
    "rotation_geodesic_error_deg": 0.1,
    "translation_error_m": 0.004,
    "pixel_error_median": 2.5,
    "pixel_error_p90": 7.25,
    "edge_alignment_score": 1.84,
    "bev_ground_contact_error_m": 0.12,
    "valid": True,
    "invalid_reason": None,
}


def test_a_complete_result_is_accepted_and_keeps_both_fault_and_estimate() -> None:
    """Recovery is fault minus estimate, so the artifact must carry both, not the difference."""

    from bevcalib.artifacts.results import CalibrationResultV1

    result = CalibrationResultV1.model_validate(VALID)

    assert result.fault.rotation_rpy_deg == (1.0, 0.0, 0.0)
    assert result.estimate.rotation_rpy_deg == (0.9, 0.0, 0.0)
    assert result.valid is True


@pytest.mark.parametrize(
    "field",
    [
        "rotation_geodesic_error_deg",
        "translation_error_m",
        "pixel_error_median",
        "pixel_error_p90",
        "edge_alignment_score",
        "bev_ground_contact_error_m",
    ],
)
@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_a_non_finite_measurement_is_rejected(field: str, value: float) -> None:
    """One NaN reaching a mean turns the entire aggregate into NaN, silently."""

    from bevcalib.artifacts.results import CalibrationResultV1

    with pytest.raises(ValidationError, match=rf"\n{field}\n"):
        CalibrationResultV1.model_validate(VALID | {field: value})


@pytest.mark.parametrize(
    "field",
    [
        "rotation_geodesic_error_deg",
        "translation_error_m",
        "pixel_error_median",
        "pixel_error_p90",
        "edge_alignment_score",
        "bev_ground_contact_error_m",
    ],
)
def test_a_negative_magnitude_is_rejected(field: str) -> None:
    """These are distances, angles and ratios; a negative one means a sign bug upstream."""

    from bevcalib.artifacts.results import CalibrationResultV1

    with pytest.raises(ValidationError, match=rf"\n{field}\n"):
        CalibrationResultV1.model_validate(VALID | {field: -0.1})


def test_a_ninetieth_percentile_below_the_median_is_rejected() -> None:
    """Swapped percentile arguments produce numbers that look fine one at a time."""

    from bevcalib.artifacts.results import CalibrationResultV1

    with pytest.raises(
        ValidationError,
        match=r"^1 validation error for CalibrationResultV1\n  Value error, pixel_error_p90 must not be below pixel_error_median",
    ):
        CalibrationResultV1.model_validate(VALID | {"pixel_error_median": 9.0})


def test_an_invalid_result_must_say_why() -> None:
    """An unexplained dropped sample is indistinguishable from a silently biased cohort."""

    from bevcalib.artifacts.results import CalibrationResultV1

    invalid = CalibrationResultV1.model_validate(
        VALID | {"valid": False, "invalid_reason": "no LiDAR points project into the image"}
    )
    assert invalid.invalid_reason is not None

    with pytest.raises(
        ValidationError,
        match=r"^1 validation error for CalibrationResultV1\n  Value error, an invalid result requires a non-empty invalid_reason",
    ):
        CalibrationResultV1.model_validate(VALID | {"valid": False})


def test_a_valid_result_must_not_carry_a_reason_for_being_invalid() -> None:
    """A row that is both valid and explained away is ambiguous to every consumer."""

    from bevcalib.artifacts.results import CalibrationResultV1

    with pytest.raises(
        ValidationError,
        match=r"^1 validation error for CalibrationResultV1\n  Value error, a valid result must not carry an invalid_reason",
    ):
        CalibrationResultV1.model_validate(VALID | {"invalid_reason": "partially occluded"})


def test_an_empty_reason_does_not_count_as_an_explanation() -> None:
    """An empty string satisfies `is not None` and explains nothing."""

    from bevcalib.artifacts.results import CalibrationResultV1

    with pytest.raises(ValidationError):
        CalibrationResultV1.model_validate(VALID | {"valid": False, "invalid_reason": ""})


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("schema_version", "bev-calibration-result/v2"),
        ("sample_token", ""),
        ("scene_token", ""),
    ],
)
def test_a_result_that_cannot_identify_its_sample_is_rejected(field: str, value: str) -> None:
    """A result that cannot be joined back to its scene cannot be audited."""

    from bevcalib.artifacts.results import CalibrationResultV1

    with pytest.raises(ValidationError, match=rf"\n{field}\n"):
        CalibrationResultV1.model_validate(VALID | {field: value})


def test_a_fault_needs_three_rotation_and_three_translation_components() -> None:
    """A two-element rotation would broadcast somewhere downstream instead of failing."""

    from bevcalib.artifacts.results import CalibrationFaultModel

    with pytest.raises(ValidationError):
        CalibrationFaultModel.model_validate(FAULT | {"rotation_rpy_deg": [1.0, 0.0]})


@pytest.mark.parametrize("field", ["rotation_rpy_deg", "translation_xyz_m"])
def test_a_non_finite_fault_component_is_rejected(field: str) -> None:
    """An infinite injected fault is not a fault, it is a broken experiment definition."""

    from bevcalib.artifacts.results import CalibrationFaultModel

    with pytest.raises(ValidationError, match=rf"\n{field}\.\d+\n"):
        CalibrationFaultModel.model_validate(FAULT | {field: [float("nan"), 0.0, 0.0]})


def test_a_validated_result_cannot_be_mutated() -> None:
    """Results are evidence; anything that can be adjusted after the fact is not."""

    from bevcalib.artifacts.results import CalibrationResultV1

    result = CalibrationResultV1.model_validate(VALID)

    with pytest.raises(ValidationError):
        result.pixel_error_median = 0.0  # type: ignore[misc]


def test_an_unknown_field_is_rejected() -> None:
    """A renamed metric must break loudly rather than be dropped on read."""

    from bevcalib.artifacts.results import CalibrationResultV1

    with pytest.raises(ValidationError):
        CalibrationResultV1.model_validate(VALID | {"iou": 0.5})
