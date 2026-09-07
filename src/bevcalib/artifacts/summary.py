"""Portable safe summary contract for rendering without private run documents."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from bevcalib.artifacts.measurements import EvaluationMeasurements
from bevcalib.artifacts.result_documents import condition_inventory, digest
from bevcalib.artifacts.results import FiniteFloat, NonNegativeFiniteFloat
from bevcalib.cohort.manifest import Digest
from bevcalib.metrics.reprojection import RANGE_BINS
from bevcalib.metrics.summary import FORMAL_RUNS, SYNTHETIC_PAIR, condition_key

Count = Annotated[int, Field(ge=0, strict=True)]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Statistic(_Strict):
    count: Count
    mean: FiniteFloat | None
    median: FiniteFloat | None
    p90: FiniteFloat | None

    @model_validator(mode="after")
    def coherent(self) -> Self:
        values = (self.mean, self.median, self.p90)
        if (self.count == 0 and any(value is not None for value in values)) or (
            self.count > 0 and any(value is None for value in values)
        ):
            raise ValueError(
                "empty measurements require null statistics; measured values require a denominator"
            )
        return self


class MagnitudeStatistic(Statistic):
    mean: NonNegativeFiniteFloat | None
    median: NonNegativeFiniteFloat | None
    p90: NonNegativeFiniteFloat | None


class EdgeStatistic(Statistic):
    mean: Annotated[float, Field(allow_inf_nan=False, le=0)] | None
    median: Annotated[float, Field(allow_inf_nan=False, le=0)] | None
    p90: Annotated[float, Field(allow_inf_nan=False, le=0)] | None


class Validity(_Strict):
    total: Annotated[int, Field(gt=0, strict=True)]
    valid: Count
    invalid: Count
    invalid_rate: Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
    reasons: dict[str, Annotated[int, Field(gt=0, strict=True)]]

    @model_validator(mode="after")
    def coherent(self) -> Self:
        if (
            self.valid + self.invalid != self.total
            or self.invalid_rate != self.invalid / self.total
            or sum(self.reasons.values()) != self.invalid
        ):
            raise ValueError("summary validity counts/rate/reasons disagree")
        return self


class GroundStatistic(MagnitudeStatistic):
    total: Count
    invalid: Count
    reasons: dict[str, Annotated[int, Field(gt=0, strict=True)]]

    @model_validator(mode="after")
    def counts(self) -> Self:
        if self.count + self.invalid != self.total or sum(self.reasons.values()) != self.invalid:
            raise ValueError("ground-contact counts/reasons disagree")
        return self


class PixelStatistic(MagnitudeStatistic):
    projection_count: Count

    @model_validator(mode="after")
    def counts(self) -> Self:
        if self.count > self.projection_count:
            raise ValueError("pixel count exceeds projection count")
        return self


class PoseSummary(_Strict):
    rotation_rpy_error_deg: tuple[Statistic, Statistic, Statistic]
    translation_xyz_error_m: tuple[Statistic, Statistic, Statistic]
    rotation_geodesic_error_deg: MagnitudeStatistic
    translation_error_m: MagnitudeStatistic


class Recovery(_Strict):
    valid: Count
    recovered: Count
    rate: Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)] | None

    @model_validator(mode="after")
    def coherent(self) -> Self:
        expected = self.recovered / self.valid if self.valid else None
        if self.recovered > self.valid or self.rate != expected:
            raise ValueError("pose recovery counts/rate disagree")
        return self


class ConditionSummary(_Strict):
    fault_axis: Literal["roll", "pitch", "yaw", "x", "y", "z", "time"]
    fault_level: FiniteFloat
    validity: Validity
    pose: PoseSummary
    pixel_error_px: PixelStatistic
    edge_alignment_score: EdgeStatistic
    ground_contact_by_range: dict[str, GroundStatistic]
    pose_recovery: Recovery | None

    @model_validator(mode="after")
    def ranges(self) -> Self:
        if set(self.ground_contact_by_range) != set(RANGE_BINS):
            raise ValueError("summary must retain all declared range bins")
        return self


class RunSummary(_Strict):
    method: Literal["identity", "classical", "learned"]
    seed: Literal[17, 42, 73] | None
    checkpoint_sha256: Digest | None
    run_identity_sha256: Digest
    source_complete_sha256: Digest
    conditions: dict[str, ConditionSummary]

    @model_validator(mode="after")
    def coherent(self) -> Self:
        if (self.method == "learned" and (self.seed is None or self.checkpoint_sha256 is None)) or (
            self.method != "learned"
            and (self.seed is not None or self.checkpoint_sha256 is not None)
        ):
            raise ValueError("summary method/seed/checkpoint disagree")
        if set(self.conditions) != {
            condition_key(axis, level) for axis, level in condition_inventory(self.method)
        }:
            raise ValueError("summary condition inventory differs from method")
        if any(
            key != condition_key(condition.fault_axis, condition.fault_level)
            for key, condition in self.conditions.items()
        ):
            raise ValueError("summary condition label differs from fault metadata")
        if any(
            (condition.pose_recovery is None) != key.startswith("time:")
            for key, condition in self.conditions.items()
        ):
            raise ValueError("timing stress must not declare pose recovery")
        return self


class CalibrationSummaryV1(_Strict):
    schema_version: Literal["bev-calibration-summary/v1"]
    protocol_hash: Digest
    dataset_manifest_hash: Digest
    evidence_type: Literal["synthetic", "observed"]
    measurement_policy: dict[str, Any]
    measurement_identity_sha256: Digest
    runs: dict[str, RunSummary]
    document_sha256: Digest

    @model_validator(mode="after")
    def coherent(self) -> Self:
        expected_policy = EvaluationMeasurements(table_sha256={}, images={}).model_dump(
            mode="json", exclude={"images", "table_sha256"}
        )
        if self.measurement_policy != expected_policy:
            raise ValueError("unsupported safe summary measurement policy")
        keys = {(run.method, run.seed) for run in self.runs.values()}
        if len(keys) != len(self.runs) or (
            keys != FORMAL_RUNS
            and not (self.evidence_type == "synthetic" and keys == SYNTHETIC_PAIR)
        ):
            raise ValueError("safe summary run inventory is incomplete or duplicated")
        if any(
            label != (run.method if run.seed is None else f"learned-{run.seed}")
            for label, run in self.runs.items()
        ):
            raise ValueError("safe summary run label differs from identity")
        checkpoints = [
            run.checkpoint_sha256 for run in self.runs.values() if run.method == "learned"
        ]
        if len(checkpoints) != len(set(checkpoints)):
            raise ValueError("safe summary reused a learned checkpoint")
        if self.document_sha256 != digest(
            self.model_dump(mode="json", exclude={"document_sha256"})
        ):
            raise ValueError("safe summary document hash mismatch")
        return self


def load_safe_summary(path: Path) -> CalibrationSummaryV1:
    return CalibrationSummaryV1.model_validate_json(path.read_bytes())


def write_safe_summary(document: dict[str, Any], path: Path) -> Path:
    verified = CalibrationSummaryV1.model_validate(document)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(verified.model_dump(mode="json"), sort_keys=True, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return path
