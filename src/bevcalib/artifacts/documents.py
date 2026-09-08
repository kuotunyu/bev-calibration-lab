"""The portable five-document formal result set and its shared source identities."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from bevcalib.analysis.policy import (
    COMPARISON_METRICS,
    COMPARISONS,
    ESTIMAND_DESCRIPTION,
    HIGHER_BETTER,
    METRIC_UNITS,
)
from bevcalib.artifacts.result_documents import condition_inventory, digest
from bevcalib.artifacts.result_sets import FORMAL_RUNS
from bevcalib.artifacts.statistics import Estimate, PairedEstimate, Support
from bevcalib.artifacts.summary import MagnitudeStatistic, Statistic
from bevcalib.cohort.manifest import Digest
from bevcalib.metrics.summary import condition_key
from bevcalib.perturbations.schedule import TIMING_OFFSET_MS


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class FormalRunSource(Strict):
    method: Literal["identity", "classical", "learned"]
    seed: Literal[17, 42, 73] | None
    checkpoint_sha256: Digest | None
    run_identity_sha256: Digest
    source_complete_sha256: Digest
    producer_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    producer_lock_sha256: Digest


class FormalIdentity(Strict):
    protocol_hash: Digest
    dataset_manifest_hash: Digest
    dataset_version: Literal["v1.0-trainval"]
    evidence_type: Literal["synthetic", "observed"]
    measurement_identity_sha256: Digest
    source_runs: dict[str, FormalRunSource]
    scene_count: int = Field(gt=0)
    sample_count: int = Field(gt=0)
    estimands: dict[str, str]
    units: dict[str, str]

    @model_validator(mode="after")
    def coherent(self) -> Self:
        if self.estimands != ESTIMAND_DESCRIPTION or self.units != METRIC_UNITS:
            raise ValueError("formal analysis policy differs from the predeclared estimands")
        if self.evidence_type == "observed" and self.scene_count != 30:
            raise ValueError("formal observed source inventory requires thirty scenes")
        keys = {(run.method, run.seed) for run in self.source_runs.values()}
        if keys != FORMAL_RUNS or len(self.source_runs) != len(FORMAL_RUNS):
            raise ValueError("formal source run inventory differs")
        checkpoints = []
        producers = set()
        for label, run in self.source_runs.items():
            expected = run.method if run.seed is None else f"learned-{run.seed}"
            if label != expected or (run.checkpoint_sha256 is None) != (run.method != "learned"):
                raise ValueError("formal source run label/checkpoint differs")
            if run.checkpoint_sha256 is not None:
                checkpoints.append(run.checkpoint_sha256)
            producers.add((run.producer_commit, run.producer_lock_sha256))
        if len(set(checkpoints)) != 3 or len(producers) != 1:
            raise ValueError("formal source checkpoint or producer identities differ")
        return self


class FormalDocument(Strict):
    identity: FormalIdentity
    document_sha256: Digest

    @model_validator(mode="after")
    def checked_hash(self) -> Self:
        if self.document_sha256 != digest(
            self.model_dump(mode="json", exclude={"document_sha256"})
        ):
            raise ValueError("formal document hash mismatch")
        return self


def conditions_for(method: str) -> set[str]:
    return {condition_key(axis, level) for axis, level in condition_inventory(method)}


def check_runs(
    runs: dict[str, Any], identity: FormalIdentity, *, extrinsic_only: bool = False
) -> None:
    if set(runs) != set(identity.source_runs):
        raise ValueError("formal run inventory differs")
    for label, conditions in runs.items():
        method = "classical" if extrinsic_only else identity.source_runs[label].method
        if set(conditions) != conditions_for(method):
            raise ValueError("formal condition inventory differs")


def check_support(support: Support, identity: FormalIdentity) -> None:
    if (
        support.total_frames != identity.sample_count
        or support.total_scenes != identity.scene_count
    ):
        raise ValueError("formal support denominator differs from complete frozen inventory")


def check_metric_value(metric: str, value: float | None) -> None:
    if value is None or "_bias_" in metric:
        return
    if metric == "edge_score_px":
        allowed = value <= 0
    elif metric == "recovery_rate_pct":
        allowed = 0 <= value <= 100
    else:
        allowed = value >= 0
    if not allowed:
        raise ValueError("formal metric value violates its physical sign or rate units")


def check_improvement(metric: str, estimate: PairedEstimate) -> None:
    if estimate.before is None or estimate.after is None or estimate.improvement is None:
        return
    direction = -1 if metric in HIGHER_BETTER else 1
    expected = direction * (estimate.before - estimate.after)
    # Means of differences and differences of means have different rounding order.
    # This bound is numerical (64 ulps at operand scale), never a physical tolerance.
    tolerance = 64 * math.ulp(max(abs(estimate.before), abs(estimate.after), 1.0))
    if abs(estimate.improvement - expected) > tolerance:
        raise ValueError("paired improvement disagrees with before/after and metric direction")


class MetricsDocument(FormalDocument):
    schema_version: Literal["bev-calibration-metrics/v1"]
    runs: dict[str, dict[str, dict[str, Estimate]]]

    @model_validator(mode="after")
    def inventory(self) -> Self:
        check_runs(self.runs, self.identity)
        for conditions in self.runs.values():
            for measurements in conditions.values():
                if set(measurements) != set(METRIC_UNITS):
                    raise ValueError("formal metric inventory differs")
                for metric, estimate in measurements.items():
                    check_support(estimate.support, self.identity)
                    check_metric_value(metric, estimate.value)
        return self


class IntervalsDocument(FormalDocument):
    schema_version: Literal["bev-calibration-intervals/v1"]
    comparisons: dict[str, dict[str, dict[str, PairedEstimate]]]

    @model_validator(mode="after")
    def inventory(self) -> Self:
        if set(self.comparisons) != set(COMPARISONS):
            raise ValueError("formal comparison inventory differs")
        for conditions in self.comparisons.values():
            if set(conditions) != conditions_for("classical"):
                raise ValueError("formal comparison condition inventory differs")
            for measurements in conditions.values():
                if set(measurements) != set(COMPARISON_METRICS):
                    raise ValueError("formal comparison metric inventory differs")
                for metric, estimate in measurements.items():
                    check_support(estimate.support, self.identity)
                    check_metric_value(metric, estimate.before)
                    check_metric_value(metric, estimate.after)
                    check_improvement(metric, estimate)
        return self


class RecoveryDocument(FormalDocument):
    schema_version: Literal["bev-calibration-recovery/v1"]
    runs: dict[str, dict[str, Estimate]]
    comparisons: dict[str, dict[str, PairedEstimate]]

    @model_validator(mode="after")
    def inventory(self) -> Self:
        check_runs(self.runs, self.identity, extrinsic_only=True)
        if set(self.comparisons) != set(COMPARISONS) or any(
            set(value) != conditions_for("classical") for value in self.comparisons.values()
        ):
            raise ValueError("recovery comparison inventory differs")
        return self


class TimingSummary(Strict):
    requested_offset_ms: int
    total: int = Field(gt=0)
    selected: int = Field(ge=0)
    valid: int = Field(ge=0)
    invalid: int = Field(ge=0)
    valid_fraction: float = Field(ge=0, le=1, allow_inf_nan=False)
    reasons: dict[str, int]
    realized_offset_ms: Statistic
    absolute_error_ms: MagnitudeStatistic
    metrics: dict[str, Estimate]

    @model_validator(mode="after")
    def coherent(self) -> Self:
        if (
            self.valid + self.invalid != self.total
            or self.valid_fraction != self.valid / self.total
            or not self.valid <= self.selected <= self.total
            or sum(self.reasons.values()) != self.total
            or self.reasons.get("valid", 0) != self.valid
            or self.reasons.get("no_available_lidar", 0) != self.total - self.selected
            or any(
                reason not in ("valid", "outside_tolerance", "no_available_lidar") or count < 0
                for reason, count in self.reasons.items()
            )
            or self.realized_offset_ms.count != self.selected
            or self.absolute_error_ms.count != self.selected
        ):
            raise ValueError("timing selection counts/rate/reasons differ")
        return self


class TimingDocument(FormalDocument):
    schema_version: Literal["bev-calibration-timing/v1"]
    offsets: dict[str, TimingSummary]

    @model_validator(mode="after")
    def inventory(self) -> Self:
        if set(self.offsets) != {str(value) for value in TIMING_OFFSET_MS}:
            raise ValueError("formal timing offset inventory differs")
        if any(
            str(value.requested_offset_ms) != key or value.total != self.identity.sample_count
            for key, value in self.offsets.items()
        ):
            raise ValueError("timing request or denominator differs")
        return self


class ConditionCounts(Strict):
    total: int = Field(gt=0)
    valid: int = Field(ge=0)
    invalid: int = Field(ge=0)
    reasons: dict[str, int]
    projection_input_points: int = Field(ge=0)
    within_row_pixel_error_count: int = Field(ge=0)
    operators: dict[str, Support]

    @model_validator(mode="after")
    def coherent(self) -> Self:
        if (
            self.valid + self.invalid != self.total
            or sum(self.reasons.values()) != self.invalid
            or any(count < 0 for count in self.reasons.values())
            or self.within_row_pixel_error_count > self.projection_input_points
        ):
            raise ValueError("operator/global validity counts differ")
        return self


class ExclusionsDocument(FormalDocument):
    schema_version: Literal["bev-calibration-exclusions/v1"]
    runs: dict[str, dict[str, ConditionCounts]]
    comparisons: dict[str, dict[str, dict[str, Support]]]

    @model_validator(mode="after")
    def inventory(self) -> Self:
        check_runs(self.runs, self.identity)
        return self


class FormalArtifactSet(Strict):
    metrics: MetricsDocument
    intervals: IntervalsDocument
    recovery: RecoveryDocument
    timing: TimingDocument
    exclusions: ExclusionsDocument

    @model_validator(mode="after")
    def coherent(self) -> Self:
        documents = (self.metrics, self.intervals, self.recovery, self.timing, self.exclusions)
        if any(document.identity != self.metrics.identity for document in documents):
            raise ValueError("five-document source identity differs")
        expected_recovery = {
            label: {
                key: values["recovery_rate_pct"]
                for key, values in conditions.items()
                if not key.startswith("time:")
            }
            for label, conditions in self.metrics.runs.items()
        }
        expected_comparisons = {
            label: {key: values["recovery_rate_pct"] for key, values in conditions.items()}
            for label, conditions in self.intervals.comparisons.items()
        }
        if (
            self.recovery.runs != expected_recovery
            or self.recovery.comparisons != expected_comparisons
        ):
            raise ValueError("recovery differs from metrics/intervals")
        if any(
            value.metrics != self.metrics.runs["identity"][f"time:{offset}"]
            for offset, value in self.timing.offsets.items()
        ):
            raise ValueError("timing stress differs from identity metrics")
        for label, conditions in self.metrics.runs.items():
            for key, values in conditions.items():
                counts = self.exclusions.runs[label][key]
                if counts.total != self.metrics.identity.sample_count or counts.operators != {
                    metric: value.support for metric, value in values.items()
                }:
                    raise ValueError("exclusions differ from metric support")
        expected_support = {
            label: {
                key: {metric: value.support for metric, value in values.items()}
                for key, values in conditions.items()
            }
            for label, conditions in self.intervals.comparisons.items()
        }
        if self.exclusions.comparisons != expected_support:
            raise ValueError("exclusions differ from paired support")
        return self


DOCUMENT_TYPES: dict[str, type[FormalDocument]] = {
    "metrics": MetricsDocument,
    "intervals": IntervalsDocument,
    "recovery": RecoveryDocument,
    "timing": TimingDocument,
    "exclusions": ExclusionsDocument,
}


def load_formal_artifact_set(directory: Path) -> FormalArtifactSet:
    if {path.name for path in directory.glob("*.json")} != {
        f"{name}.json" for name in DOCUMENT_TYPES
    }:
        raise ValueError("formal five-document file inventory differs")
    return FormalArtifactSet.model_validate(
        {
            name: cls.model_validate_json((directory / f"{name}.json").read_bytes())
            for name, cls in DOCUMENT_TYPES.items()
        }
    )
