"""Orchestrating one learned-corrector run, with the checks that keep it honest."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, Protocol

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

from bevcalib.artifacts.envelope import canonical_json_bytes
from bevcalib.artifacts.run_record import RunRecordV1, load_run_provenance
from bevcalib.cohort.manifest import CohortManifestV2, load_formal_manifest
from bevcalib.cohort.protocol import resolve_protocol
from bevcalib.cohort.records import validate_records

# The protocol reserves exactly twenty distinct-log train scenes for choosing the
# checkpoint. A shorter manifest would select on less evidence while looking the same.
CHECKPOINT_FILENAME = "selected_checkpoint.pt"
RUN_RECORD_FILENAME = "run_record.json"


class TrainingBackend(Protocol):
    """The framework boundary. No tensor or optimiser code lives in this engine."""

    def seed_all(self, seed: int) -> None: ...

    def create_model_and_optimizer(self, config: Mapping[str, object]) -> tuple[object, object]: ...

    def run_epoch(
        self, model: object, optimizer: object, development_manifest: Path, epoch: int
    ) -> float: ...

    def evaluate_loss(self, model: object, calibration_manifest: Path) -> float: ...

    def save_checkpoint(self, model: object, path: Path, metadata: Mapping[str, object]) -> str: ...


@dataclass(frozen=True)
class CalibrationTrainingResult:
    """Which epoch was chosen, on what evidence, and where the proof was written."""

    selected_epoch: int
    checkpoint_path: Path
    calibration_loss: float
    run_record_path: Path


class CorrectorConfigV1(BaseModel):
    """Everything that could change a result, pinned before the first formal run."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["bev-corrector-config/v1"]
    protocol_path: str = Field(min_length=1)
    architecture: Literal["convnextv2_tiny"]
    input_channels: Literal[5]
    input_height: int = Field(gt=0)
    input_width: int = Field(gt=0)
    epochs: int = Field(gt=0)
    batch_size: int = Field(gt=0)
    optimizer: Literal["adamw"]
    learning_rate: float = Field(gt=0.0)
    weight_decay: float = Field(ge=0.0)
    schedule: Literal["cosine"]
    warmup_epochs: int = Field(ge=0)
    seeds: tuple[int, ...] = Field(min_length=1)
    rotation_scale_deg: float = Field(gt=0.0)
    translation_scale_m: float = Field(gt=0.0)
    huber_delta: float = Field(gt=0.0)

    @field_validator("input_height", "input_width")
    @classmethod
    def validate_stride(cls, value: int) -> int:
        if value % 32 != 0:
            raise ValueError(f"{value} is not a multiple of 32; ConvNeXtV2 would pad silently")
        return value


def load_corrector_config(path: Path) -> CorrectorConfigV1:
    """Load and strictly validate one corrector configuration."""

    return CorrectorConfigV1.model_validate(yaml.safe_load(Path(path).read_text(encoding="utf-8")))


def _require_disjoint(development: CohortManifestV2, calibration: CohortManifestV2) -> None:
    """Refuse any overlap between the two cohorts, log overlap included.

    The log check is the one that matters. Two scenes from one log are the same
    road minutes apart, not independent samples, so a shared log means the
    checkpoint is chosen on something the model has effectively already seen and
    the selection stops being a check on generalisation at all.
    """

    for name, left, right in (
        (
            "scene",
            {scene.scene_token for scene in development.scenes},
            {scene.scene_token for scene in calibration.scenes},
        ),
        (
            "log",
            {scene.log_token for scene in development.scenes},
            {scene.log_token for scene in calibration.scenes},
        ),
        (
            "sample",
            {token for scene in development.scenes for token in scene.sample_tokens},
            {token for scene in calibration.scenes for token in scene.sample_tokens},
        ),
        (
            "camera sample-data",
            {token for scene in development.scenes for token in scene.camera_sample_data_tokens},
            {token for scene in calibration.scenes for token in scene.camera_sample_data_tokens},
        ),
        (
            "LiDAR sample-data",
            {token for scene in development.scenes for token in scene.lidar_sample_data_tokens},
            {token for scene in calibration.scenes for token in scene.lidar_sample_data_tokens},
        ),
    ):
        shared = sorted(left & right)
        if shared:
            raise ValueError(
                f"the development and calibration cohorts share a {name}: {shared[:5]}"
            )
    # Preserve the producer's combined identifier namespace across both roles,
    # including camera/LiDAR and sample/sensor collisions in different fields.
    validate_records(development.scenes + calibration.scenes)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _utc_now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def train_learned_corrector(
    config: Path,
    development_manifest: Path,
    calibration_manifest: Path,
    output_dir: Path,
    seed: int,
    *,
    backend: TrainingBackend,
) -> CalibrationTrainingResult:
    """Train the corrector and keep the epoch the calibration cohort liked best.

    Selection reads the calibration cohort and NOTHING else. The training loss
    falls every epoch by construction, so selecting on it would simply pick the
    last epoch and call it a choice; and the evaluation cohort is untouchable,
    because a checkpoint chosen on it makes every later number a report on a
    decision rather than a measurement.

    P1 removed the choice entirely, keeping only the final step, because it had no
    cohort to spend on selection: its calibration split was already committed to
    temperature fitting. P2 has twenty distinct-log scenes reserved for exactly
    this, so the choice is kept and made where it is free.
    """

    config_path = Path(config)
    loaded = load_corrector_config(config_path)
    if seed not in loaded.seeds:
        raise ValueError(f"seed {seed} is not one of the approved seeds {list(loaded.seeds)}")

    protocol_path = (config_path.parent / loaded.protocol_path).resolve()
    protocol = resolve_protocol(protocol_path)
    protocol_hash = protocol.protocol_hash
    dataset_version = protocol.dataset_version
    development = load_formal_manifest(
        development_manifest,
        expected_role="development",
        protocol_hash=protocol_hash,
        dataset_version=dataset_version,
    )
    calibration = load_formal_manifest(
        calibration_manifest,
        expected_role="calibration",
        protocol_hash=protocol_hash,
        dataset_version=dataset_version,
    )
    _require_disjoint(development, calibration)
    directory = Path(output_dir)
    record_path = directory / RUN_RECORD_FILENAME
    if record_path.exists():
        raise FileExistsError(f"a run record already exists here: {record_path}")
    directory.mkdir(parents=True, exist_ok=True)

    provenance = load_run_provenance()
    identity: dict[str, Any] = {
        "run_id": f"{loaded.architecture}-seed-{seed}",
        "config_sha256": _sha256_file(config_path),
        "protocol_sha256": protocol_hash,
        "cohort_manifest_sha256": hashlib.sha256(
            canonical_json_bytes(
                {
                    "development": development.manifest_sha256,
                    "calibration": calibration.manifest_sha256,
                }
            )
        ).hexdigest(),
    }
    started_at = _utc_now()

    def write_record(status: str, artifacts: dict[str, str]) -> None:
        record = RunRecordV1(
            schema_version="bev-calibration-run/v1",
            commit=provenance.commit,
            lock_sha256=provenance.lock_sha256,
            hardware=provenance.hardware,
            seed=seed,
            started_at_utc=started_at,
            finished_at_utc=_utc_now(),
            status=status,  # type: ignore[arg-type]
            artifacts=artifacts,
            **identity,
        )
        record_path.write_text(
            json.dumps(record.model_dump(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    checkpoint_path = directory / CHECKPOINT_FILENAME
    try:
        backend.seed_all(seed)
        model, optimizer = backend.create_model_and_optimizer(loaded.model_dump())

        selected_epoch, best_loss, reported_digest = 0, math.inf, ""
        for epoch in range(1, loaded.epochs + 1):
            backend.run_epoch(model, optimizer, Path(development_manifest), epoch)
            calibration_loss = float(backend.evaluate_loss(model, Path(calibration_manifest)))
            if not math.isfinite(calibration_loss):
                raise ValueError(
                    f"the calibration loss at epoch {epoch} is {calibration_loss}, "
                    "which cannot select anything"
                )
            # Strictly better only, so a tie keeps the earliest epoch: a later one
            # that is no better is more training for no measured gain.
            if calibration_loss < best_loss:
                selected_epoch, best_loss = epoch, calibration_loss
                reported_digest = backend.save_checkpoint(
                    model,
                    checkpoint_path,
                    {"run_id": identity["run_id"], "epoch": epoch, "calibration_loss": best_loss},
                )

        actual_digest = _sha256_file(checkpoint_path)
        if actual_digest != reported_digest:
            raise ValueError(
                "the selected checkpoint on disk does not match the digest the backend "
                f"reported: {actual_digest} against {reported_digest}"
            )
    except Exception:
        write_record("failed", {})
        raise

    write_record("succeeded", {"selected_checkpoint": actual_digest})
    return CalibrationTrainingResult(
        selected_epoch=selected_epoch,
        checkpoint_path=checkpoint_path,
        calibration_loss=best_loss,
        run_record_path=record_path,
    )
