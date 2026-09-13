"""Join independently expected identities, formal claims and verified raw inputs."""

import hashlib
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from bevcalib.analysis.formal_claims import publication_rows
from bevcalib.artifacts.documents import FormalIdentity, FormalRunSource
from bevcalib.artifacts.result_documents import digest
from bevcalib.artifacts.result_sets import load_run_set
from bevcalib.artifacts.study_expectations import (
    ExpectedStudy,
    validate_checkpoint_files,
    validate_expected_identity,
)
from bevcalib.artifacts.validator_runtime import ExpectedValidator, validate_validator_runtime
from bevcalib.report.evidence import load_display_evidence, original_documents
from bevcalib.report.scalar_binding import bind_scalar


class FaultStudyExpectations(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: Literal["bev-fault-study-validation-expectations/v1"]
    study: ExpectedStudy
    validator: ExpectedValidator


def validate_raw_sources(identity: FormalIdentity, directory: Path) -> int:
    """Re-use run/scene validators without recomputing estimates or bootstrap draws."""
    manifest_path = directory / "evaluation_manifest.json"
    manifest_bytes = manifest_path.read_bytes()
    manifest, runs = load_run_set(
        directory, synthetic_fixture=identity.evidence_type == "synthetic", require_all_seeds=True
    )
    if (
        manifest.manifest_sha256 != identity.dataset_manifest_hash
        or manifest.protocol_hash != identity.protocol_hash
        or manifest.dataset_version != identity.dataset_version
        or len(manifest.scenes) != identity.scene_count
        or sum(len(scene.sample_tokens) for scene in manifest.scenes) != identity.sample_count
        or digest(runs[0].marker.identity.measurements.model_dump(mode="json"))
        != identity.measurement_identity_sha256
    ):
        raise ValueError("raw source manifest or measurement identity differs from formal evidence")
    for run in runs:
        source = run.marker.identity
        actual = FormalRunSource(
            method=source.method,
            seed=source.seed,
            checkpoint_sha256=source.checkpoint_sha256,
            run_identity_sha256=source.run_id,
            source_complete_sha256=run.source_complete_sha256,
            producer_commit=source.producer.commit,
            producer_lock_sha256=source.producer.lock_sha256,
        )
        if actual != identity.source_runs[run.label]:
            raise ValueError(f"raw run identity differs from formal evidence: {run.label}")
    count = 0
    for run in runs:
        for scene in manifest.scenes:
            # Returned rows are immediately discarded; never retain a second scene.
            run.scene_rows(manifest, scene.scene_token)
            count += 1
    if manifest_path.read_bytes() != manifest_bytes:
        raise ValueError("raw manifest changed during validation")
    for run in runs:
        if hashlib.sha256((run.directory / "run_complete.json").read_bytes()).hexdigest() != (
            run.source_complete_sha256
        ):
            raise ValueError("raw run marker changed during validation")
    return count


def validate_fault_study(
    expected_path: Path,
    artifacts_dir: Path,
    claims_path: Path,
    raw_runs_dir: Path,
    checkpoints: Mapping[str, Path],
    *,
    repository_root: Path,
) -> dict[str, Any]:
    """Validate all required inputs; no training, inference, installs or output writes.

    This is input/provenance acceptance, not numerical reaggregation, installed-
    dependency attestation, checkpoint deserialization or an efficacy conclusion.
    """
    expected_bytes = expected_path.read_bytes()
    expected = FaultStudyExpectations.model_validate_json(expected_bytes)
    runtime = validate_validator_runtime(expected.validator)
    artifacts, registry_bytes, scalar_claims = load_display_evidence(
        claims_path, artifacts_dir, repository_root=repository_root
    )
    identity = artifacts.metrics.identity
    validate_expected_identity(identity, expected.study)
    checked_checkpoints = validate_checkpoint_files(expected.study, checkpoints)
    scalar_count = 0
    for row in publication_rows(artifacts):
        for _, pointer, value in row.cells:
            identifier = bind_scalar(
                row.document,
                pointer,
                value,
                getattr(artifacts, row.document).document_sha256,
                scalar_claims,
            )
            scalar_count += identifier is not None
    scene_count = validate_raw_sources(identity, raw_runs_dir)
    originals = original_documents(artifacts, registry_bytes, claims_path, artifacts_dir)
    if expected_path.read_bytes() != expected_bytes:
        raise ValueError("frozen expectations changed during validation")
    validate_validator_runtime(expected.validator)
    return {
        "schema_version": "bev-fault-study-input-validation/v1",
        "status": "validated-inputs",
        "evidence_type": identity.evidence_type,
        "expectations_sha256": hashlib.sha256(expected_bytes).hexdigest(),
        "validator": {key: value for key, value in runtime.items() if key != "executable"},
        "documents": {
            name: {
                "file_sha256": hashlib.sha256(raw).hexdigest(),
                "document_sha256": getattr(artifacts, name).document_sha256,
            }
            for name, raw in originals.items()
        },
        "claims_sha256": hashlib.sha256(registry_bytes).hexdigest(),
        "checkpoints": checked_checkpoints,
        "validated_scene_files": scene_count,
        "validated_scalar_bindings": scalar_count,
    }
