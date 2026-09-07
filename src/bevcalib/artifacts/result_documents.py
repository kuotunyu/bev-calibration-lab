"""Atomic V2 scene documents and verified complete-run inventories."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, model_validator

from bevcalib.artifacts.envelope import canonical_json_bytes
from bevcalib.artifacts.measurements import EvaluationMeasurements
from bevcalib.artifacts.results import CalibrationFaultModel, CalibrationResultV2
from bevcalib.artifacts.run_record import RunProvenance
from bevcalib.cohort.manifest import CohortManifestV2, Digest
from bevcalib.perturbations.schedule import FaultAxis, formal_single_axis_faults


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


class EvaluationIdentity(BaseModel):
    """Common authority for every row; seed/checkpoint combinations never share a run."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: Literal["bev-calibration-evaluation-identity/v2"] = (
        "bev-calibration-evaluation-identity/v2"
    )
    method: Literal["identity", "classical", "learned"]
    seed: int | None
    checkpoint_sha256: Digest | None
    protocol_hash: Digest
    dataset_manifest_hash: Digest
    dataset_version: str
    evidence_type: Literal["observed", "synthetic"]
    producer: RunProvenance
    measurements: EvaluationMeasurements

    @model_validator(mode="after")
    def coherent(self) -> Self:
        if self.method == "learned":
            if self.seed not in (17, 42, 73) or self.checkpoint_sha256 is None:
                raise ValueError("learned run requires an approved seed and checkpoint identity")
        elif self.seed is not None or self.checkpoint_sha256 is not None:
            raise ValueError("nonlearned method cannot declare learned seed/checkpoint")
        return self

    @property
    def run_id(self) -> str:
        return digest(self.model_dump(mode="json"))


class SceneResultDocumentV2(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: Literal["bev-calibration-scene/v2"]
    identity: EvaluationIdentity
    rows: tuple[CalibrationResultV2, ...]
    document_sha256: Digest


class RunCompleteV2(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: Literal["bev-calibration-complete/v2"]
    identity: EvaluationIdentity
    files: dict[str, Digest]
    document_sha256: Digest


def condition_inventory(method: str) -> tuple[tuple[FaultAxis, float], ...]:
    """Timing is identity-only stress; correction methods contain sixty SE(3) rows."""
    if method not in ("identity", "classical", "learned"):
        raise ValueError("unknown evaluation method")
    return tuple(
        condition
        for condition in formal_single_axis_faults()
        if method == "identity" or condition[0] != "time"
    )


def fault_for_condition(axis: FaultAxis, level: float) -> CalibrationFaultModel:
    if (axis, level) not in formal_single_axis_faults():
        raise ValueError("fault condition is not in the formal inventory")
    values = {
        name: level if name == axis else 0.0
        for name in ("roll", "pitch", "yaw", "x", "y", "z", "time")
    }
    return CalibrationFaultModel(
        rotation_rpy_deg=(values["roll"], values["pitch"], values["yaw"]),
        translation_xyz_m=(values["x"], values["y"], values["z"]),
        requested_time_offset_ms=int(values["time"]),
    )


def scene_filename(scene_token: str) -> str:
    return f"scene-{hashlib.sha256(scene_token.encode()).hexdigest()}.json"


def _validate_identity(identity: EvaluationIdentity, manifest: CohortManifestV2) -> None:
    verified = CohortManifestV2.model_validate(manifest.model_dump(mode="json"))
    EvaluationIdentity.model_validate(identity.model_dump(mode="json"))
    if (
        identity.dataset_manifest_hash != verified.manifest_sha256
        or identity.protocol_hash != verified.protocol_hash
        or identity.dataset_version != verified.dataset_version
    ):
        raise ValueError("run identity differs from verified cohort")
    if set(identity.measurements.images) != {
        token for scene in verified.scenes for token in scene.camera_sample_data_tokens
    }:
        raise ValueError("measurement image inventory differs from verified cohort")
    if identity.evidence_type == "observed" and (
        verified.role != "evaluation"
        or verified.dataset_version != "v1.0-trainval"
        or len(verified.scenes) != 30
        or any(scene.official_split != "val" for scene in verified.scenes)
    ):
        raise ValueError("observed results require a complete formal evaluation cohort")


def _validate_rows(
    rows: tuple[CalibrationResultV2, ...], manifest: CohortManifestV2, scene_token: str, method: str
) -> None:
    scene = next((scene for scene in manifest.scenes if scene.scene_token == scene_token), None)
    if scene is None:
        raise ValueError("scene is absent from manifest")
    expected = {
        (sample, axis, level)
        for sample in scene.sample_tokens
        for axis, level in condition_inventory(method)
    }
    if (
        len(rows) != len(expected)
        or {(row.sample_token, row.fault_axis, row.fault_level) for row in rows} != expected
    ):
        raise ValueError("scene row inventory is incomplete, duplicated or unexpected")
    for raw in rows:
        row = CalibrationResultV2.model_validate(raw.model_dump(mode="json"))
        index = scene.sample_tokens.index(row.sample_token)
        if (
            row.scene_token != scene_token
            or row.camera.token != scene.camera_sample_data_tokens[index]
            or row.camera.timestamp_us != scene.camera_timestamps[index]
            or row.fault != fault_for_condition(row.fault_axis, row.fault_level)
        ):
            raise ValueError("row provenance differs from manifest or fault condition")
        if row.fault_axis != "time" and (
            row.lidar is None
            or row.lidar.token != scene.lidar_sample_data_tokens[index]
            or row.lidar.timestamp_us != scene.lidar_timestamps[index]
        ):
            raise ValueError("metadata fault changed its nominal LiDAR observation")


def _atomic_document(path: Path, body: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing evidence: {path}")
    payload = body | {"document_sha256": digest(body)}
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with temporary.open("xb") as handle:
        handle.write(canonical_json_bytes(payload) + b"\n")
    temporary.replace(path)


def save_scene(
    directory: Path,
    identity: EvaluationIdentity,
    manifest: CohortManifestV2,
    rows: tuple[CalibrationResultV2, ...],
) -> Path:
    _validate_identity(identity, manifest)
    if not rows:
        raise ValueError("scene row inventory is empty")
    scene_token = rows[0].scene_token
    _validate_rows(rows, manifest, scene_token, identity.method)
    path = directory / scene_filename(scene_token)
    _atomic_document(
        path,
        {
            "schema_version": "bev-calibration-scene/v2",
            "identity": identity.model_dump(mode="json"),
            "rows": [row.model_dump(mode="json") for row in rows],
        },
    )
    return path


def _checked_body(path: Path) -> dict[str, Any]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("document_sha256") != digest(
        {key: value for key, value in document.items() if key != "document_sha256"}
    ):
        raise ValueError("document hash mismatch")
    return document


def _scene_inventory(
    directory: Path, identity: EvaluationIdentity, manifest: CohortManifestV2
) -> tuple[dict[str, str], tuple[CalibrationResultV2, ...]]:
    _validate_identity(identity, manifest)
    expected = {scene_filename(scene.scene_token): scene for scene in manifest.scenes}
    actual = {path.name for path in directory.glob("*.json") if path.name != "run_complete.json"}
    if actual != set(expected):
        raise ValueError("scene file inventory is missing or unexpected")
    hashes: dict[str, str] = {}
    rows: list[CalibrationResultV2] = []
    for name, scene in expected.items():
        path = directory / name
        document = SceneResultDocumentV2.model_validate(_checked_body(path))
        if document.identity != identity:
            raise ValueError("scene identity differs from run identity")
        _validate_rows(document.rows, manifest, scene.scene_token, identity.method)
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        rows.extend(document.rows)
    return hashes, tuple(rows)


def finalize_run(directory: Path, identity: EvaluationIdentity, manifest: CohortManifestV2) -> Path:
    files, _ = _scene_inventory(directory, identity, manifest)
    path = directory / "run_complete.json"
    _atomic_document(
        path,
        {
            "schema_version": "bev-calibration-complete/v2",
            "identity": identity.model_dump(mode="json"),
            "files": files,
        },
    )
    return path


def load_result_run(
    directory: Path, identity: EvaluationIdentity, manifest: CohortManifestV2
) -> tuple[CalibrationResultV2, ...]:
    marker = RunCompleteV2.model_validate(_checked_body(directory / "run_complete.json"))
    if marker.identity != identity:
        raise ValueError("complete marker differs from expected run identity")
    files, rows = _scene_inventory(directory, identity, manifest)
    if files != marker.files:
        raise ValueError("completed run file hash mismatch")
    return rows
