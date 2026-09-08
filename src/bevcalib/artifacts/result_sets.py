"""Shared complete-run identities and bounded, verified scene reads."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from bevcalib.artifacts.result_documents import (
    RunCompleteV2,
    _validate_identity,
    digest,
    load_result_scene,
    scene_filename,
)
from bevcalib.artifacts.results import CalibrationResultV2
from bevcalib.cohort.manifest import CohortManifestV2, load_formal_manifest, load_manifest

FORMAL_RUNS = {
    ("identity", None),
    ("classical", None),
    ("learned", 17),
    ("learned", 42),
    ("learned", 73),
}
SYNTHETIC_PAIR = {("identity", None), ("classical", None)}


@dataclass(frozen=True)
class VerifiedRun:
    directory: Path
    marker: RunCompleteV2
    source_complete_sha256: str

    @property
    def label(self) -> str:
        identity = self.marker.identity
        return identity.method if identity.seed is None else f"learned-{identity.seed}"

    def scene_rows(
        self, manifest: CohortManifestV2, scene_token: str
    ) -> tuple[CalibrationResultV2, ...]:
        name = scene_filename(scene_token)
        return load_result_scene(
            self.directory / name,
            self.marker.identity,
            manifest,
            scene_token,
            self.marker.files[name],
        )


def load_run_set(
    artifacts_dir: Path, *, synthetic_fixture: bool = False, require_all_seeds: bool = False
) -> tuple[CohortManifestV2, tuple[VerifiedRun, ...]]:
    manifest_path = artifacts_dir / "evaluation_manifest.json"
    manifest = load_manifest(manifest_path)
    if not isinstance(manifest, CohortManifestV2) or manifest.role != "evaluation":
        raise ValueError("summary requires a verified V2 evaluation manifest")
    if not synthetic_fixture:
        manifest = load_formal_manifest(
            manifest_path,
            expected_role="evaluation",
            protocol_hash=manifest.protocol_hash,
            dataset_version="v1.0-trainval",
        )
    runs = []
    for directory in sorted(path for path in artifacts_dir.iterdir() if path.is_dir()):
        payload = (directory / "run_complete.json").read_bytes()
        marker = RunCompleteV2.model_validate_json(payload)
        if marker.document_sha256 != digest(
            marker.model_dump(mode="json", exclude={"document_sha256"})
        ):
            raise ValueError("document hash mismatch")
        runs.append(VerifiedRun(directory, marker, hashlib.sha256(payload).hexdigest()))
    keys = [(run.marker.identity.method, run.marker.identity.seed) for run in runs]
    if len(keys) != len(set(keys)) or (
        set(keys) != FORMAL_RUNS
        and not (synthetic_fixture and not require_all_seeds and set(keys) == SYNTHETIC_PAIR)
    ):
        raise ValueError(
            "complete run inventory must contain identity/classical and all three learned seeds; synthetic pair is explicit only"
        )
    checkpoints = [
        run.marker.identity.checkpoint_sha256
        for run in runs
        if run.marker.identity.method == "learned"
    ]
    if len(checkpoints) != len(set(checkpoints)):
        raise ValueError("learned seeds must have distinct checkpoint identities")
    measurements = runs[0].marker.identity.measurements
    expected = {scene_filename(scene.scene_token) for scene in manifest.scenes}
    for run in runs:
        identity = run.marker.identity
        if identity.measurements != measurements or identity.evidence_type != (
            "synthetic" if synthetic_fixture else "observed"
        ):
            raise ValueError("run measurement identity or evidence type differs")
        _validate_identity(identity, manifest)
        actual = {
            path.name for path in run.directory.glob("*.json") if path.name != "run_complete.json"
        }
        if actual != expected or set(run.marker.files) != expected:
            raise ValueError("scene file inventory is missing or unexpected")
    return manifest, tuple(runs)
