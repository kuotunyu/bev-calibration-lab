"""One authoritative full-provenance cohort contract, plus explicit legacy reading."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from bevcalib.artifacts.envelope import canonical_json_bytes

from .records import LOCATIONS, Location, SceneRecord, validate_records

Role = Literal["development", "calibration", "evaluation"]
ROLE_COUNTS: dict[Role, int] = {"development": 100, "calibration": 20, "evaluation": 30}
Digest = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


class CohortSceneV1(BaseModel):
    """Legacy partial provenance. Missing sensor metadata is never synthesized."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    scene_token: str = Field(min_length=1)
    log_token: str = Field(min_length=1)
    sample_tokens: tuple[str, ...]


class CohortManifestV1(BaseModel):
    """Readable historical shape; ineligible at every formal boundary."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: Literal["bev-calibration-cohort/v1"]
    role: Role
    scenes: tuple[CohortSceneV1, ...]
    manifest_sha256: Digest


class StratumAllocation(BaseModel):
    """Requested quota, original availability and actual selection for one location."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    location: Location
    requested: int = Field(ge=0)
    available: int = Field(ge=0)
    available_logs: int = Field(ge=0)
    selected: int = Field(ge=0)
    shortage_reason: Literal["insufficient_scenes", "joint_log_capacity"] | None


def manifest_hash(document: Mapping[str, Any]) -> str:
    """Hash the entire canonical body, excluding only its own digest field."""

    return hashlib.sha256(
        canonical_json_bytes({k: v for k, v in document.items() if k != "manifest_sha256"})
    ).hexdigest()


class CohortManifestV2(BaseModel):
    """Full observations and hashed allocation diagnostics, including underfill."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: Literal["bev-calibration-cohort/v2"]
    dataset_version: str = Field(min_length=1)
    protocol_hash: Digest
    role: Role
    scenes: tuple[SceneRecord, ...]
    allocation: tuple[StratumAllocation, ...]
    manifest_sha256: Digest

    @model_validator(mode="after")
    def validate_body(self) -> Self:
        if manifest_hash(self.model_dump(mode="json")) != self.manifest_sha256:
            raise ValueError("manifest hash does not match canonical body")
        if len(validate_records(self.scenes)) != len(self.scenes):
            raise ValueError("duplicate scene in manifest")
        if tuple(d.location for d in self.allocation) != LOCATIONS:
            raise ValueError("allocation must contain every location in canonical order")
        if sum(d.requested for d in self.allocation) != ROLE_COUNTS[self.role]:
            raise ValueError("allocation requested count does not match role")
        for diagnostic in self.allocation:
            selected = sum(s.location == diagnostic.location for s in self.scenes)
            if (
                diagnostic.selected != selected
                or selected > diagnostic.requested
                or selected > diagnostic.available
                or diagnostic.available_logs
                < len({s.log_token for s in self.scenes if s.location == diagnostic.location})
                or diagnostic.available_logs > diagnostic.available
            ):
                raise ValueError("allocation counts disagree with selected scenes or availability")
            if (selected < diagnostic.requested) != (diagnostic.shortage_reason is not None):
                raise ValueError(
                    "allocation shortage reason must describe exactly the underfilled strata"
                )
        return self


def save_manifest(manifest: CohortManifestV2, path: Path) -> None:
    """Revalidate before an exclusive write; sorted JSON always ends with one LF."""

    verified = CohortManifestV2.model_validate(manifest.model_dump(mode="json"))
    payload = json.dumps(verified.model_dump(mode="json"), sort_keys=True, indent=2) + "\n"
    with Path(path).open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(payload)


def load_manifest(path: Path) -> CohortManifestV2 | CohortManifestV1:
    """Read V2 with integrity verification, or V1 explicitly as legacy-only data."""

    document = json.loads(Path(path).read_text(encoding="utf-8"))
    if document.get("schema_version") == "bev-calibration-cohort/v1":
        legacy = CohortManifestV1.model_validate(document)
        if manifest_hash(legacy.model_dump(mode="json")) != legacy.manifest_sha256:
            raise ValueError("legacy manifest hash does not match canonical body")
        return legacy
    return CohortManifestV2.model_validate(document)


def load_formal_manifest(
    path: Path, *, expected_role: str, protocol_hash: str, dataset_version: str
) -> CohortManifestV2:
    """Verify identity, completeness and split/log rules before any backend work."""

    manifest = load_manifest(path)
    if isinstance(manifest, CohortManifestV1):
        raise ValueError("legacy V1 cohort lacks full provenance; formal input requires V2")
    if manifest.role != expected_role:
        raise ValueError(
            f"the {expected_role} manifest declares role {manifest.role!r}; wrong formal role"
        )
    if manifest.protocol_hash != protocol_hash:
        raise ValueError("cohort protocol hash does not match resolved protocol")
    if manifest.dataset_version != dataset_version:
        raise ValueError("cohort dataset version does not match resolved protocol")
    count = ROLE_COUNTS[manifest.role]
    if len(manifest.scenes) != count:
        raise ValueError(
            f"the {manifest.role} cohort must hold exactly {count} scenes, got {len(manifest.scenes)}"
        )
    expected_split = "val" if manifest.role == "evaluation" else "train"
    if any(s.official_split != expected_split for s in manifest.scenes):
        raise ValueError("cohort scenes do not preserve the required official split")
    if manifest.role == "calibration" and len({s.log_token for s in manifest.scenes}) != count:
        raise ValueError("calibration scenes must come from 20 distinct logs")
    return manifest
