"""Resolve the cohort protocol and bind its transitive perturbation bytes."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

from bevcalib.artifacts.envelope import canonical_json_bytes


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class _Dataset(_StrictModel):
    version: Literal["v1.0-trainval"]
    keyframes_only: Literal[True]


class _Cohort(_StrictModel):
    development: Literal[100]
    calibration: Literal[20]
    evaluation: Literal[30]
    location_stratified: Literal[True]
    log_disjoint: Literal[True]


class _Bootstrap(_StrictModel):
    resamples: Literal[5000]
    seed: Literal[20260831]


class CohortProtocolV1(_StrictModel):
    """The pinned cohort contract; dataset version is explicit, never inferred."""

    schema_version: Literal["bev-calibration-protocol/v1"]
    dataset: _Dataset
    cohort: _Cohort
    perturbations: str = Field(min_length=1)
    bootstrap: _Bootstrap


@dataclass(frozen=True)
class ResolvedProtocol:
    """A portable identity; machine-specific paths are excluded from the hash."""

    protocol: CohortProtocolV1
    protocol_hash: str
    dataset_version: str
    perturbations_path: Path


def resolve_protocol(path: Path) -> ResolvedProtocol:
    """Validate YAML and resolve references relative to that YAML, never the cwd."""

    source = Path(path)
    protocol = CohortProtocolV1.model_validate(yaml.safe_load(source.read_text(encoding="utf-8")))
    perturbations_path = (source.parent / protocol.perturbations).resolve()
    identity = {
        "schema_version": "bev-calibration-protocol-identity/v1",
        "protocol": protocol.model_dump(mode="json"),
        "perturbations_sha256": hashlib.sha256(perturbations_path.read_bytes()).hexdigest(),
    }
    return ResolvedProtocol(
        protocol=protocol,
        protocol_hash=hashlib.sha256(canonical_json_bytes(identity)).hexdigest(),
        dataset_version=protocol.dataset.version,
        perturbations_path=perturbations_path,
    )
