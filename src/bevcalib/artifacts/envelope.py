"""Strict cross-repository portfolio artifact envelope.

Implemented here from the shared specification rather than imported from another
project. Three independent implementations that agree on the same bytes are
evidence that the specification is unambiguous; one implementation imported three
times would only be evidence that it imports.
"""

from __future__ import annotations

import hashlib
import json
import secrets
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class PortfolioArtifactEnvelopeV1(BaseModel):
    """Immutable provenance wrapper around one artifact payload."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["portfolio-artifact-envelope/v1"]
    producer_repository: Literal[
        "driving-risk-metrics",
        "bev-calibration-lab",
        "perception-error-to-aeb",
    ]
    producer_release: str = Field(pattern=r"^v[0-9]+\.[0-9]+\.[0-9]+$")
    producer_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    artifact_type: str = Field(min_length=1)
    protocol_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    dataset_manifest_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    created_at_utc: str
    payload_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    payload: dict[str, Any]

    @field_validator("created_at_utc")
    @classmethod
    def validate_created_at_utc(cls, value: str) -> str:
        """Require an unambiguous, parseable UTC timestamp in canonical Z form."""

        # Requiring the Z and then parsing with an explicit +00:00 offset is what
        # makes the result unambiguous. There is deliberately no separate
        # "is it timezone-aware" check: after this substitution it always is, so
        # such a check could never fail and could never be tested.
        if not value.endswith("Z"):
            raise ValueError("created_at_utc must end in Z")
        try:
            datetime.fromisoformat(value[:-1] + "+00:00")
        except ValueError as exc:
            raise ValueError("created_at_utc must be a valid ISO 8601 timestamp") from exc
        return value


def canonical_json_bytes(value: object) -> bytes:
    """Serialize a JSON-compatible value to stable, compact UTF-8 bytes.

    Sorted keys and compact separators are what let two producers on two machines
    hash the same payload and get the same digest. `allow_nan=False` is not
    tidiness: NaN is not JSON, and a payload containing one would round-trip
    through some readers and fail in others.
    """

    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def verify_envelope(path: Path, expected_artifact_type: str) -> PortfolioArtifactEnvelopeV1:
    """Load one envelope and fail closed on type or canonical payload drift.

    The digest is recomputed rather than trusted, because a declared hash only
    proves what the producer believed at write time.
    """

    raw_value = json.loads(path.read_text(encoding="utf-8"))
    envelope = PortfolioArtifactEnvelopeV1.model_validate(raw_value)
    if envelope.artifact_type != expected_artifact_type:
        raise ValueError(
            "unexpected artifact type: "
            f"expected {expected_artifact_type!r}, got {envelope.artifact_type!r}"
        )

    actual_sha256 = hashlib.sha256(canonical_json_bytes(envelope.payload)).hexdigest()
    if not secrets.compare_digest(actual_sha256, envelope.payload_sha256):
        raise ValueError("payload SHA-256 mismatch")
    return envelope
