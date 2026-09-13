"""Compare validated study metadata and checkpoint bytes with external expectations."""

import hashlib
from collections.abc import Mapping
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from bevcalib.artifacts.documents import FormalIdentity
from bevcalib.cohort.manifest import Digest


class ExpectedStudy(BaseModel):
    """A separately frozen input, never inferred from the artifact being accepted."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: Literal["bev-fault-study-expectations/v1"]
    evidence_type: Literal["synthetic", "observed"]
    protocol_hash: Digest
    dataset_manifest_hash: Digest
    raw_producer_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    raw_producer_lock_sha256: Digest
    checkpoints: dict[str, Digest]

    @model_validator(mode="after")
    def fixed_checkpoints(self) -> Self:
        if (
            set(self.checkpoints) != {"learned-17", "learned-42", "learned-73"}
            or len(set(self.checkpoints.values())) != 3
        ):
            raise ValueError("expected study requires three distinct fixed-seed checkpoints")
        return self


def validate_expected_identity(identity: FormalIdentity, expected: ExpectedStudy) -> None:
    """Check an already schema-validated identity against independent expectations.

    The producer is the original raw-run producer, not the current analyzer HEAD.
    This comparison does not prove when/how the expected input was frozen, validate
    the current Python runtime, re-read raw scene shards, or establish efficacy.
    """
    for field in ("protocol_hash", "dataset_manifest_hash", "evidence_type"):
        if getattr(identity, field) != getattr(expected, field):
            raise ValueError(f"frozen study expectation mismatch: {field}")
    for label, source in identity.source_runs.items():
        if (source.producer_commit, source.producer_lock_sha256) != (
            expected.raw_producer_commit,
            expected.raw_producer_lock_sha256,
        ):
            raise ValueError(f"frozen study expectation mismatch: {label} raw producer")
        if (
            source.checkpoint_sha256 is not None
            and source.checkpoint_sha256 != expected.checkpoints[label]
        ):
            raise ValueError(f"frozen study expectation mismatch: {label} checkpoint")


def validate_checkpoint_files(
    expected: ExpectedStudy,
    paths: Mapping[str, Path],
) -> dict[str, str]:
    """Read all three checkpoint files as bytes, with no Torch or device allocation."""
    if set(paths) != set(expected.checkpoints):
        raise ValueError("checkpoint file inventory differs from frozen study expectations")
    actual = {}
    for label in sorted(paths):
        with paths[label].open("rb") as handle:
            checksum = hashlib.file_digest(handle, "sha256").hexdigest()
        if checksum != expected.checkpoints[label]:
            raise ValueError(f"checkpoint bytes differ from frozen study expectation: {label}")
        actual[label] = checksum
    return actual
