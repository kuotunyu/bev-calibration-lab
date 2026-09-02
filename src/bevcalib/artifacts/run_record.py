"""Immutable provenance record for one calibration experiment run."""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

PROVENANCE_ENV_VAR = "BEVCALIB_RUN_PROVENANCE"


def _parse_utc_z(value: str, field_name: str) -> datetime:
    """Parse a canonical `...Z` UTC timestamp, or say exactly why it is not one.

    Substituting an explicit `+00:00` offset makes the result unconditionally
    timezone-aware, so there is no separate awareness check here: it could never
    fail, and untestable defensive code is worse than none.
    """

    if not value.endswith("Z"):
        raise ValueError(f"{field_name} must end in Z")
    try:
        return datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise ValueError(f"{field_name} must be a valid ISO 8601 timestamp") from exc


class RunRecordV1(BaseModel):
    """Frozen identity, environment, lifecycle and artifact references for one run."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["bev-calibration-run/v1"]
    run_id: str = Field(min_length=1)
    commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    protocol_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    cohort_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    lock_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    hardware: dict[str, str]
    seed: int
    started_at_utc: str
    finished_at_utc: str | None
    status: Literal["running", "succeeded", "failed", "aborted"]
    artifacts: dict[str, str]

    @field_validator("started_at_utc", "finished_at_utc")
    @classmethod
    def validate_utc_timestamp(cls, value: str | None, info: object) -> str | None:
        """Require every present lifecycle timestamp to use canonical UTC Z form."""

        if value is None:
            return None
        _parse_utc_z(value, getattr(info, "field_name", "timestamp"))
        return value

    @model_validator(mode="after")
    def validate_lifecycle(self) -> Self:
        """Keep the running and terminal states logically consistent.

        A record found after a crash is the reason this is enforced at the type
        rather than at the writer: the writer is the thing that crashed.
        """

        if self.status == "running":
            if self.finished_at_utc is not None:
                raise ValueError("running run must not have finished_at_utc")
            return self

        if self.finished_at_utc is None:
            raise ValueError("terminal run requires finished_at_utc")
        if _parse_utc_z(self.finished_at_utc, "finished_at_utc") < _parse_utc_z(
            self.started_at_utc, "started_at_utc"
        ):
            raise ValueError("finished_at_utc must not precede started_at_utc")
        return self


class RunProvenance(BaseModel):
    """Environment facts a run cannot derive and must never invent."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    lock_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    hardware: dict[str, str] = Field(min_length=1)


def load_run_provenance() -> RunProvenance:
    """Read this run's commit, lock hash and hardware from the environment.

    Making it an explicit environment contract keeps the hardware record a
    deliberate act by whoever launched the job, and fails closed when it is
    absent rather than filling in something plausible.
    """

    raw_value = os.environ.get(PROVENANCE_ENV_VAR)
    if raw_value is None:
        raise ValueError(
            f"{PROVENANCE_ENV_VAR} must supply the commit, lock hash and hardware of this run"
        )
    try:
        document = json.loads(raw_value)
    except json.JSONDecodeError as error:
        raise ValueError(f"{PROVENANCE_ENV_VAR} must contain a JSON object") from error
    if not isinstance(document, dict):
        raise ValueError(f"{PROVENANCE_ENV_VAR} must contain a JSON object")
    return RunProvenance.model_validate(document)
