"""Contracts for the run record: what a calibration run must prove about itself.

A result nobody can trace back to a commit, a protocol, a cohort and a machine is
an anecdote. These fields are the difference, and the lifecycle rules exist because
a half-written record is the one you find after a crash.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from pydantic import ValidationError

VALID: dict[str, Any] = {
    "schema_version": "bev-calibration-run/v1",
    "run_id": "identity-seed-17",
    "commit": "a" * 40,
    "config_sha256": "b" * 64,
    "protocol_sha256": "c" * 64,
    "cohort_manifest_sha256": "d" * 64,
    "lock_sha256": "e" * 64,
    "hardware": {"gpu": "none", "runtime": "pytest"},
    "seed": 17,
    "started_at_utc": "2026-09-02T00:00:00Z",
    "finished_at_utc": "2026-09-02T01:00:00Z",
    "status": "succeeded",
    "artifacts": {"results": "f" * 64},
}


def test_a_complete_run_record_is_accepted() -> None:
    """The happy path has to be reachable, or every other rule is untestable."""

    from bevcalib.artifacts.run_record import RunRecordV1

    record = RunRecordV1.model_validate(VALID)

    assert record.run_id == "identity-seed-17"
    assert record.status == "succeeded"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("schema_version", "bev-calibration-run/v2"),
        ("run_id", ""),
        ("commit", "abc"),
        ("config_sha256", "abc"),
        ("protocol_sha256", "abc"),
        ("cohort_manifest_sha256", "abc"),
        ("lock_sha256", "abc"),
        ("status", "finished"),
        ("started_at_utc", "2026-09-02T00:00:00"),
        ("finished_at_utc", "yesterday"),
        # Ends in Z, so it clears the cheap check and must still fail to parse.
        ("started_at_utc", "2026-13-45T00:00:00Z"),
    ],
)
def test_a_record_that_cannot_identify_its_run_is_rejected(field: str, value: str) -> None:
    """Every one of these would leave a published number unattributable."""

    from bevcalib.artifacts.run_record import RunRecordV1

    with pytest.raises(ValidationError):
        RunRecordV1.model_validate(VALID | {field: value})


def test_an_unknown_field_is_rejected() -> None:
    """A typo in a provenance key must not be silently accepted as a new fact."""

    from bevcalib.artifacts.run_record import RunRecordV1

    with pytest.raises(ValidationError):
        RunRecordV1.model_validate(VALID | {"gpu_hours": 3})


def test_a_running_record_has_not_finished() -> None:
    """A finish time on a running job is either a lie or a copy-paste error."""

    from bevcalib.artifacts.run_record import RunRecordV1

    running = RunRecordV1.model_validate(
        VALID | {"status": "running", "finished_at_utc": None, "artifacts": {}}
    )
    assert running.finished_at_utc is None

    with pytest.raises(ValidationError, match="running"):
        RunRecordV1.model_validate(VALID | {"status": "running"})


@pytest.mark.parametrize("status", ["succeeded", "failed", "aborted"])
def test_a_terminal_record_must_say_when_it_ended(status: str) -> None:
    """A crashed run still ended; refusing to record when is how duration gets invented."""

    from bevcalib.artifacts.run_record import RunRecordV1

    with pytest.raises(ValidationError, match="finished_at_utc"):
        RunRecordV1.model_validate(VALID | {"status": status, "finished_at_utc": None})


def test_a_run_cannot_finish_before_it_started() -> None:
    """Clock skew between machines is real, and a negative duration is not a result."""

    from bevcalib.artifacts.run_record import RunRecordV1

    with pytest.raises(ValidationError, match="precede"):
        RunRecordV1.model_validate(VALID | {"finished_at_utc": "2026-09-01T00:00:00Z"})


def test_provenance_is_read_from_the_environment_and_never_guessed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The machine a run used is a fact the process cannot infer, so it must be told."""

    from bevcalib.artifacts.run_record import PROVENANCE_ENV_VAR, load_run_provenance

    monkeypatch.setenv(
        PROVENANCE_ENV_VAR,
        json.dumps({"commit": "a" * 40, "lock_sha256": "b" * 64, "hardware": {"gpu": "A100-40GB"}}),
    )

    provenance = load_run_provenance()

    assert provenance.hardware == {"gpu": "A100-40GB"}


@pytest.mark.parametrize(
    "raw",
    [
        None,
        "not json",
        '"a string"',
        '{"commit": "short"}',
        # Well-formed and complete except for the hardware, which is the field a
        # process is most tempted to fill in for itself.
        json.dumps({"commit": "a" * 40, "lock_sha256": "b" * 64}),
    ],
)
def test_missing_or_malformed_provenance_fails_closed(
    monkeypatch: pytest.MonkeyPatch, raw: str | None
) -> None:
    """Inventing a plausible hardware string is the exact failure this prevents."""

    from bevcalib.artifacts.run_record import PROVENANCE_ENV_VAR, load_run_provenance

    if raw is None:
        monkeypatch.delenv(PROVENANCE_ENV_VAR, raising=False)
    else:
        monkeypatch.setenv(PROVENANCE_ENV_VAR, raw)

    with pytest.raises(ValueError):
        load_run_provenance()
