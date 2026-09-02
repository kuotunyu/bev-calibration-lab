"""Contracts for the cross-repository portfolio artifact envelope.

Three repositories implement this envelope independently, with no shared package
between them, so the only thing holding them together is that each one accepts
exactly the same bytes and rejects exactly the same lies. That makes these tests
the specification rather than a description of an implementation.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

FIXTURE = Path(__file__).resolve().parents[2] / "fixtures" / "portfolio_artifact_envelope_v1.json"


def canonical_document() -> dict[str, Any]:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def test_the_canonical_fixture_round_trips_to_identical_bytes() -> None:
    """A repository that reserialises these bytes differently cannot exchange artifacts.

    The canonical bytes are what gets hashed and exchanged, and they carry no
    trailing newline. The fixture on disk is a text file and does carry one, so
    the two are compared deliberately rather than by accident.
    """

    from bevcalib.artifacts.envelope import PortfolioArtifactEnvelopeV1, canonical_json_bytes

    on_disk = FIXTURE.read_bytes()
    canonical = on_disk.removesuffix(b"\n")
    assert on_disk == canonical + b"\n", "the fixture is a newline-terminated text file"

    envelope = PortfolioArtifactEnvelopeV1.model_validate_json(canonical)

    assert canonical_json_bytes(envelope.model_dump()) == canonical


def test_the_payload_hash_is_computed_over_canonical_payload_bytes() -> None:
    """Both sides must agree on what was hashed, or every hash check is theatre."""

    from bevcalib.artifacts.envelope import canonical_json_bytes

    document = canonical_document()

    recomputed = hashlib.sha256(canonical_json_bytes(document["payload"])).hexdigest()

    assert recomputed == document["payload_sha256"]


def test_canonical_bytes_are_sorted_compact_utf8_without_nan() -> None:
    """Sorted and compact so two producers agree; no NaN because JSON has no NaN."""

    from bevcalib.artifacts.envelope import canonical_json_bytes

    assert canonical_json_bytes({"b": 1, "a": "ü"}) == b'{"a":"\xc3\xbc","b":1}'
    with pytest.raises(ValueError):
        canonical_json_bytes({"x": float("nan")})


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("producer_repository", "some-other-repo"),
        ("producer_release", "1.0.0"),
        ("producer_commit", "not-a-commit"),
        ("protocol_hash", "abc"),
        ("dataset_manifest_hash", "abc"),
        ("payload_sha256", "abc"),
        ("artifact_type", ""),
        ("schema_version", "portfolio-artifact-envelope/v2"),
        ("created_at_utc", "2026-08-31T00:00:00"),
        ("created_at_utc", "not-a-timestamp"),
        # Ends in Z, so it clears the cheap check and must still fail to parse.
        ("created_at_utc", "2026-13-45T99:99:99Z"),
    ],
)
def test_an_envelope_that_misstates_its_provenance_is_rejected(field: str, value: str) -> None:
    """Each of these is a way a foreign or corrupted artifact could look plausible."""

    from pydantic import ValidationError

    from bevcalib.artifacts.envelope import PortfolioArtifactEnvelopeV1

    document = canonical_document() | {field: value}

    with pytest.raises(ValidationError):
        PortfolioArtifactEnvelopeV1.model_validate(document)


def test_an_envelope_carrying_an_unknown_field_is_rejected() -> None:
    """Silently dropping an unknown field would let a producer think it was honoured."""

    from pydantic import ValidationError

    from bevcalib.artifacts.envelope import PortfolioArtifactEnvelopeV1

    with pytest.raises(ValidationError):
        PortfolioArtifactEnvelopeV1.model_validate(canonical_document() | {"extra": 1})


def test_a_validated_envelope_cannot_be_mutated() -> None:
    """Provenance that can be edited after validation is not provenance."""

    from pydantic import ValidationError

    from bevcalib.artifacts.envelope import PortfolioArtifactEnvelopeV1

    envelope = PortfolioArtifactEnvelopeV1.model_validate(canonical_document())

    with pytest.raises(ValidationError):
        envelope.producer_release = "v9.9.9"  # type: ignore[misc]


def test_verifying_an_envelope_checks_its_type_and_recomputes_its_payload_hash(
    tmp_path: Path,
) -> None:
    """A consumer that trusts the declared hash inherits the producer's bugs."""

    from bevcalib.artifacts.envelope import verify_envelope

    path = tmp_path / "artifact.json"
    path.write_bytes(FIXTURE.read_bytes())

    envelope = verify_envelope(path, "portfolio-contract-fixture/v1")

    assert envelope.producer_repository == "driving-risk-metrics"


def test_verifying_fails_closed_on_a_wrong_artifact_type(tmp_path: Path) -> None:
    """Reading a calibration result as a metric table would silently compare nothing."""

    from bevcalib.artifacts.envelope import verify_envelope

    path = tmp_path / "artifact.json"
    path.write_bytes(FIXTURE.read_bytes())

    with pytest.raises(ValueError, match="artifact type"):
        verify_envelope(path, "bev-calibration-result-set/v1")


def test_verifying_fails_closed_when_the_payload_no_longer_hashes_to_its_digest(
    tmp_path: Path,
) -> None:
    """This is the whole point: a payload edited in transit must not pass."""

    from bevcalib.artifacts.envelope import canonical_json_bytes, verify_envelope

    document = canonical_document()
    document["payload"] = {"fixture": "synthetic", "values": [0, 2]}
    path = tmp_path / "artifact.json"
    path.write_bytes(canonical_json_bytes(document))

    with pytest.raises(ValueError, match="payload SHA-256"):
        verify_envelope(path, "portfolio-contract-fixture/v1")
