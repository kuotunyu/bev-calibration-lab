"""Contracts for the public-claim registry and its evidence audit.

Every number that reaches a README has to be traceable to a committed artifact and
labelled with how it was obtained. The audit exists so that "the README says 0.42"
and "the artifact says 0.42" cannot drift apart without a test going red.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
COMMITTED_CLAIMS = REPO_ROOT / "docs" / "claims.yaml"


@pytest.mark.parametrize("fallback", [False, True])
def test_registry_yaml_loader_keeps_safe_semantics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fallback: bool
) -> None:
    from bevcalib.analysis.claims import load_registry

    if fallback:
        monkeypatch.delattr(yaml, "CSafeLoader", raising=False)
    path = write_registry(tmp_path, [])
    assert load_registry(path).claims == ()
    path.write_text("!!python/name:builtins.eval", encoding="utf-8")
    with pytest.raises(yaml.YAMLError):
        load_registry(path)


VOCABULARY: dict[str, Any] = {
    "allowed_evidence_types": ["observed", "derived", "synthetic", "illustrative"],
    "claim_required_fields": [
        "claim_id",
        "text",
        "evidence_type",
        "protocol_hash",
        "dataset_manifest_hash",
        "artifact_path",
        "metric_path",
        "status",
    ],
    "allowed_statuses": ["draft", "verified", "rejected", "superseded"],
}


def write_registry(tmp_path: Path, claims: list[dict[str, Any]], **overrides: Any) -> Path:
    path = tmp_path / "claims.yaml"
    path.write_text(yaml.safe_dump(VOCABULARY | {"claims": claims} | overrides), encoding="utf-8")
    return path


def claim(**overrides: Any) -> dict[str, Any]:
    return {
        "claim_id": "recovery-rate",
        "text": "The classical corrector recovers 0.75 of injected rotation faults.",
        "evidence_type": "synthetic",
        "protocol_hash": "a" * 64,
        "dataset_manifest_hash": "b" * 64,
        "artifact_path": "artifacts/summary.json",
        "metric_path": "/recovery/rotation",
        "status": "verified",
    } | overrides


def write_artifact(root: Path, value: Any, **overrides: Any) -> None:
    (root / "artifacts").mkdir(parents=True, exist_ok=True)
    document = {
        "protocol_hash": "a" * 64,
        "dataset_manifest_hash": "b" * 64,
        "recovery": {"rotation": value},
    } | overrides
    (root / "artifacts" / "summary.json").write_text(json.dumps(document), encoding="utf-8")


def test_the_committed_registry_uses_the_approved_vocabulary_and_claims_nothing_yet() -> None:
    """No experiment has run, so any claim in this file today would be fabricated."""

    from bevcalib.analysis.claims import load_registry

    registry = load_registry(COMMITTED_CLAIMS)

    assert registry.allowed_evidence_types == ("observed", "derived", "synthetic", "illustrative")
    assert registry.claims == ()


@pytest.mark.parametrize(
    "override",
    [
        {"allowed_evidence_types": ["observed", "vibes"]},
        {"allowed_statuses": ["draft", "probably-fine"]},
        {"claim_required_fields": ["claim_id", "text"]},
    ],
)
def test_a_registry_that_widens_the_shared_vocabulary_is_rejected(
    tmp_path: Path, override: dict[str, Any]
) -> None:
    """The vocabulary is shared across three repositories; one repo cannot extend it."""

    from pydantic import ValidationError

    from bevcalib.analysis.claims import load_registry

    expected = r"^1 validation error for ClaimsRegistryV1\n  Value error, "
    expected += rf"{next(iter(override))} must match the approved vocabulary"
    with pytest.raises(ValidationError, match=expected):
        load_registry(write_registry(tmp_path, [], **override))


def test_only_verified_claims_may_reach_a_public_page(tmp_path: Path) -> None:
    """Draft and rejected claims exist precisely so they can be written down and withheld."""

    from bevcalib.analysis.claims import verified_claims

    path = write_registry(
        tmp_path,
        [
            claim(claim_id="published", status="verified"),
            claim(claim_id="pending", status="draft"),
            claim(claim_id="withdrawn", status="rejected"),
            claim(claim_id="replaced", status="superseded"),
        ],
    )

    assert [entry.claim_id for entry in verified_claims(path)] == ["published"]


def test_a_claim_whose_numbers_appear_in_its_artifact_passes_the_audit(tmp_path: Path) -> None:
    """The audit has to be satisfiable by an honest claim, or it teaches nothing."""

    from bevcalib.analysis.claims import audit_claims

    write_artifact(tmp_path, 0.75)

    assert audit_claims(write_registry(tmp_path, [claim()]), tmp_path) == ()


def test_a_claim_stating_a_number_absent_from_its_artifact_is_reported(tmp_path: Path) -> None:
    """This is the failure mode: the artifact is regenerated and the prose is not."""

    from bevcalib.analysis.claims import audit_claims

    write_artifact(tmp_path, 0.42)

    violations = audit_claims(write_registry(tmp_path, [claim()]), tmp_path)

    assert len(violations) == 1
    assert "0.75" in violations[0]


@pytest.mark.parametrize(
    ("setup", "expected"),
    [
        ("missing_artifact", "does not exist"),
        ("escaping_path", "escapes repository"),
        ("not_json", "not valid UTF-8 JSON"),
        ("not_an_object", "must be an object"),
        ("wrong_protocol", "protocol hash mismatch"),
        ("wrong_cohort", "dataset manifest hash mismatch"),
        ("missing_pointer", "JSON pointer does not exist"),
    ],
)
def test_evidence_that_cannot_be_reproduced_is_reported_with_its_reason(
    tmp_path: Path, setup: str, expected: str
) -> None:
    """Each of these is a different way a claim can look supported and not be."""

    from bevcalib.analysis.claims import audit_claims

    entry = claim()
    if setup == "escaping_path":
        entry = claim(artifact_path="../outside.json")
    elif setup == "missing_artifact":
        pass
    elif setup == "not_json":
        (tmp_path / "artifacts").mkdir(parents=True, exist_ok=True)
        (tmp_path / "artifacts" / "summary.json").write_text("{nope", encoding="utf-8")
    elif setup == "not_an_object":
        (tmp_path / "artifacts").mkdir(parents=True, exist_ok=True)
        (tmp_path / "artifacts" / "summary.json").write_text("[1, 2]", encoding="utf-8")
    elif setup == "wrong_protocol":
        write_artifact(tmp_path, 0.75, protocol_hash="c" * 64)
    elif setup == "wrong_cohort":
        write_artifact(tmp_path, 0.75, dataset_manifest_hash="c" * 64)
    else:
        write_artifact(tmp_path, 0.75)
        entry = claim(metric_path="/recovery/translation")

    violations = audit_claims(write_registry(tmp_path, [entry]), tmp_path)

    assert any(expected in violation for violation in violations), violations


def test_a_registry_that_cannot_be_read_is_itself_a_violation(tmp_path: Path) -> None:
    """Returning `no violations` because the file was unreadable would fail open."""

    from bevcalib.analysis.claims import audit_claims

    path = tmp_path / "claims.yaml"
    path.write_text("allowed_evidence_types: [observed]\n", encoding="utf-8")

    violations = audit_claims(path, tmp_path)

    assert len(violations) == 1
    assert "registry is invalid" in violations[0]


def test_a_non_finite_number_in_an_artifact_is_refused_rather_than_parsed(tmp_path: Path) -> None:
    """Python json accepts NaN by default; a metric of NaN must never validate a claim."""

    from bevcalib.analysis.claims import audit_claims

    protocol_hash, cohort_hash = "a" * 64, "b" * 64
    (tmp_path / "artifacts").mkdir(parents=True, exist_ok=True)
    (tmp_path / "artifacts" / "summary.json").write_text(
        f'{{"protocol_hash": "{protocol_hash}", "dataset_manifest_hash": "{cohort_hash}", '
        '"recovery": {"rotation": NaN}}',
        encoding="utf-8",
    )

    violations = audit_claims(write_registry(tmp_path, [claim()]), tmp_path)

    assert any("not valid UTF-8 JSON" in violation for violation in violations), violations


@pytest.mark.parametrize(
    ("claim_id", "valid"),
    [("recovery-rate", True), ("Recovery Rate", False), ("-leading-dash", False), ("", False)],
)
def test_a_claim_identifier_must_be_a_stable_slug(
    tmp_path: Path, claim_id: str, valid: bool
) -> None:
    """Claim ids appear in reports and cross-references, so they cannot be free text."""

    from pydantic import ValidationError

    from bevcalib.analysis.claims import load_registry

    path = write_registry(tmp_path, [claim(claim_id=claim_id)])

    if valid:
        assert load_registry(path).claims[0].claim_id == claim_id
    else:
        with pytest.raises(ValidationError, match=r"\nclaims\.0\.claim_id\n"):
            load_registry(path)


def test_a_metric_path_must_be_a_json_pointer(tmp_path: Path) -> None:
    """A bare key would silently resolve to nothing on a nested document."""

    from pydantic import ValidationError

    from bevcalib.analysis.claims import load_registry

    with pytest.raises(ValidationError):
        load_registry(write_registry(tmp_path, [claim(metric_path="recovery.rotation")]))


def test_a_metric_path_can_index_into_an_array(tmp_path: Path) -> None:
    """Per-seed and per-corrector results are naturally arrays, so pointers must index."""

    from bevcalib.analysis.claims import audit_claims

    (tmp_path / "artifacts").mkdir(parents=True, exist_ok=True)
    (tmp_path / "artifacts" / "summary.json").write_text(
        json.dumps(
            {
                "protocol_hash": "a" * 64,
                "dataset_manifest_hash": "b" * 64,
                "seeds": [{"recovery": 0.10}, {"recovery": 0.75}],
            }
        ),
        encoding="utf-8",
    )

    assert audit_claims(write_registry(tmp_path, [claim(metric_path="/seeds/1")]), tmp_path) == ()
    assert audit_claims(write_registry(tmp_path, [claim(metric_path="/seeds/0")]), tmp_path) != ()


@pytest.mark.parametrize("pointer", ["/seeds/9", "/seeds/last", "/seeds/01"])
def test_an_unusable_array_index_is_reported_rather_than_guessed(
    tmp_path: Path, pointer: str
) -> None:
    """Out of range, non-numeric and zero-padded indices are all ways to miss silently."""

    from bevcalib.analysis.claims import audit_claims

    (tmp_path / "artifacts").mkdir(parents=True, exist_ok=True)
    (tmp_path / "artifacts" / "summary.json").write_text(
        json.dumps(
            {
                "protocol_hash": "a" * 64,
                "dataset_manifest_hash": "b" * 64,
                "seeds": [{"recovery": 0.75}],
            }
        ),
        encoding="utf-8",
    )

    violations = audit_claims(write_registry(tmp_path, [claim(metric_path=pointer)]), tmp_path)

    assert any("JSON pointer does not exist" in violation for violation in violations), violations


def test_numbers_are_collected_from_anywhere_under_the_pointer(tmp_path: Path) -> None:
    """A claim may cite any number in the object it points at, however deeply nested."""

    from bevcalib.analysis.claims import audit_claims

    write_artifact(
        tmp_path,
        {"per_seed": [0.70, 0.75], "converged": True, "notes": None, "nested": {"p90": 0.9}},
    )

    assert audit_claims(write_registry(tmp_path, [claim()]), tmp_path) == ()


def test_a_boolean_is_not_a_number_a_claim_can_cite(tmp_path: Path) -> None:
    """`True` equals 1 in Python; a claim saying "1 scene" must not be satisfied by it."""

    from bevcalib.analysis.claims import audit_claims

    write_artifact(tmp_path, {"converged": True})

    violations = audit_claims(
        write_registry(tmp_path, [claim(text="Exactly 1 corrector converged.")]), tmp_path
    )

    assert any("absent from metric" in violation for violation in violations), violations


def test_a_qualitative_claim_may_cite_a_metric_that_is_not_a_number(tmp_path: Path) -> None:
    """Not every claim is numeric; "the ranking reverses" points at a label, not a value."""

    from bevcalib.analysis.claims import audit_claims

    write_artifact(tmp_path, "identity")

    qualitative = claim(text="The identity corrector ranks last on rotation recovery.")
    assert audit_claims(write_registry(tmp_path, [qualitative]), tmp_path) == ()

    numeric = claim(text="The identity corrector recovers 0.75 of rotation faults.")
    violations = audit_claims(write_registry(tmp_path, [numeric]), tmp_path)
    assert any("absent from metric" in violation for violation in violations), violations
