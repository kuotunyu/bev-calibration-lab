"""Portable, claim-bound reporting without private inputs or nondeterministic assets."""

import json
from copy import deepcopy
from pathlib import Path

import pytest
from tests.unit.analysis.test_claims import VOCABULARY
from tests.unit.metrics.test_summary import run_fixture


def report_paths(summary):
    for label, run in summary["runs"].items():
        base = "/runs/" + label
        if run["seed"] is not None:
            yield base + "/seed"
        for key, condition in run["conditions"].items():
            path = base + "/conditions/" + key
            yield path + "/fault_level"
            for field in ("total", "valid", "invalid", "invalid_rate"):
                yield path + "/validity/" + field
            for reason in condition["validity"]["reasons"]:
                yield path + "/validity/reasons/" + reason.replace("~", "~0").replace("/", "~1")
        for name, group in run["conditions"]["yaw:0"]["ground_contact_by_range"].items():
            for field in ("total", "count", "invalid", "mean"):
                if group[field] is not None:
                    yield base + "/conditions/yaw:0/ground_contact_by_range/" + name + "/" + field
            for reason in group["reasons"]:
                yield (
                    base
                    + "/conditions/yaw:0/ground_contact_by_range/"
                    + name
                    + "/reasons/"
                    + reason.replace("~", "~0").replace("/", "~1")
                )
    yield "/runs/identity/conditions/yaw:0/edge_alignment_score/mean"


@pytest.fixture(scope="module")
def summary_document(tmp_path_factory):
    from bevcalib.metrics.summary import summarize_result_runs

    root, _ = run_fixture(tmp_path_factory.mktemp("portable-summary"))
    return summarize_result_runs(root, synthetic_fixture=True)


@pytest.fixture
def report_workspace(tmp_path: Path, summary_document):
    summary = deepcopy(summary_document)
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    (artifacts / "calibration_summary.json").write_text(json.dumps(summary), encoding="utf-8")
    claims = []
    for index, pointer in enumerate(report_paths(summary)):
        claims.append(
            {
                "claim_id": f"synthetic-{index}",
                "text": "Synthetic measured value.",
                "evidence_type": "synthetic",
                "protocol_hash": summary["protocol_hash"],
                "dataset_manifest_hash": summary["dataset_manifest_hash"],
                "artifact_path": "artifacts/calibration_summary.json",
                "metric_path": pointer,
                "status": "verified",
            }
        )
    path = tmp_path / "claims.yaml"
    path.write_text(json.dumps(VOCABULARY | {"claims": claims}), encoding="utf-8")
    return path, artifacts, summary


def test_two_portable_report_builds_are_identical_and_self_contained(
    report_workspace, tmp_path: Path
) -> None:
    from bevcalib.report.builder import build_report

    claims, artifacts, _ = report_workspace
    first = build_report(claims, artifacts, tmp_path / "one", repository_root=tmp_path)
    second = build_report(claims, artifacts, tmp_path / "two", repository_root=tmp_path)
    assert first.read_bytes() == second.read_bytes()
    html = first.read_text(encoding="utf-8")
    assert "<style>" in html and "<svg" in html
    assert "Synthetic evidence" in html and "Invalid fraction" in html
    assert "flat-plane" in html and "Timing stress" in html and "80+" in html
    assert "<script src=" not in html and "https://" not in html
    assert "source_complete_sha256" not in html
    assert not (artifacts / "evaluation_manifest.json").exists()


@pytest.mark.parametrize(
    "case",
    [
        "parent",
        "missing_zero",
        "wrong_artifact",
        "duplicate",
        "tampered",
        "wrong_protocol",
        "upgraded_evidence",
    ],
)
def test_report_refuses_unbound_numbers_and_wrong_evidence(
    report_workspace, tmp_path: Path, case: str
) -> None:
    from bevcalib.report.builder import build_report

    claims, artifacts, summary = report_workspace
    registry = json.loads(claims.read_text(encoding="utf-8"))
    if case == "parent":
        registry["claims"] = [registry["claims"][0] | {"metric_path": "/runs"}]
    elif case == "missing_zero":
        registry["claims"] = [
            claim
            for claim in registry["claims"]
            if claim["metric_path"] != "/runs/identity/conditions/yaw:0/validity/invalid"
        ]
    elif case == "wrong_artifact":
        (artifacts / "other.json").write_text(json.dumps(summary), encoding="utf-8")
        registry["claims"][0]["artifact_path"] = "artifacts/other.json"
    elif case == "duplicate":
        registry["claims"].append(registry["claims"][0])
    elif case == "tampered":
        summary["runs"]["identity"]["conditions"]["yaw:0"]["validity"]["invalid_rate"] = 0.2
        (artifacts / "calibration_summary.json").write_text(json.dumps(summary), encoding="utf-8")
    elif case == "wrong_protocol":
        registry["claims"][0]["protocol_hash"] = "c" * 64
    else:
        registry["claims"][0]["evidence_type"] = "observed"
    claims.write_text(json.dumps(registry), encoding="utf-8")
    with pytest.raises(ValueError):
        build_report(claims, artifacts, tmp_path / "out", repository_root=tmp_path)
    assert not (tmp_path / "out").exists()


def test_draft_claim_cannot_authorize_a_displayed_zero(report_workspace, tmp_path: Path) -> None:
    from bevcalib.report.builder import build_report

    claims, artifacts, _ = report_workspace
    registry = json.loads(claims.read_text(encoding="utf-8"))
    next(
        claim
        for claim in registry["claims"]
        if claim["metric_path"] == "/runs/identity/conditions/yaw:0/validity/invalid"
    )["status"] = "draft"
    claims.write_text(json.dumps(registry), encoding="utf-8")
    with pytest.raises(ValueError, match="unclaimed"):
        build_report(claims, artifacts, tmp_path / "out", repository_root=tmp_path)


def test_distinct_scalar_claims_must_have_distinct_ids(report_workspace, tmp_path: Path) -> None:
    from bevcalib.report.builder import build_report

    claims, artifacts, _ = report_workspace
    registry = json.loads(claims.read_text(encoding="utf-8"))
    registry["claims"][1]["claim_id"] = registry["claims"][0]["claim_id"]
    claims.write_text(json.dumps(registry), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate claim ID"):
        build_report(claims, artifacts, tmp_path / "out", repository_root=tmp_path)


@pytest.mark.parametrize(
    "case", ["missing_pointer", "wrong_number", "object_number", "duplicate_pointer"]
)
def test_invalid_claim_bindings_are_refused(report_workspace, tmp_path: Path, case: str) -> None:
    from bevcalib.report.builder import build_report

    claims, artifacts, _ = report_workspace
    registry = json.loads(claims.read_text(encoding="utf-8"))
    claim = registry["claims"][0]
    if case == "missing_pointer":
        claim["metric_path"] = "/missing"
    elif case == "wrong_number":
        claim["text"] = "Measured 987654321."
    elif case == "object_number":
        claim.update(metric_path="/runs", text="Measured 987654321.")
    else:
        registry["claims"].append(claim | {"claim_id": "another-id"})
    claims.write_text(json.dumps(registry), encoding="utf-8")
    with pytest.raises(ValueError):
        build_report(claims, artifacts, tmp_path / "out", repository_root=tmp_path)


def test_complete_report_supports_learned_seeds_and_nonnumeric_context(
    summary_document, tmp_path: Path
) -> None:
    from tests.unit.report.test_summary_contract import complete_summary

    from bevcalib.artifacts.summary import write_safe_summary
    from bevcalib.report.builder import build_report

    summary = complete_summary(summary_document)
    artifacts = tmp_path / "artifacts"
    write_safe_summary(summary, artifacts / "calibration_summary.json")
    claims = [
        {
            "claim_id": f"number-{index}",
            "text": "Measured scalar.",
            "evidence_type": "synthetic",
            "protocol_hash": summary["protocol_hash"],
            "dataset_manifest_hash": summary["dataset_manifest_hash"],
            "artifact_path": "artifacts/calibration_summary.json",
            "metric_path": pointer,
            "status": "verified",
        }
        for index, pointer in enumerate(report_paths(summary))
    ]
    claims.append(
        claims[0]
        | {
            "claim_id": "context",
            "metric_path": "/runs",
            "text": "<unsafe>Context & details</unsafe>",
        }
    )
    path = tmp_path / "claims.yaml"
    path.write_text(json.dumps(VOCABULARY | {"claims": claims}), encoding="utf-8")
    output = build_report(path, artifacts, tmp_path / "out", repository_root=tmp_path)
    html = output.read_text(encoding="utf-8")
    assert "Learned" in html and "seed" in html
    assert "&lt;unsafe&gt;Context &amp; details&lt;/unsafe&gt;" in html
    assert "<unsafe>" not in html
