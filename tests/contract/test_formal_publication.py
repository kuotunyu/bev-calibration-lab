"""Formal publication consumes the complete validated five-document contract."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
import yaml
from tests.unit.artifacts.test_formal_documents import payloads as payloads

from bevcalib.analysis.claims import (
    ALLOWED_EVIDENCE_TYPES,
    ALLOWED_STATUSES,
    CLAIM_REQUIRED_FIELDS,
    audit_claims,
    load_registry,
)


@pytest.fixture
def formal_directory(payloads, tmp_path: Path) -> Path:  # type: ignore[no-untyped-def]
    directory = tmp_path / "evidence"
    directory.mkdir()
    for name, body in payloads.items():
        (directory / f"{name}.json").write_text(json.dumps(body), encoding="utf-8")
    return directory


@pytest.fixture
def formal_claims_path(formal_directory: Path, tmp_path: Path) -> Path:
    body = json.loads((formal_directory / "metrics.json").read_bytes())
    value = body["runs"]["identity"]["yaw:0"]["recovery_rate_pct"]["value"]
    registry = {
        "allowed_evidence_types": list(ALLOWED_EVIDENCE_TYPES),
        "claim_required_fields": list(CLAIM_REQUIRED_FIELDS),
        "allowed_statuses": list(ALLOWED_STATUSES),
        "claims": [
            {
                "claim_id": "formal.identity.recovery",
                "text": f"Recovery rate: {value}",
                "evidence_type": "synthetic",
                "protocol_hash": body["identity"]["protocol_hash"],
                "dataset_manifest_hash": body["identity"]["dataset_manifest_hash"],
                "artifact_path": "evidence/metrics.json",
                "metric_path": "/runs/identity/yaw:0/recovery_rate_pct/value",
                "status": "verified",
                "report_binding": {
                    "expected_summary_sha256": body["document_sha256"],
                    "expected_value": value,
                },
            }
        ],
    }
    claims = tmp_path / "claims.yaml"
    claims.write_text(yaml.safe_dump(registry), encoding="utf-8")
    return claims


def test_valid_formal_nested_identity_is_accepted_by_audit(
    formal_claims_path: Path, tmp_path: Path
) -> None:
    assert audit_claims(formal_claims_path, tmp_path) == ()


@pytest.mark.parametrize("case", ["missing_document", "stale_digest", "wrong_value", "label"])
def test_formal_audit_validates_whole_set_and_exact_binding(
    formal_claims_path: Path, formal_directory: Path, tmp_path: Path, case: str
) -> None:
    registry = yaml.safe_load(formal_claims_path.read_text(encoding="utf-8"))
    claim = registry["claims"][0]
    if case == "missing_document":
        (formal_directory / "timing.json").unlink()
    elif case == "stale_digest":
        claim["report_binding"]["expected_summary_sha256"] = "f" * 64
    elif case == "wrong_value":
        claim["report_binding"]["expected_value"] += 1
    else:
        claim["evidence_type"] = "observed"
    formal_claims_path.write_text(yaml.safe_dump(registry), encoding="utf-8")
    violations = audit_claims(formal_claims_path, tmp_path)
    assert any("formal" in item for item in violations)
    assert not any("protocol hash mismatch" in item for item in violations)


def test_formal_registry_and_report_use_original_five_documents(
    formal_directory: Path, tmp_path: Path
) -> None:
    from bevcalib.analysis.formal_claims import generate_formal_claims
    from bevcalib.report.formal import build_formal_report

    original = {path.name: path.read_bytes() for path in formal_directory.iterdir()}
    claims = generate_formal_claims(
        formal_directory, tmp_path / "claims.yaml", repository_root=tmp_path
    )
    assert load_registry(claims).claims
    assert audit_claims(claims, tmp_path) == ()
    report = build_formal_report(
        claims, formal_directory, tmp_path / "site", repository_root=tmp_path
    )
    assert report.is_file()
    assert "Formal calibration evidence" in report.read_text(encoding="utf-8")
    assert {path.name: path.read_bytes() for path in formal_directory.iterdir()} == original
    assert not (formal_directory / "calibration_summary.json").exists()


@pytest.mark.parametrize("case", ["empty", "duplicate_id", "unclaimed_scalar"])
def test_formal_report_refuses_incomplete_registry_before_writing(
    formal_claims_path: Path, formal_directory: Path, tmp_path: Path, case: str
) -> None:
    from bevcalib.report.formal import build_formal_report

    registry = yaml.safe_load(formal_claims_path.read_text(encoding="utf-8"))
    if case == "empty":
        registry["claims"] = []
    elif case == "duplicate_id":
        registry["claims"].append(registry["claims"][0].copy())
    formal_claims_path.write_text(yaml.safe_dump(registry), encoding="utf-8")
    expected = {
        "empty": "nonempty",
        "duplicate_id": "duplicate claim ID",
        "unclaimed_scalar": "unclaimed displayed",
    }
    with pytest.raises(ValueError, match=expected[case]):
        build_formal_report(
            formal_claims_path, formal_directory, tmp_path / "absent", repository_root=tmp_path
        )
    assert not (tmp_path / "absent").exists()


def test_formal_audit_rejects_duplicate_claim_ids(formal_claims_path: Path, tmp_path: Path) -> None:
    registry = yaml.safe_load(formal_claims_path.read_text(encoding="utf-8"))
    registry["claims"].append(registry["claims"][0].copy())
    formal_claims_path.write_text(yaml.safe_dump(registry), encoding="utf-8")
    assert any("duplicate claim ID" in item for item in audit_claims(formal_claims_path, tmp_path))


def test_formal_audit_does_not_fall_back_to_legacy_for_unknown_formal_schema(
    formal_claims_path: Path, formal_directory: Path, tmp_path: Path
) -> None:
    source = formal_directory / "metrics.json"
    body = json.loads(source.read_bytes())
    body["schema_version"] = "bev-calibration-metrics/unsupported"
    # Supplying legacy-shaped hashes must not authorize an unvalidated formal file.
    body["protocol_hash"] = body["identity"]["protocol_hash"]
    body["dataset_manifest_hash"] = body["identity"]["dataset_manifest_hash"]
    source.write_text(json.dumps(body), encoding="utf-8")
    assert any("formal schema" in item for item in audit_claims(formal_claims_path, tmp_path))


def test_formal_audit_checks_resolved_boundaries_of_unreferenced_required_files(
    formal_claims_path: Path,
    formal_directory: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = Path.resolve
    target = formal_directory / "timing.json"

    def resolve(path: Path, *args, **kwargs) -> Path:  # type: ignore[no-untyped-def]
        if path == target:
            return tmp_path.parent / "outside-formal-source.json"
        return original(path, *args, **kwargs)

    # Model the resolved result of a symlink without requiring OS symlink privileges.
    monkeypatch.setattr(Path, "resolve", resolve)
    assert any("escapes repository" in item for item in audit_claims(formal_claims_path, tmp_path))


def test_formal_audit_rejects_schema_filename_alias(
    formal_claims_path: Path, formal_directory: Path, tmp_path: Path
) -> None:
    shutil.copyfile(formal_directory / "metrics.json", formal_directory / "alias.json")
    registry = yaml.safe_load(formal_claims_path.read_text(encoding="utf-8"))
    registry["claims"][0]["artifact_path"] = "evidence/alias.json"
    formal_claims_path.write_text(yaml.safe_dump(registry), encoding="utf-8")
    assert any("filename differs" in item for item in audit_claims(formal_claims_path, tmp_path))


def test_formal_audit_rejects_source_changed_between_snapshot_and_set_validation(
    formal_claims_path: Path,
    formal_directory: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import bevcalib.analysis.formal_claims as publication
    from bevcalib.artifacts.result_documents import digest

    original = publication.load_publication

    def replace_source(directory: Path, repository_root: Path):  # type: ignore[no-untyped-def]
        for source in directory.glob("*.json"):
            body = json.loads(source.read_bytes())
            body["identity"]["measurement_identity_sha256"] = "f" * 64
            body["document_sha256"] = digest(
                {key: value for key, value in body.items() if key != "document_sha256"}
            )
            source.write_text(json.dumps(body), encoding="utf-8")
        # This is a valid but different complete set, not an invalid model mock.
        return original(directory, repository_root)

    monkeypatch.setattr(publication, "load_publication", replace_source)
    assert any(
        "changed during audit" in item for item in audit_claims(formal_claims_path, tmp_path)
    )


def test_generator_does_not_overwrite_registry(formal_directory: Path, tmp_path: Path) -> None:
    from bevcalib.analysis.formal_claims import generate_formal_claims

    target = tmp_path / "claims.yaml"
    target.write_bytes(b"existing registry must be retained")
    with pytest.raises(FileExistsError, match="already exists"):
        generate_formal_claims(formal_directory, target, repository_root=tmp_path)
    assert target.read_bytes() == b"existing registry must be retained"


@pytest.mark.parametrize("case", ["input", "registry"])
def test_generator_refuses_paths_outside_repository(
    formal_directory: Path, tmp_path: Path, case: str
) -> None:
    from bevcalib.analysis.formal_claims import generate_formal_claims

    root = formal_directory if case == "input" else tmp_path
    source = tmp_path if case == "input" else formal_directory
    target = tmp_path.parent / f"{tmp_path.name}-outside.yaml"
    with pytest.raises(ValueError, match="escapes repository"):
        generate_formal_claims(source, target, repository_root=root)
    assert not target.exists()


def test_publication_mapping_keeps_every_method_condition_and_all_range_bins(
    formal_directory: Path,
) -> None:
    from bevcalib.analysis.formal_claims import load_publication, publication_rows

    artifacts = load_publication(formal_directory, formal_directory.parent)
    rows = list(publication_rows(artifacts))
    assert {row.document for row in rows} == {
        "metrics",
        "intervals",
        "recovery",
        "timing",
        "exclusions",
    }
    pointers = {(row.document, pointer) for row in rows for _, pointer, _ in row.cells}
    for run, conditions in artifacts.metrics.runs.items():
        for condition in conditions:
            for metric in (
                "rotation_geodesic_deg",
                "translation_norm_cm",
                "pixel_frame_p50_px",
                "pixel_frame_p90_px",
                "edge_score_px",
            ):
                assert ("metrics", f"/runs/{run}/{condition}/{metric}/value") in pointers
            assert ("exclusions", f"/runs/{run}/{condition}/invalid") in pointers
            if not condition.startswith("time:"):
                assert ("recovery", f"/runs/{run}/{condition}/value") in pointers
    range_metrics = {
        metric
        for metric in artifacts.metrics.identity.units
        if metric.startswith("bev_frame_mean_m/")
    }
    assert len(range_metrics) == 5
    for metric in range_metrics:
        assert (
            "metrics",
            "/runs/identity/yaw:0/" + metric.replace("/", "~1") + "/value",
        ) in pointers
    assert any(value is None for row in rows for _, _, value in row.cells)


@pytest.mark.parametrize(
    "case, message",
    [
        ("audit", "formal claim audit failed"),
        ("draft", "unclaimed displayed"),
        ("wrong_artifact", "wrong artifact"),
        ("duplicate_scalar", "multiple verified claims"),
        ("mapping", "verified source snapshot"),
        ("source_change", "formal source changed"),
        ("registry_change", "formal registry changed"),
    ],
)
def test_report_refuses_bad_bindings_and_read_time_changes_before_writing(
    formal_claims_path: Path,
    formal_directory: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    case: str,
    message: str,
) -> None:
    import bevcalib.report.formal as report
    from bevcalib.analysis.formal_claims import PublicationRow

    registry = yaml.safe_load(formal_claims_path.read_text(encoding="utf-8"))
    claim = registry["claims"][0]
    value = claim["report_binding"]["expected_value"]
    if case == "audit":
        claim["report_binding"]["expected_value"] += 1
    elif case == "draft":
        claim["status"] = "draft"
    elif case == "wrong_artifact":
        shutil.copytree(formal_directory, tmp_path / "alternate")
        claim["artifact_path"] = "alternate/metrics.json"
    elif case == "duplicate_scalar":
        registry["claims"].append(claim | {"claim_id": "formal.second-claim"})
    formal_claims_path.write_text(yaml.safe_dump(registry), encoding="utf-8")

    # One legitimate display row isolates publication defenses. The separate
    # end-to-end contract exercises the complete unmodified field mapping.
    def rows(artifacts):  # type: ignore[no-untyped-def]
        if case == "source_change":
            from bevcalib.artifacts.result_documents import digest

            source = formal_directory / "metrics.json"
            document = json.loads(source.read_bytes())
            document["identity"]["measurement_identity_sha256"] = "f" * 64
            document["document_sha256"] = digest(
                {key: value for key, value in document.items() if key != "document_sha256"}
            )
            source.write_text(json.dumps(document), encoding="utf-8")
        elif case == "registry_change":
            with formal_claims_path.open("a", encoding="utf-8") as handle:
                handle.write("\n# concurrent writer\n")
        yield PublicationRow(
            "metrics",
            "identity zero-fault recovery",
            "percent",
            (("value", claim["metric_path"], value + 1 if case == "mapping" else value),),
        )

    monkeypatch.setattr(report, "publication_rows", rows)
    with pytest.raises(ValueError, match=message):
        report.build_formal_report(
            formal_claims_path, formal_directory, tmp_path / "absent", repository_root=tmp_path
        )
    assert not (tmp_path / "absent").exists()
