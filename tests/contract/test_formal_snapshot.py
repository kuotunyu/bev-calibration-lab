"""Schema-normalized snapshots accept defaults but preserve original source bytes."""

import json
from pathlib import Path

import pytest
import yaml
from tests.contract.test_formal_publication import formal_claims_path as formal_claims_path
from tests.contract.test_formal_publication import formal_directory as formal_directory
from tests.unit.artifacts.test_formal_documents import payloads as payloads

from bevcalib.analysis.claims import audit_claims


def test_small_publication_preserves_missing_support_and_document_boundaries(
    formal_directory: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A small real-source mapping isolates null rendering, not full publication acceptance."""
    import html

    import bevcalib.analysis.formal_claims as publication
    import bevcalib.report.formal as report
    from bevcalib.analysis.claims import load_registry

    artifacts = publication.load_publication(formal_directory, tmp_path)
    rows = tuple(publication.publication_rows(artifacts))
    available = next(row for row in rows if row.document == "metrics" and row.reason is None)
    unavailable = next(row for row in rows if row.document == "metrics" and row.reason is not None)
    recovery = next(row for row in rows if row.document == "recovery")
    selected = (available, unavailable, recovery)
    monkeypatch.setattr(publication, "publication_rows", lambda _: iter(selected))
    monkeypatch.setattr(report, "publication_rows", lambda _: iter(selected))
    claims = publication.generate_formal_claims(
        formal_directory, tmp_path / "small-claims.yaml", repository_root=tmp_path
    )
    registry = load_registry(claims)
    assert audit_claims(claims, tmp_path) == ()
    bound = {(Path(claim.artifact_path).stem, claim.metric_path) for claim in registry.claims}
    for row in selected:
        for _, pointer, value in row.cells:
            assert ((row.document, pointer) in bound) == (value is not None)
    page = report.build_formal_report(
        claims, formal_directory, tmp_path / "site", repository_root=tmp_path
    )
    text = page.read_text(encoding="utf-8")
    assert "<dd>unavailable</dd>" in text
    assert unavailable.reason is not None
    assert html.escape(unavailable.reason) in text
    assert text.count("<details>") == text.count("</details>") == 2
    assert text.count("<tbody>") == text.count("</tbody>") == 2
    assert (page.parent / "claims.yaml").read_bytes() == claims.read_bytes()
    for original in formal_directory.glob("*.json"):
        assert (page.parent / "evidence" / original.name).read_bytes() == original.read_bytes()
    assert not list(tmp_path.glob(".formal-claims-*"))


def test_failed_formal_set_is_checked_once_per_audit_and_retried_next_time(
    formal_claims_path: Path,
    formal_directory: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import bevcalib.analysis.formal_claims as publication

    registry = yaml.safe_load(formal_claims_path.read_text(encoding="utf-8"))
    original_claim = registry["claims"][0]
    registry["claims"] = [
        {**original_claim, "claim_id": f"formal.repeated.{index}"} for index in range(3)
    ]
    formal_claims_path.write_text(yaml.safe_dump(registry), encoding="utf-8")
    source = formal_directory / "exclusions.json"
    original_bytes = source.read_bytes()
    corrupted = json.loads(original_bytes)
    corrupted["document_sha256"] = "0" * 64
    source.write_text(json.dumps(corrupted), encoding="utf-8")
    original_load = publication.load_publication
    calls = []

    def counted(directory, root):  # type: ignore[no-untyped-def]
        calls.append(directory)
        return original_load(directory, root)

    monkeypatch.setattr(publication, "load_publication", counted)
    violations = audit_claims(formal_claims_path, tmp_path)
    assert len(violations) == 3
    assert all("formal evidence invalid" in item for item in violations)
    assert calls == [formal_directory]
    source.write_bytes(original_bytes)
    assert audit_claims(formal_claims_path, tmp_path) == ()
    assert calls == [formal_directory, formal_directory]


def test_atomic_registry_creation_retains_a_competing_writers_file(
    formal_directory: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import bevcalib.analysis.formal_claims as publication

    row = publication.PublicationRow(
        "metrics", "source scenes", "count", (("scenes", "/identity/scene_count", 1),)
    )
    monkeypatch.setattr(publication, "publication_rows", lambda _: iter((row,)))
    target = tmp_path / "claims.yaml"
    original = publication.os.link

    def competitor(source, destination):  # type: ignore[no-untyped-def]
        target.write_bytes(b"another writer owns this file")
        original(source, destination)

    monkeypatch.setattr(publication.os, "link", competitor)
    with pytest.raises(FileExistsError):
        publication.generate_formal_claims(formal_directory, target, repository_root=tmp_path)
    assert target.read_bytes() == b"another writer owns this file"
    assert not list(tmp_path.glob(".formal-claims-*"))


def test_existing_report_directory_is_preserved(
    formal_claims_path: Path,
    formal_directory: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import bevcalib.report.formal as report
    from bevcalib.analysis.formal_claims import PublicationRow
    from bevcalib.artifacts.documents import load_formal_artifact_set

    value = (
        load_formal_artifact_set(formal_directory)
        .metrics.runs["identity"]["yaw:0"]["recovery_rate_pct"]
        .value
    )
    row = PublicationRow(
        "metrics",
        "identity zero-fault recovery",
        "percent",
        (("value", "/runs/identity/yaw:0/recovery_rate_pct/value", value),),
    )
    monkeypatch.setattr(report, "publication_rows", lambda _: iter((row,)))
    output = tmp_path / "existing"
    output.mkdir()
    (output / "index.html").write_bytes(b"retained report")
    with pytest.raises(FileExistsError):
        report.build_formal_report(
            formal_claims_path, formal_directory, output, repository_root=tmp_path
        )
    assert {path.name: path.read_bytes() for path in output.iterdir()} == {
        "index.html": b"retained report"
    }


@pytest.mark.parametrize("operation", ["audit", "report"])
def test_optional_support_defaults_do_not_look_like_concurrent_writes(
    formal_claims_path: Path,
    formal_directory: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
) -> None:
    import bevcalib.report.formal as report
    from bevcalib.analysis.formal_claims import PublicationRow
    from bevcalib.artifacts.documents import load_formal_artifact_set

    def omit_optional(value):  # type: ignore[no-untyped-def]
        if isinstance(value, dict):
            if {"total_frames", "total_scenes", "excluded_frames"} <= value.keys():
                for field in ("total_objects", "objects", "excluded_objects"):
                    if value[field] is None:
                        del value[field]
            for child in value.values():
                omit_optional(child)
        elif isinstance(value, list):
            for child in value:
                omit_optional(child)

    for source in formal_directory.glob("*.json"):
        body = json.loads(source.read_bytes())
        omit_optional(body)
        source.write_text(json.dumps(body), encoding="utf-8")
    artifacts = load_formal_artifact_set(formal_directory)
    if operation == "audit":
        assert audit_claims(formal_claims_path, tmp_path) == ()
    else:
        value = artifacts.metrics.runs["identity"]["yaw:0"]["recovery_rate_pct"].value
        row = PublicationRow(
            "metrics",
            "identity zero-fault recovery",
            "percent",
            (("value", "/runs/identity/yaw:0/recovery_rate_pct/value", value),),
        )
        monkeypatch.setattr(report, "publication_rows", lambda _: iter((row,)))
        path = report.build_formal_report(
            formal_claims_path, formal_directory, tmp_path / "site", repository_root=tmp_path
        )
        assert path.is_file()
        assert (path.parent / "evidence/metrics.json").read_bytes() == (
            formal_directory / "metrics.json"
        ).read_bytes()
