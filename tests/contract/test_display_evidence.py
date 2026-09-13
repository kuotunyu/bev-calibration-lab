"""Tables and figures share audited source loading and final snapshot checks."""

from pathlib import Path

import pytest
from tests.contract.test_formal_publication import formal_claims_path as formal_claims_path
from tests.contract.test_formal_publication import formal_directory as formal_directory
from tests.unit.artifacts.test_formal_documents import payloads as payloads


def test_audited_display_evidence_retains_exact_registry_and_original_documents(
    formal_claims_path: Path,
    formal_directory: Path,
    tmp_path: Path,
) -> None:
    from bevcalib.report.evidence import load_display_evidence, original_documents

    artifacts, registry_bytes, claims = load_display_evidence(
        formal_claims_path, formal_directory, repository_root=tmp_path
    )
    assert registry_bytes == formal_claims_path.read_bytes()
    assert set(claims) == {("metrics", "/runs/identity/yaw:0/recovery_rate_pct/value")}
    originals = original_documents(artifacts, registry_bytes, formal_claims_path, formal_directory)
    assert set(originals) == {"metrics", "intervals", "recovery", "timing", "exclusions"}
    for name, raw in originals.items():
        assert raw == (formal_directory / f"{name}.json").read_bytes()


def test_display_evidence_refuses_missing_unreferenced_required_document(
    formal_claims_path: Path,
    formal_directory: Path,
    tmp_path: Path,
) -> None:
    from bevcalib.report.evidence import load_display_evidence

    (formal_directory / "timing.json").unlink()
    with pytest.raises(ValueError, match="file inventory differs"):
        load_display_evidence(formal_claims_path, formal_directory, repository_root=tmp_path)
