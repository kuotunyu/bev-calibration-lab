"""Every number bound in a published Markdown document equals its evidence.

A bound number in prose names its source in an invisible marker. Formal sources
must be verified claims in `docs/claims.yaml` whose report binding matches the
released document; derived sources must come from a derived document whose own
digest and source digests check out. A number that drifts from its source, a
source that loses its claim and a marker that points nowhere all fail here.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = REPO_ROOT / "docs" / "evidence" / "nuscenes_calibration_v1"
ENVELOPE = REPO_ROOT / "docs" / "analysis" / "operating_envelope_v1" / "operating-envelope.json"
# Documents whose numbers are the point of the document; each must carry bindings.
REQUIRED = ("docs/errata.md", "docs/errata.zh-TW.md")


@pytest.fixture(scope="module")
def evidence() -> tuple[Callable[[str, str], Any], int]:
    from bevcalib.analysis.claims import _resolve_json_pointer
    from bevcalib.artifacts.documents import DOCUMENT_TYPES
    from bevcalib.artifacts.result_documents import digest
    from bevcalib.report.evidence import load_display_evidence
    from bevcalib.report.scalar_binding import bind_scalar

    artifacts, _, scalar_claims = load_display_evidence(
        REPO_ROOT / "docs" / "claims.yaml", EVIDENCE, repository_root=REPO_ROOT
    )
    envelope = json.loads(ENVELOPE.read_bytes())
    body = {key: value for key, value in envelope.items() if key != "document_sha256"}
    assert envelope["evidence_type"] == "derived"
    assert envelope["document_sha256"] == digest(body)
    for name, source in envelope["source"]["documents"].items():
        assert source["document_sha256"] == getattr(artifacts, name).document_sha256
    dumps = {name: getattr(artifacts, name).model_dump(mode="json") for name in DOCUMENT_TYPES}

    def resolve(document: str, pointer: str) -> Any:
        if document == "envelope":
            return _resolve_json_pointer(envelope, pointer)
        if document not in dumps:
            raise LookupError(f"unknown evidence document {document!r}")
        value = _resolve_json_pointer(dumps[document], pointer)
        sha = getattr(artifacts, document).document_sha256
        bind_scalar(document, pointer, value, sha, scalar_claims)
        return value

    return resolve, len(scalar_claims)


def _documents() -> list[tuple[str, str]]:
    from bevcalib.dev import _iter_markdown_files

    return [
        (path.relative_to(REPO_ROOT).as_posix(), path.read_text(encoding="utf-8"))
        for path in _iter_markdown_files(REPO_ROOT)
    ]


def test_every_bound_number_in_the_published_documents_matches_its_evidence(evidence) -> None:  # type: ignore[no-untyped-def]
    from bevcalib.report.document_bindings import audit_markdown_documents

    resolve, _ = evidence
    documents = _documents()
    bindings, violations = audit_markdown_documents(documents, resolve)

    assert violations == ()
    names = dict(documents)
    for required in REQUIRED:
        found, _ = audit_markdown_documents([(required, names[required])], resolve)
        assert found, f"{required} carries no bound numbers"
    assert len(bindings) >= len(REQUIRED)


def test_a_drifted_number_or_an_unclaimed_source_is_caught(evidence) -> None:  # type: ignore[no-untyped-def]
    from bevcalib.report.document_bindings import audit_markdown

    resolve, _ = evidence
    pointer = "/runs/identity/roll:1/pixel_frame_p50_px"
    _, violations = audit_markdown(
        f"Shift 22.45 px. <!-- bind: 22.45 = metrics#{pointer}/value -->\n"
        f"Frames 1207. <!-- bind: 1207 = metrics#{pointer}/support/total_frames -->\n"
        f"Bias 0.00. <!-- bind: 0.00 = metrics#/runs/identity/roll:1/rotation_bias_roll_deg/value -->\n"
        "Other 1. <!-- bind: 1 = figures#/value -->\n",
        resolve,
    )

    assert [violation.split(":")[0] for violation in violations] == ["line 1", "line 3", "line 4"]
    assert "does not match" in violations[0]
    assert "unclaimed displayed formal scalar" in violations[1]
    assert "unknown evidence document" in violations[2]
