"""The Pages landing page states the result with bound numbers and stays small."""

from __future__ import annotations

import copy
from collections.abc import Iterator, Mapping
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
EVIDENCE = REPO_ROOT / "docs" / "evidence" / "nuscenes_calibration_v1"
PATHS = {
    name: f"docs/evidence/nuscenes_calibration_v1/{name}.json"
    for name in ("metrics", "intervals", "recovery")
}


class VerifiedClaims(Mapping[tuple[str, str], Any]):
    """Stands in for the audited registry: one verified claim per formal scalar."""

    def __init__(self, artifacts: Any) -> None:
        self.artifacts = artifacts
        self.requested: list[tuple[str, str]] = []

    def _value(self, key: tuple[str, str]) -> Any:
        value = getattr(self.artifacts, key[0]).model_dump(mode="json")
        for token in key[1].split("/")[1:]:
            value = value[token]
        return value

    def __getitem__(self, key: tuple[str, str]) -> Any:
        from bevcalib.analysis.claims import ClaimV1, ReportScalarBinding
        from bevcalib.analysis.formal_claims import claim_id

        document = getattr(self.artifacts, key[0])
        self.requested.append(key)
        return ClaimV1(
            claim_id=claim_id(*key),
            text="Formal scalar",
            evidence_type="observed",
            protocol_hash=document.identity.protocol_hash,
            dataset_manifest_hash=document.identity.dataset_manifest_hash,
            artifact_path=PATHS.get(key[0], key[0]),
            metric_path=key[1],
            status="verified",
            report_binding=ReportScalarBinding(
                expected_summary_sha256=document.document_sha256,
                expected_value=self._value(key),
            ),
        )

    def __contains__(self, key: object) -> bool:
        return isinstance(key, tuple) and key[0] in ("metrics", "intervals")

    def __iter__(self) -> Iterator[tuple[str, str]]:
        return iter(())

    def __len__(self) -> int:
        return 0


@pytest.fixture(scope="module")
def released():  # type: ignore[no-untyped-def]
    from bevcalib.analysis.formal_claims import load_publication

    return load_publication(EVIDENCE, REPO_ROOT)


@pytest.fixture(scope="module")
def envelope(released) -> dict[str, Any]:  # type: ignore[no-untyped-def]
    from bevcalib.analysis.operating_envelope import analyse

    return analyse(released, PATHS)


class Page(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[str] = []
        self.claims: list[str] = []
        self.meta: dict[str, str] = {}

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        if tag in ("a", "img") and (target := attributes.get("href") or attributes.get("src")):
            self.links.append(target)
        if claim := attributes.get("data-claim"):
            self.claims.append(claim)
        if tag == "meta" and (name := attributes.get("property") or attributes.get("name")):
            self.meta[name] = attributes.get("content") or ""


def test_landing_page_leads_with_bound_results_and_links_the_evidence(released, envelope) -> None:  # type: ignore[no-untyped-def]
    from bevcalib.report.landing import build_landing

    claims = VerifiedClaims(released)
    html = build_landing(released, claims, envelope)
    page = Page()
    page.feed(html)

    assert len(html.encode("utf-8")) < 100_000
    assert html == build_landing(released, VerifiedClaims(released), envelope)
    assert "og:title" in page.meta and "og:description" in page.meta
    assert {
        "demo/calibration-explorer.html",
        "evidence/index.html",
        "analysis/operating-envelope.svg",
        "analysis/operating-envelope.json",
        "https://github.com/kuotunyu/bev-calibration-lab",
        "https://github.com/kuotunyu/bev-calibration-lab/releases/tag/v1.0.0",
        "https://github.com/kuotunyu/bev-calibration-lab/blob/main/docs/errata.md",
    } <= set(page.links)
    assert "nuScenes" in html and "Kuo Tun-Yu" in html
    assert "22.44</span>&ndash;" in html and "23.95</span> px" in html
    assert (
        "in 60 of 60 grid conditions (54 single-axis faults plus the zero-fault condition, "
        "listed once per axis)"
    ) in html
    assert "0.62&ndash;0.83°" in html
    assert (
        "its pixel error is lower, with the paired 95% interval above zero, only from ±1° tilt, "
        "±1° pan, ±2° in-plane rotation, ±0.2 m lateral offset or ±0.2 m vertical offset (never "
        "for forward offset within the tested range)"
    ) in html
    assert "with the calibration each method ends with (for identity, the faulty one)" in html
    assert "the claim ID is in the page source (data-claim)" in html
    assert "or BEV error beyond 10 m" in html
    assert "0.60</span>° (+" in html and "10.55</span> px" in html
    assert len(page.claims) == len(claims.requested) == 3 + 2 + 55
    assert page.claims[0].startswith("formal.metrics.")


def test_axis_summary_names_every_break_even_and_the_axes_without_one() -> None:
    from bevcalib.report.landing import _axes

    reached = {axis: {"magnitude": 2.0} for axis in ("roll", "pitch", "yaw")} | {
        axis: {"magnitude": 0.1} for axis in ("x", "y", "z")
    }
    assert _axes(reached) == (
        "from ±2° tilt, ±2° pan, ±2° in-plane rotation, ±0.1 m lateral offset, "
        "±0.1 m vertical offset or ±0.1 m forward offset"
    )
    none = {axis: {"magnitude": None} for axis in reached}
    assert _axes(none).startswith("at no tested fault (never for tilt, pan, ")
    single = none | {"z": {"magnitude": 0.2}}
    assert _axes(single).startswith("from ±0.2 m forward offset (never for tilt, pan, ")
    fine = none | {"roll": {"magnitude": 0.5}, "x": {"magnitude": 0.05}}
    assert _axes(fine).startswith("from ±0.5° tilt or ±0.05 m lateral offset (never for pan, ")


def test_joint_wins_count_conditions_better_on_all_four_estimands(released, envelope) -> None:  # type: ignore[no-untyped-def]
    from bevcalib.report.landing import build_landing

    other = copy.deepcopy(envelope)
    counts = other["interval_counts"]["classical->learned-fixed-three-seed-mean"]
    for metric, missing in (
        ("rotation_geodesic_deg", {"roll:1", "roll:2"}),
        ("translation_norm_cm", {"x:0.2", "y:0.2"}),
    ):
        kept = [key for key in counts[metric]["after_better_conditions"] if key not in missing]
        counts[metric]["after_better_conditions"] = kept
        counts[metric]["after_better"] = len(kept)

    html = build_landing(released, VerifiedClaims(released), other)
    assert "in 56 of 60 grid conditions" in html


@pytest.mark.parametrize("change", ["type", "source"])
def test_an_envelope_from_other_evidence_is_refused(released, envelope, change: str) -> None:  # type: ignore[no-untyped-def]
    from bevcalib.report.landing import build_landing

    other = copy.deepcopy(envelope)
    if change == "type":
        other["evidence_type"] = "observed"
    else:
        other["source"]["documents"]["metrics"]["document_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="operating envelope of this evidence"):
        build_landing(released, VerifiedClaims(released), other)


def test_an_unclaimed_displayed_number_is_refused(released, envelope) -> None:  # type: ignore[no-untyped-def]
    from bevcalib.report.landing import build_landing

    with pytest.raises(ValueError, match="unclaimed displayed formal scalar"):
        build_landing(released, {}, envelope)


def test_overview_bundles_landing_and_derived_files_from_audited_evidence(
    released, monkeypatch: pytest.MonkeyPatch
) -> None:  # type: ignore[no-untyped-def]
    import bevcalib.report.evidence as evidence
    from bevcalib.report.landing import build_overview

    calls = []

    def audited(claims_path: Path, artifacts_dir: Path, *, repository_root: Path):  # type: ignore[no-untyped-def]
        calls.append((claims_path, artifacts_dir, repository_root))
        return released, b"registry", VerifiedClaims(released)

    monkeypatch.setattr(evidence, "load_display_evidence", audited)
    files = build_overview(REPO_ROOT / "docs/claims.yaml", EVIDENCE, repository_root=REPO_ROOT)

    assert calls == [(REPO_ROOT / "docs/claims.yaml", EVIDENCE, REPO_ROOT)]
    assert sorted(files) == [
        "analysis/operating-envelope.json",
        "analysis/operating-envelope.svg",
        "index.html",
    ]
    committed = REPO_ROOT / "docs" / "analysis" / "operating_envelope_v1"
    assert (
        files["analysis/operating-envelope.json"].encode()
        == (committed / "operating-envelope.json").read_bytes()
    )
    assert (
        files["analysis/operating-envelope.svg"].encode()
        == (committed / "operating-envelope.svg").read_bytes()
    )
