"""Portable formal SVG package with the same evidence boundary as formal tables."""

from pathlib import Path

from bevcalib.analysis.claims import ClaimV1
from bevcalib.analysis.formal_claims import publication_rows
from bevcalib.artifacts.documents import DOCUMENT_TYPES, FormalArtifactSet
from bevcalib.report.evidence import load_display_evidence, original_documents
from bevcalib.report.figure_data import figure_panels
from bevcalib.report.svg import render_svg


def render_formal_figures(
    artifacts: FormalArtifactSet,
    scalar_claims: dict[tuple[str, str], ClaimV1],
) -> dict[str, str]:
    """Render a loaded/audited snapshot without creating files or reaggregating."""
    panels = figure_panels(
        publication_rows(artifacts),
        {name: getattr(artifacts, name).document_sha256 for name in DOCUMENT_TYPES},
        scalar_claims,
    )
    return {
        name: render_svg(name, items, evidence_type=artifacts.metrics.identity.evidence_type)
        for name, items in panels.items()
    }


def build_formal_figures(
    claims_path: Path,
    artifacts_dir: Path,
    output_dir: Path,
    *,
    repository_root: Path,
) -> tuple[Path, ...]:
    artifacts, registry_bytes, scalar_claims = load_display_evidence(
        claims_path, artifacts_dir, repository_root=repository_root
    )
    rendered = render_formal_figures(artifacts, scalar_claims)
    originals = original_documents(artifacts, registry_bytes, claims_path, artifacts_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "evidence").mkdir()
    for name, raw in originals.items():
        (output_dir / "evidence" / f"{name}.json").write_bytes(raw)
    (output_dir / "claims.yaml").write_bytes(registry_bytes)
    paths = []
    for name, svg in rendered.items():
        path = output_dir / f"{name}.svg"
        path.write_text(svg, encoding="utf-8", newline="\n")
        paths.append(path)
    return tuple(paths)
