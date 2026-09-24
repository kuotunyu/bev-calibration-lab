"""Standalone formal tables bound to the original five documents and exact claims."""

from __future__ import annotations

import html
from pathlib import Path, PurePosixPath

from bevcalib.analysis.formal_claims import publication_rows
from bevcalib.report.evidence import load_display_evidence, original_documents
from bevcalib.report.formal_figures import render_formal_figures
from bevcalib.report.scalar_binding import bind_scalar


def build_formal_report(
    claims_path: Path,
    artifacts_dir: Path,
    output_dir: Path,
    *,
    repository_root: Path,
    include_figures: bool = False,
    page: str = "index.html",
) -> Path:
    """Write the report page at `page`, a relative path inside `output_dir`.

    Assets keep their places at the top of `output_dir`, so a page placed in a
    subdirectory links back to them with `../` and every asset URL stays the same.
    """
    artifacts, registry_bytes, scalar_claims = load_display_evidence(
        claims_path, artifacts_dir, repository_root=repository_root
    )
    figures = render_formal_figures(artifacts, scalar_claims) if include_figures else {}

    identity = artifacts.metrics.identity
    escape = html.escape
    up = "../" * (len(PurePosixPath(page).parts) - 1)
    parts = [
        '<!doctype html><html lang="en"><meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        "<title>Formal calibration evidence</title>",
        "<style>body{font:16px system-ui,sans-serif;max-width:1100px;margin:2rem auto;padding:0 1rem;color:#172638;background:#fafbfc}table{width:100%;border-collapse:collapse}td,th{padding:.7rem;text-align:left;border-bottom:1px solid #ccd4dc;vertical-align:top}dl{margin:0;display:grid;grid-template-columns:auto auto;gap:.25rem 1rem}dd{margin:0;overflow-wrap:anywhere}code{overflow-wrap:anywhere}summary{font-size:1.3rem;cursor:pointer;padding:1rem 0}a{color:#075c99}</style>",
        "<h1>Formal calibration evidence</h1>",
        f"<p>Evidence type: <strong>{escape(identity.evidence_type)}</strong></p>",
        "<p>These are scene-level calibration measurements, not evidence of real-vehicle safety. "
        "Zero-fault conditions and identity remain in every applicable table. "
        "The fixed-three-seed mean is a statistic on common eligible support, not a prediction ensemble.</p>",
        "<p>Missing values remain unavailable. Global row validity and operator support have different denominators. "
        "Signed and per-axis descriptors remain available in the complete linked metrics document.</p>",
        f"<p>Protocol: <code>{identity.protocol_hash}</code><br>Manifest: <code>{identity.dataset_manifest_hash}</code></p>",
        "<h2>Predeclared estimands and interpretation</h2><dl>",
    ]
    for key, text in identity.estimands.items():
        parts.append(f"<dt>{escape(key)}</dt><dd>{escape(text)}</dd>")
    parts.extend(["</dl>", f'<p><a href="{up}claims.yaml">Exact scalar registry</a></p>'])
    for name in figures:
        title = name.replace("-", " ").capitalize()
        parts.append(
            f'<details><summary>{escape(title)}</summary><p><a href="{up}figures/{name}.svg">'
            "Open standalone SVG with exact-value and source tooltips</a></p>"
            f'<img src="{up}figures/{name}.svg" alt="{escape(title)}" style="width:100%;height:auto"></details>'
        )
    current = None
    for row in publication_rows(artifacts):
        if row.document != current:
            if current is not None:
                parts.append("</tbody></table></details>")
            current = row.document
            parts.append(
                f'<details><summary>{escape(current)}</summary><p><a href="{up}evidence/{current}.json">'
                "Original source document</a></p><table><thead><tr><th>Condition and estimand</th>"
                "<th>Metric unit</th><th>Values and support</th></tr></thead><tbody>"
            )
        parts.append(f"<tr><th>{escape(row.label)}</th><td>{escape(row.unit)}</td><td><dl>")
        for label, pointer, value in row.cells:
            parts.append(f"<dt>{escape(label)}</dt>")
            identifier = bind_scalar(
                row.document,
                pointer,
                value,
                getattr(artifacts, row.document).document_sha256,
                scalar_claims,
            )
            if value is None:
                parts.append("<dd>unavailable</dd>")
            else:
                parts.append(f'<dd data-claim="{escape(str(identifier))}">{value}</dd>')
        parts.append("</dl>")
        if row.reason is not None:
            parts.append(f"<p>Unavailable reason: {escape(row.reason)}</p>")
        parts.append("</td></tr>")
    parts.extend(["</tbody></table></details>", "</html>\n"])
    originals = original_documents(artifacts, registry_bytes, claims_path, artifacts_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "evidence").mkdir()
    for name, raw in originals.items():
        (output_dir / "evidence" / f"{name}.json").write_bytes(raw)
    (output_dir / "claims.yaml").write_bytes(registry_bytes)
    if figures:
        (output_dir / "figures").mkdir()
        for name, svg in figures.items():
            (output_dir / "figures" / f"{name}.svg").write_text(svg, encoding="utf-8", newline="\n")
    path = output_dir / page
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts), encoding="utf-8", newline="\n")
    return path
