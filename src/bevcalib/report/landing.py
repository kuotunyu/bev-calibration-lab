"""A small Pages landing page: the result first, then the ways into the evidence.

Every observed number on the page is read from the loaded formal documents and
bound to its verified claim through `bind_scalar`, as in the full report. Counts
and break-even levels come from the derived operating-envelope document. No
number is typed into this module.
"""

from __future__ import annotations

import html
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from bevcalib.analysis.claims import ClaimV1
from bevcalib.artifacts.documents import FormalArtifactSet
from bevcalib.report.document_bindings import format_bound
from bevcalib.report.scalar_binding import bind_scalar

REPOSITORY = "https://github.com/kuotunyu/bev-calibration-lab"
RELEASE = f"{REPOSITORY}/releases/tag/v1.0.0"
ERRATA = f"{REPOSITORY}/blob/main/docs/errata.md"
METHODS = ("identity", "classical", "learned-17", "learned-42", "learned-73")
ROWS = (
    ("pitch:0", "none"),
    ("roll:0.5", "tilt 0.5°"),
    ("roll:1", "tilt 1°"),
    ("roll:2", "tilt 2°"),
    ("pitch:0.5", "pan 0.5°"),
    ("pitch:1", "pan 1°"),
    ("pitch:2", "pan 2°"),
    ("yaw:2", "in-plane 2°"),
    ("x:0.2", "lateral 0.2 m"),
    ("y:0.2", "vertical 0.2 m"),
    ("z:0.2", "forward 0.2 m"),
)
MEAN = "identity->learned-fixed-three-seed-mean"
VERSUS_CLASSICAL = "classical->learned-fixed-three-seed-mean"
SUMMARY = (
    "Controlled LiDAR-camera extrinsic faults on nuScenes: what they cost in reprojection "
    "error, and when classical or learned correction beats leaving the calibration alone."
)
STYLE = (
    ":root{color-scheme:light dark;--ink:#172638;--muted:#40586e;--paper:#fafbfc;"
    "--card:#ffffff;--line:#ccd4dc;--link:#075c99}"
    "@media (prefers-color-scheme:dark){:root{--ink:#e6edf3;--muted:#a9b8c6;"
    "--paper:#0f1720;--card:#16212c;--line:#2c3a48;--link:#7cc4ff}}"
    "*{box-sizing:border-box}body{margin:0;background:var(--paper);color:var(--ink);"
    "font:16px/1.6 system-ui,sans-serif}main{max-width:980px;margin:auto;padding:24px 16px}"
    "h1{font-size:clamp(26px,5vw,36px);line-height:1.2;margin:0 0 8px}"
    "h2{font-size:20px;margin:32px 0 8px}a{color:var(--link)}.lead{color:var(--muted)}"
    ".actions{display:flex;flex-wrap:wrap;gap:8px;margin:20px 0}"
    ".actions a{display:inline-block;min-height:44px;padding:10px 14px;"
    "border:1px solid var(--line);border-radius:6px;background:var(--card);text-decoration:none}"
    ".tldr{background:var(--card);border:1px solid var(--line);border-radius:6px;padding:4px 16px}"
    "figure{margin:16px 0}figure img{width:100%;height:auto;background:#ffffff;"
    "border:1px solid var(--line)}figcaption,.note,footer{font-size:14px;color:var(--muted)}"
    ".table{overflow-x:auto}table{border-collapse:collapse;width:100%;"
    "font-variant-numeric:tabular-nums}th,td{padding:6px 8px;border-bottom:1px solid var(--line);"
    "text-align:right}th:first-child{text-align:left}footer{margin-top:32px}"
)


def _either(items: list[str]) -> str:
    return " or ".join(item for item in (", ".join(items[:-1]), items[-1]) if item)


def _axes(break_even: Mapping[str, Any]) -> str:
    names = {
        "roll": ("tilt", "°"),
        "pitch": ("pan", "°"),
        "yaw": ("in-plane rotation", "°"),
        "x": ("lateral offset", " m"),
        "y": ("vertical offset", " m"),
        "z": ("forward offset", " m"),
    }
    reached = [
        f"±{value['magnitude']:g}{unit} {name}"
        for axis, (name, unit) in names.items()
        if (value := break_even[axis])["magnitude"] is not None
    ]
    never = [name for axis, (name, _) in names.items() if break_even[axis]["magnitude"] is None]
    text = "from " + _either(reached) if reached else "at no tested fault"
    if never:
        text += " (never for " + _either(never) + " within the tested range)"
    return text


def build_landing(
    artifacts: FormalArtifactSet,
    scalar_claims: Mapping[tuple[str, str], ClaimV1],
    envelope: Mapping[str, Any],
) -> str:
    """Return the landing page for one audited snapshot and its derived envelope."""

    declared = {
        name: value["document_sha256"] for name, value in envelope["source"]["documents"].items()
    }
    if envelope["evidence_type"] != "derived" or declared != {
        name: getattr(artifacts, name).document_sha256 for name in declared
    }:
        raise ValueError("landing page requires the operating envelope of this evidence")
    escape = html.escape
    dumps: dict[str, Any] = {}

    def observed(document: str, pointer: str, decimals: int) -> str:
        source = getattr(artifacts, document)
        if document not in dumps:
            dumps[document] = source.model_dump(mode="json")
        value = dumps[document]
        for token in pointer.split("/")[1:]:
            value = value[token.replace("~1", "/").replace("~0", "~")]
        claim = bind_scalar(document, pointer, value, source.document_sha256, scalar_claims)
        return f'<span data-claim="{escape(str(claim))}">{format_bound(value, decimals)}</span>'

    counts = envelope["interval_counts"][VERSUS_CLASSICAL]
    # Conditions where the learned mean is better on all four estimands at once.
    wins = len(
        set.intersection(
            *(
                set(counts[metric]["after_better_conditions"])
                for metric in (
                    "rotation_geodesic_deg",
                    "translation_norm_cm",
                    "pixel_frame_p50_px",
                    "recovery_rate_pct",
                )
            )
        )
    )
    grid = envelope["grid"]
    residual = envelope["residual"]
    floor = [residual[seed]["rotation_geodesic_deg"] for seed in ("learned-17", "learned-73")]
    low = format_bound(min(item["min"] for item in floor), 2)
    high = format_bound(max(item["max"] for item in floor), 2)
    pixel = "pixel_frame_p50_px"
    tldr = (
        f"On {observed('metrics', f'/runs/identity/roll:1/{pixel}/support/scenes', 0)} locked "
        "nuScenes validation scenes, a 1° tilt or pan error in the CAM_FRONT&ndash;LIDAR_TOP "
        "extrinsic shifts projected LiDAR points by "
        f"{observed('metrics', f'/runs/identity/roll:1/{pixel}/value', 2)}&ndash;"
        f"{observed('metrics', f'/runs/identity/pitch:1/{pixel}/value', 2)} px "
        "(scene mean of the per-frame median).",
        "A ConvNeXtV2-Tiny corrector (mean of three seeds) beats a single-frame "
        "edge-alignment optimizer on rotation, translation, pixel error and recovery in "
        f"{wins} of {grid['conditions']} grid conditions ({grid['injected_fault_conditions']} "
        "single-axis faults plus the zero-fault condition, listed once per axis), but seeds 17 "
        f"and 73 keep a residual rotation error of {low}&ndash;{high}° whatever the fault. "
        "Compared with leaving the calibration alone, its pixel error is lower, with the paired "
        f"95% interval above zero, only {_axes(envelope['break_even'][MEAN])}, and it moves a "
        "correct calibration by "
        f"{observed('intervals', f'/comparisons/{MEAN}/pitch:0/rotation_geodesic_deg/after', 2)}° "
        f"(+{observed('intervals', f'/comparisons/{MEAN}/pitch:0/{pixel}/after', 2)} px).",
        "Implication, not tested here: online recalibration needs a miscalibration detector "
        "or an abstain gate in front of the corrector.",
    )
    header = "".join(f'<th scope="col">{escape(method)}</th>' for method in METHODS)
    rows = "".join(
        f'<tr><th scope="row">{escape(label)} <code>{escape(key)}</code></th>'
        + "".join(
            f"<td>{observed('metrics', f'/runs/{method}/{key}/{pixel}/value', 2)}</td>"
            for method in METHODS
        )
        + "</tr>"
        for key, label in ROWS
    )
    title = "bev-calibration-lab · LiDAR-camera calibration fault study"
    parts = [
        '<!doctype html><html lang="en"><head><meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        f"<title>{escape(title)}</title>",
        f'<meta name="description" content="{escape(SUMMARY)}">',
        f'<meta property="og:title" content="{escape(title)}">',
        f'<meta property="og:description" content="{escape(SUMMARY)}">',
        '<meta property="og:type" content="website">',
        f"<style>{STYLE}</style></head><body><main>",
        "<h1>bev-calibration-lab</h1>",
        '<p class="lead">How much does a small LiDAR-camera calibration error cost, and how '
        "much of it can a corrector get back? A controlled fault study on nuScenes.</p>",
        '<nav class="actions" aria-label="Project links">'
        '<a href="demo/calibration-explorer.html">Calibration explorer (synthetic)</a>'
        '<a href="evidence/index.html">Full evidence report</a>'
        f'<a href="{REPOSITORY}">GitHub repository</a>'
        f'<a href="{RELEASE}">Release v1.0.0</a>'
        f'<a href="{ERRATA}">Known issues</a></nav>',
        '<section class="tldr" aria-labelledby="tldr"><h2 id="tldr">TL;DR</h2>',
        *(f"<p>{sentence}</p>" for sentence in tldr),
        "</section>",
        '<h2 id="envelope">Operating envelope</h2>',
        '<figure><a href="analysis/operating-envelope.svg"><img '
        'src="analysis/operating-envelope.svg" alt="Pixel P50 against the injected fault on '
        "each camera axis for identity, classical and three learned seeds, with the break-even "
        'against leaving the calibration alone."></a><figcaption>Derived from the released '
        "metrics and paired intervals; data, definitions and source digests are in "
        '<a href="analysis/operating-envelope.json">operating-envelope.json</a>. Open the SVG '
        "for per-point sources.</figcaption></figure>",
        '<h2 id="key-results">Key results: pixel P50 (px)</h2>',
        '<p class="note">Scene mean of the per-frame median shift between LiDAR points '
        "projected with the true calibration and with the calibration each method ends with "
        "(for identity, the faulty one); lower is better. Formal axes are CAM_FRONT "
        "optical-frame axes: roll is tilt, pitch is pan and yaw is in-plane rotation. Each "
        "value is bound to a verified claim in claims.yaml; the claim ID is in the page source "
        "(data-claim). Negative levels and every other estimand are in the full evidence "
        "report.</p>",
        f'<div class="table"><table><thead><tr><th scope="col">Fault</th>{header}</tr></thead>'
        f"<tbody>{rows}</tbody></table></div>",
        f'<p class="note">Read the <a href="{ERRATA}">known issues</a> before citing identity '
        "recovery at ±0.25°, the timing stress test or BEV error beyond 10 m.</p>",
        "<footer><p>Kuo Tun-Yu (kuotunyu). Source code under the MIT licence.</p>"
        "<p>Aggregate results derived from the nuScenes v1.0-trainval dataset (Caesar et al., "
        "CVPR 2020), shared for non-commercial research under the nuScenes terms of use. No "
        "images, point clouds, sample tokens or model weights are published here.</p>"
        "<p>Not detector performance, deployment qualification or real-vehicle safety "
        "evidence.</p></footer>",
        "</main></body></html>\n",
    ]
    return "\n".join(parts)


def build_overview(
    claims_path: Path, artifacts_dir: Path, *, repository_root: Path
) -> dict[str, str]:
    """Return the landing page and the derived envelope files, keyed by site path."""

    from bevcalib.analysis.operating_envelope import SOURCES, analyse, encode
    from bevcalib.report.envelope_figure import render_envelope_svg
    from bevcalib.report.evidence import load_display_evidence

    artifacts, _, scalar_claims = load_display_evidence(
        claims_path, artifacts_dir, repository_root=repository_root
    )
    root = repository_root.resolve()
    paths = {
        name: (artifacts_dir / f"{name}.json").resolve().relative_to(root).as_posix()
        for name in SOURCES
    }
    envelope = analyse(artifacts, paths)
    return {
        "index.html": build_landing(artifacts, scalar_claims, envelope),
        "analysis/operating-envelope.json": encode(envelope).decode("utf-8"),
        "analysis/operating-envelope.svg": render_envelope_svg(envelope),
    }
