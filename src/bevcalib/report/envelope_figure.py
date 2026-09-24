"""One compact, deterministic SVG of the derived operating envelope."""

from __future__ import annotations

import html
import json
import math
from collections.abc import Mapping
from typing import Any

from bevcalib.report.svg import COLORS

_VERDICT_STYLE = {
    "after_better": 'fill="#15803d"',
    "before_better": 'fill="#b91c1c"',
    "inconclusive": 'fill="#ffffff" stroke="#64748b"',
}
_METHODS = ("identity", "classical", "learned-17", "learned-42", "learned-73")
_AXES = ("roll", "pitch", "yaw", "x", "y", "z")
_UNIT_LABEL = {"degree": ("°", "deg"), "metre": (" m", "m")}


def _level(key: str) -> tuple[str, float]:
    axis, _, level = key.partition(":")
    return axis, float(level)


def render_envelope_svg(document: Mapping[str, Any]) -> str:
    """Render an already derived document; never read or change formal evidence."""

    if document.get("evidence_type") != "derived":
        raise ValueError("operating envelope figure requires a derived document")
    series = document["series"]["pixel_frame_p50_px"]
    verdicts = document["series"]["verdict"]
    pointer = document["series"]["pointers"]["pixel_frame_p50_px"]
    break_even = document["break_even"]["identity->learned-fixed-three-seed-mean"]
    values = [
        value for method in _METHODS for value in series[method].values() if value is not None
    ]
    top_value = max(10.0, math.ceil(max(values, default=0.0) / 10) * 10)
    escape = html.escape
    width, header, panel_width, panel_height = 960, 170, 320, 260
    plot_width, plot_height = 250, 160
    height = header + 2 * panel_height + 30
    title = "Derived — where a corrector beats leaving the calibration alone"
    scenes = document["source"]["scene_count"]
    description = (
        "Scene mean of per-frame median reprojection shift (pixel P50) against each "
        f"single-axis fault on {scenes} locked nuScenes validation scenes. Lower is better."
    )
    metadata = {
        "document_sha256": document["document_sha256"],
        "evidence_type": "derived",
        "source_documents": {
            name: value["document_sha256"]
            for name, value in document["source"]["documents"].items()
        },
    }
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" aria-labelledby="title description">',
        f'<title id="title">{escape(title)}</title>',
        f'<desc id="description">{escape(description)} Squares show the paired 95% interval '
        "of identity versus the fixed-three-seed learned mean; intervals are pointwise.</desc>",
        f"<metadata>{escape(json.dumps(metadata, sort_keys=True))}</metadata>",
        "<style>text{font-family:system-ui,sans-serif;fill:#172638;font-size:12px}"
        ".heading{font-size:19px;font-weight:650}.panel{font-size:13px;font-weight:600}"
        ".note{font-size:11px;fill:#40586e}.grid{stroke:#dce3eb;stroke-width:1}"
        ".axis{stroke:#8493a5;stroke-width:1}</style>",
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
        f'<text x="20" y="30" class="heading">{escape(title)}</text>',
        '<text x="20" y="54">Pixel P50: scene mean of the per-frame median reprojection shift, '
        f"{scenes} locked nuScenes validation scenes. Lower is better.</text>",
        '<text x="20" y="74">Axes are CAM_FRONT optical-frame axes: roll = tilt, pitch = pan, '
        "yaw = in-plane rotation; x lateral, y vertical, z forward.</text>",
        '<text x="20" y="94">Squares: identity → learned-mean paired 95% interval above 0 '
        "(green, learned better), below 0 (red) or containing 0 (white).</text>",
        '<text x="20" y="114">Dashed lines: break-even, the smallest fault from which the '
        "interval is above 0 at both signs. The mean is not an ensemble.</text>",
    ]
    for index, method in enumerate(_METHODS):
        legend = 20 + index * 150
        parts.append(
            f'<line x1="{legend}" x2="{legend + 18}" y1="140" y2="140" stroke="{COLORS[method]}" '
            f'stroke-width="2.5"/><text x="{legend + 24}" y="144">{escape(method)}</text>'
        )
    for index, axis in enumerate(_AXES):
        meaning = document["axes"][axis]
        left = (index % 3) * panel_width
        top = header + (index // 3) * panel_height
        plot_left, plot_top = left + 52, top + 30
        keys = sorted(
            (key for key in verdicts if _level(key)[0] == axis), key=lambda key: _level(key)[1]
        )
        limit = max(abs(_level(key)[1]) for key in keys)
        unit, word = _UNIT_LABEL[meaning["unit"]]

        def sx(level: float, left: float = plot_left, limit: float = limit) -> float:
            return left + (level + limit) / (2 * limit) * plot_width

        def sy(value: float, top: float = plot_top) -> float:
            return top + plot_height - value / top_value * plot_height

        parts.append(
            f'<g data-axis="{escape(axis)}"><text x="{left + 20}" y="{top + 18}" class="panel">'
            f"{escape(axis)} = {escape(meaning['physical'])}</text>"
        )
        for tick in range(5):
            value = top_value * tick / 4
            parts.append(
                f'<line class="grid" x1="{plot_left}" x2="{plot_left + plot_width}" '
                f'y1="{sy(value):.2f}" y2="{sy(value):.2f}"/><text text-anchor="end" '
                f'x="{plot_left - 6}" y="{sy(value) + 4:.2f}">{value:g}</text>'
            )
        for key in keys:
            level = _level(key)[1]
            x = sx(level)
            parts.append(
                f'<line class="axis" x1="{x:.2f}" x2="{x:.2f}" y1="{plot_top + plot_height}" '
                f'y2="{plot_top + plot_height + 4}"/>'
            )
            if level in (-limit, -limit / 2, 0.0, limit / 2, limit):
                parts.append(
                    f'<text text-anchor="middle" x="{x:.2f}" y="{plot_top + plot_height + 16}">'
                    f"{level:g}</text>"
                )
            verdict = verdicts[key]
            if verdict == "unavailable":
                parts.append(
                    f'<text text-anchor="middle" x="{x:.2f}" y="{plot_top + plot_height + 36}">'
                    "&#215;</text>"
                )
            else:
                parts.append(
                    f'<rect x="{x - 4:.2f}" y="{plot_top + plot_height + 26}" width="8" '
                    f'height="8" {_VERDICT_STYLE[verdict]}><title>{escape(key)}: '
                    f"{escape(verdict.replace('_', ' '))}</title></rect>"
                )
        magnitude = break_even[axis]["magnitude"]
        if magnitude is None:
            label = f"no break-even up to ±{limit:g}{unit}"
        else:
            label = f"break-even ±{magnitude:g}{unit}"
            for sign in (-1, 1):
                x = sx(sign * magnitude)
                parts.append(
                    f'<line x1="{x:.2f}" x2="{x:.2f}" y1="{plot_top}" y2="{plot_top + plot_height}" '
                    'stroke="#15803d" stroke-width="1" stroke-dasharray="4 3"/>'
                )
        parts.append(
            f'<text x="{plot_left}" y="{plot_top + plot_height + 52}" class="note">'
            f"fault ({word}) · {escape(label)}</text>"
        )
        for method in _METHODS:
            color = COLORS[method]
            points = [(key, series[method][key]) for key in keys]
            path = []
            connected = False
            for key, value in points:
                if value is None:
                    connected = False
                    continue
                path.append(f"{'L' if connected else 'M'}{sx(_level(key)[1]):.2f},{sy(value):.2f}")
                connected = True
            parts.append(
                f'<path data-series="{escape(method)}" d="{" ".join(path)}" fill="none" '
                f'stroke="{color}" stroke-width="1.6"/>'
            )
            for key, value in points:
                if value is not None:
                    source = pointer.format(method=method, condition=key)
                    parts.append(
                        f'<circle cx="{sx(_level(key)[1]):.2f}" cy="{sy(value):.2f}" r="2.4" '
                        f'fill="{color}"><title>{escape(method)} · {escape(key)}: '
                        f"{value:.2f} px · {escape(source)}</title></circle>"
                    )
        parts.append("</g>")
    parts.append(
        f'<text x="20" y="{height - 10}" class="note">Derived by '
        "bevcalib.analysis.operating_envelope from the v1.0.0 metrics.json and "
        "intervals.json. Not detector or vehicle safety evidence.</text>"
    )
    parts.append("</svg>\n")
    return "\n".join(parts)
