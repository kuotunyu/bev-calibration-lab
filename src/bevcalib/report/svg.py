"""Deterministic, offline scientific SVGs from exactly bound panel data."""

import html
import json
import math

from bevcalib.report.figure_data import FigurePanel, FigurePoint

COLORS = {
    "identity": "#64748b",
    "classical": "#007f7f",
    "learned-17": "#2563eb",
    "learned-42": "#b45309",
    "learned-73": "#be185d",
    "learned-fixed-three-seed-mean": "#6d28d9",
}


def _tooltip(point: FigurePoint) -> str:
    text = [point.series, f"{point.document} SHA256 {point.document_sha256}"]
    for cell in point.cells:
        text.append(f"{cell.label}: {cell.value} | {cell.pointer} | claim {cell.claim_id}")
    if point.reason is not None:
        text.append(f"Unavailable: {point.reason}")
    return html.escape("\n".join(text))


def render_svg(name: str, panels: tuple[FigurePanel, ...], *, evidence_type: str) -> str:
    """Render already audited/mapped data; never load or change experimental outputs."""
    if name not in ("recovery-by-fault-level", "bev-error-by-range"):
        raise ValueError("unknown formal figure")
    if evidence_type not in ("observed", "synthetic") or not panels:
        raise ValueError("formal figure needs declared evidence and nonempty panels")
    hashes: dict[str, str] = {}
    limits: dict[str, tuple[float, float]] = {}
    for panel in panels:
        low, high = {
            "percent": (0.0, 100.0),
            "percentage points": (-100.0, 100.0),
            "metres": (0.0, 1.0),
        }[panel.y_unit]
        low, high = limits.get(panel.y_unit, (low, high))
        for point in panel.points:
            if point.document in hashes and hashes[point.document] != point.document_sha256:
                raise ValueError("figure mixes source snapshots")
            hashes[point.document] = point.document_sha256
            for value in (point.value, point.low, point.high):
                if value is not None:
                    low, high = min(low, value), max(high, value)
        limits[panel.y_unit] = (low, high)

    title = (
        "Recovery by fault level" if name == "recovery-by-fault-level" else "BEV error by GT range"
    )
    title = f"{evidence_type.capitalize()} — {title}"
    description = (
        "Higher recovery / positive improvement is better. Comparisons use paired support; percentage points are not percent changes."
        if name == "recovery-by-fault-level"
        else "Lower BEV error is better. Shared vertical scale across all conditions; GT range bins retain unsupported 80+ values."
    )
    width, panel_width, panel_height, header = 1440, 480, 350, 180
    height = header + math.ceil(len(panels) / 3) * panel_height
    escape = html.escape
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title description">',
        f'<title id="title">{escape(title)}</title>',
        f'<desc id="description">{escape(description)} Point tooltips include exact values, support, source SHA and claim IDs. Nulls are not zero.</desc>',
        "<metadata>"
        + escape(
            json.dumps({"evidence_type": evidence_type, "document_sha256": hashes}, sort_keys=True)
        )
        + "</metadata>",
        "<style>text{font-family:system-ui,sans-serif;fill:#172638;font-size:12px}.heading{font-size:24px;font-weight:650}.panel-title{font-size:15px;font-weight:600}.note{font-size:11px}.grid{stroke:#dce3eb;stroke-width:1}.axis{stroke:#8493a5;stroke-width:1}</style>",
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
        f'<text x="24" y="36" class="heading">{escape(title)}</text>',
        f'<text x="24" y="62">{escape(description)}</text>',
        '<text x="24" y="84">All fixed seeds shown. Fixed-three-seed mean is a paired statistic, not an ensemble. Intervals are pointwise, not simultaneous.</text>',
        '<text x="24" y="106">Zero references across axes are not independent studies. Identity timing stress is separate in the full report. No real-vehicle safety claim.</text>',
    ]
    for index, (series, color) in enumerate(COLORS.items()):
        legend_x = 24 + index * 228
        parts.append(
            f'<circle cx="{legend_x}" cy="132" r="4" fill="{color}"/><text x="{legend_x + 10}" y="136">{escape(series)}</text>'
        )
    parts.append(
        '<text x="24" y="158" class="note">Hover a point for exact values and provenance. &#215; below an axis means unavailable; colors match the method legend.</text>'
    )

    for index, panel in enumerate(panels):
        left = (index % 3) * panel_width
        top = header + (index // 3) * panel_height
        plot_left, plot_top, plot_width, plot_height = left + 60, top + 42, 390, 180
        minimum_x, maximum_x = panel.ticks[0][0], panel.ticks[-1][0]
        minimum_y, maximum_y = limits[panel.y_unit]

        def sx(
            value: float,
            minimum: float = minimum_x,
            maximum: float = maximum_x,
            start: float = plot_left,
            length: float = plot_width,
        ) -> float:
            return start + (value - minimum) / (maximum - minimum) * length

        def sy(
            value: float,
            minimum: float = minimum_y,
            maximum: float = maximum_y,
            start: float = plot_top,
            length: float = plot_height,
        ) -> float:
            return start + length - (value - minimum) / (maximum - minimum) * length

        parts.append(
            f'<g data-panel="{escape(panel.key)}"><text x="{left + 24}" y="{top + 22}" class="panel-title">{escape(panel.key)} · {escape(panel.y_unit)}</text>'
        )
        for tick in range(5):
            value = minimum_y + (maximum_y - minimum_y) * tick / 4
            y = sy(value)
            parts.append(
                f'<line class="grid" x1="{plot_left}" x2="{plot_left + plot_width}" y1="{y:.3f}" y2="{y:.3f}"/><text text-anchor="end" x="{plot_left - 8}" y="{y + 4:.3f}">{value:.3g}</text>'
            )
        for value, label in panel.ticks:
            x = sx(value)
            parts.append(
                f'<line class="axis" x1="{x:.3f}" x2="{x:.3f}" y1="{plot_top + plot_height}" y2="{plot_top + plot_height + 4}"/><text text-anchor="end" transform="translate({x:.3f},{plot_top + plot_height + 15}) rotate(-35)">{escape(label)}</text>'
            )
        parts.append(
            f'<text x="{plot_left}" y="{top + 275}" class="note">{escape(panel.x_unit)} · unavailable lane below</text>'
        )
        for series_index, series in enumerate(dict.fromkeys(p.series for p in panel.points)):
            points = sorted((p for p in panel.points if p.series == series), key=lambda p: p.x)
            color = COLORS[series]
            path = []
            connected = False
            for point in points:
                if point.value is None:
                    connected = False
                else:
                    command = "L" if connected else "M"
                    path.append(f"{command}{sx(point.x):.3f},{sy(point.value):.3f}")
                    connected = True
            dash = ' stroke-dasharray="6 4"' if series == "learned-fixed-three-seed-mean" else ""
            parts.append(
                f'<path data-curve="{series}" d="{" ".join(path)}" fill="none" stroke="{color}" stroke-width="1.5"{dash}/>'
            )
            for point in points:
                primary = "improvement" if panel.y_unit == "percentage points" else "value"
                cell = next(cell for cell in point.cells if cell.label == primary)
                unavailable = ' data-unavailable="true"' if point.value is None else ""
                tooltip = _tooltip(point)
                if point.value is not None:
                    neighbors = [
                        other
                        for other in panel.points
                        if other.value is not None
                        and (sx(other.x) - sx(point.x)) ** 2
                        + (sy(other.value) - sy(point.value)) ** 2
                        <= 36
                    ]
                    if len(neighbors) > 1:
                        tooltip = "Overlapping markers (within two marker radii):\n" + "\n\n".join(
                            _tooltip(other) for other in neighbors
                        )
                parts.append(
                    f'<g data-series="{series}" data-claim="{escape(cell.claim_id or "")}" data-pointer="{escape(cell.pointer)}"{unavailable}><title>{tooltip}</title>'
                )
                x = sx(point.x)
                if point.value is None:
                    parts.append(
                        f'<text x="{x:.3f}" y="{top + 290 + series_index * 10}" text-anchor="middle" style="fill:{color}">&#215;</text>'
                    )
                else:
                    if point.low is not None and point.high is not None:
                        parts.append(
                            f'<line data-interval="pointwise" x1="{x:.3f}" x2="{x:.3f}" y1="{sy(point.low):.3f}" y2="{sy(point.high):.3f}" stroke="{color}" stroke-width="2"/>'
                        )
                    parts.append(
                        f'<circle cx="{x:.3f}" cy="{sy(point.value):.3f}" r="3" fill="{color}"/>'
                    )
                parts.append("</g>")
        parts.append("</g>")
    parts.append("</svg>\n")
    return "\n".join(parts)
