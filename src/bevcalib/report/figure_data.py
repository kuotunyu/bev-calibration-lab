"""Predeclared figure panels from audited publication rows, without reaggregation."""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from bevcalib.analysis.claims import ClaimV1
from bevcalib.analysis.formal_claims import PublicationRow
from bevcalib.metrics.reprojection import RANGE_BINS
from bevcalib.perturbations.schedule import ROTATION_SINGLE_AXIS_DEG, TRANSLATION_SINGLE_AXIS_M
from bevcalib.report.scalar_binding import bind_scalar

METHODS = ("identity", "classical", "learned-17", "learned-42", "learned-73")
AXES = ("roll", "pitch", "yaw", "x", "y", "z")
FIXED_MEAN = "learned-fixed-three-seed-mean"


@dataclass(frozen=True)
class FigureCell:
    label: str
    pointer: str
    value: int | float | None
    claim_id: str | None


@dataclass(frozen=True)
class FigurePoint:
    series: str
    x: float
    value: int | float | None
    low: int | float | None
    high: int | float | None
    document: str
    document_sha256: str
    cells: tuple[FigureCell, ...]
    reason: str | None


@dataclass(frozen=True)
class FigurePanel:
    key: str
    x_unit: str
    y_unit: str
    ticks: tuple[tuple[float, str], ...]
    points: tuple[FigurePoint, ...]


def figure_panels(
    rows: Iterable[PublicationRow],
    document_hashes: Mapping[str, str],
    scalar_claims: Mapping[tuple[str, str], ClaimV1],
) -> dict[str, tuple[FigurePanel, ...]]:
    """Map the fixed inventory; caller must audit the original five-document set.

    Pointers, not display labels, select the estimand. Every displayed support or
    interval scalar uses the same exact binding as tables. Identity timing stays
    in the separate formal report; no timing point is used as a pose observation.
    """
    indexed = {}
    for row in rows:
        if row.document not in ("metrics", "recovery"):
            continue
        parts = row.cells[0][1].split("/")
        group, series, condition = parts[1:4]
        if condition.startswith("time:"):
            continue
        metric = parts[4].replace("~1", "/").replace("~0", "~") if row.document == "metrics" else ""
        if row.document == "metrics" and not metric.startswith("bev_frame_mean_m/"):
            continue
        key = (row.document, group, series, condition, metric)
        if key in indexed:
            raise ValueError("duplicate figure source row")
        indexed[key] = row

    def point(
        document: str,
        group: str,
        source: str,
        condition: str,
        metric: str,
        series: str,
        x: float,
    ) -> FigurePoint:
        key = (document, group, source, condition, metric)
        if key not in indexed:
            raise ValueError(f"missing required figure source row: {key}")
        row = indexed[key]
        sha256 = document_hashes[document]
        cells = tuple(
            FigureCell(
                label, pointer, value, bind_scalar(document, pointer, value, sha256, scalar_claims)
            )
            for label, pointer, value in row.cells
        )
        values = {cell.label: cell.value for cell in cells}
        field = "improvement" if group == "comparisons" else "value"
        return FigurePoint(
            series,
            x,
            values[field],
            values.get("interval low"),
            values.get("interval high"),
            document,
            sha256,
            cells,
            row.reason,
        )

    recovery = []
    bev = []
    for axis in AXES:
        rotation = axis in ("roll", "pitch", "yaw")
        levels = ROTATION_SINGLE_AXIS_DEG if rotation else TRANSLATION_SINGLE_AXIS_M
        ticks = tuple((level, f"{level:g}") for level in levels)
        for baseline in (None, "identity", "classical"):
            group = "runs" if baseline is None else "comparisons"
            methods = (
                METHODS
                if baseline is None
                else (
                    (*METHODS[1:], FIXED_MEAN)
                    if baseline == "identity"
                    else (*METHODS[2:], FIXED_MEAN)
                )
            )
            suffix = "absolute" if baseline is None else f"vs-{baseline}"
            points = tuple(
                point(
                    "recovery",
                    group,
                    method if baseline is None else f"{baseline}->{method}",
                    f"{axis}:{level:g}",
                    "",
                    method,
                    level,
                )
                for method in methods
                for level in levels
            )
            recovery.append(
                FigurePanel(
                    f"{axis}:{suffix}",
                    "degrees" if rotation else "metres",
                    "percent" if baseline is None else "percentage points",
                    ticks,
                    points,
                )
            )
        for level in levels:
            condition = f"{axis}:{level:g}"
            points = tuple(
                point(
                    "metrics",
                    "runs",
                    method,
                    condition,
                    f"bev_frame_mean_m/{bin_name}",
                    method,
                    float(index),
                )
                for method in METHODS
                for index, bin_name in enumerate(RANGE_BINS)
            )
            bev.append(
                FigurePanel(
                    condition,
                    "GT range bin (metres)",
                    "metres",
                    tuple((float(i), name) for i, name in enumerate(RANGE_BINS)),
                    points,
                )
            )
    return {"recovery-by-fault-level": tuple(recovery), "bev-error-by-range": tuple(bev)}
