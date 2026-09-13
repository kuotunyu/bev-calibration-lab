"""Small synthetic publication rows exercise mapping, not full artifact acceptance."""

from dataclasses import replace
from xml.etree import ElementTree as ET

import pytest

from bevcalib.analysis.claims import ClaimV1, ReportScalarBinding
from bevcalib.analysis.formal_claims import PublicationRow, claim_id, pointer_token

METHODS = ("identity", "classical", "learned-17", "learned-42", "learned-73")
BINS = ("0-10", "10-20", "20-40", "40-80", "80+")
ROTATIONS = (-2, -1, -0.5, -0.25, -0.1, 0, 0.1, 0.25, 0.5, 1, 2)
TRANSLATIONS = (-0.2, -0.1, -0.05, -0.02, 0, 0.02, 0.05, 0.1, 0.2)


@pytest.fixture
def mapped_source():  # type: ignore[no-untyped-def]
    rows = []
    claims = {}
    comparisons = (
        "identity->classical",
        *(
            f"{before}->{after}"
            for before in ("identity", "classical")
            for after in (*METHODS[2:], "learned-fixed-three-seed-mean")
        ),
    )

    def add(document, group, series, condition, metric=""):  # type: ignore[no-untyped-def]
        path = f"/{group}/{series}/{condition}"
        if metric:
            path += "/" + pointer_token(metric)
        missing = metric.endswith("80+")
        fields = [("improvement" if group == "comparisons" else "value", None if missing else 25.0)]
        if group == "comparisons":
            fields = [("before", 60.0), ("after", 85.0), *fields]
            fields.extend((("interval low", -5.0), ("interval high", 40.0)))
        fields.append(("scenes", 0 if missing else 30))
        cells = tuple(
            (
                name,
                path + "/" + ("support/scenes" if name == "scenes" else name.replace(" ", "/")),
                value,
            )
            for name, value in fields
        )
        rows.append(
            PublicationRow(
                document,
                "label not used for identity",
                "unit",
                cells,
                "no supported objects" if missing else None,
            )
        )
        for _, pointer, value in cells:
            if value is not None:
                claims[document, pointer] = ClaimV1(
                    claim_id=claim_id(document, pointer),
                    text=f"Synthetic {value}",
                    evidence_type="synthetic",
                    protocol_hash="a" * 64,
                    dataset_manifest_hash="b" * 64,
                    artifact_path=f"evidence/{document}.json",
                    metric_path=pointer,
                    status="verified",
                    report_binding=ReportScalarBinding(
                        expected_summary_sha256="c" * 64, expected_value=value
                    ),
                )

    for axis in ("roll", "pitch", "yaw", "x", "y", "z"):
        for level in ROTATIONS if axis in ("roll", "pitch", "yaw") else TRANSLATIONS:
            condition = f"{axis}:{level:g}"
            for method in METHODS:
                add("recovery", "runs", method, condition)
                for bin_name in BINS:
                    add("metrics", "runs", method, condition, f"bev_frame_mean_m/{bin_name}")
            for comparison in comparisons:
                add("recovery", "comparisons", comparison, condition)
    add("metrics", "runs", "identity", "time:0", "bev_frame_mean_m/0-10")
    add("metrics", "runs", "identity", "yaw:0", "rotation_geodesic_deg")
    rows.append(PublicationRow("timing", "separate timing stress", "ms", ()))
    return rows, {"recovery": "c" * 64, "metrics": "c" * 64}, claims


def test_every_predeclared_panel_and_series_is_preserved(mapped_source) -> None:  # type: ignore[no-untyped-def]
    from bevcalib.report.figure_data import figure_panels

    result = figure_panels(*mapped_source)
    recovery = result["recovery-by-fault-level"]
    bev = result["bev-error-by-range"]
    assert len(recovery) == 18
    assert len(bev) == 60
    assert sum(len(panel.points) for panel in recovery) == 840
    assert sum(len(panel.points) for panel in bev) == 1500
    assert [panel.key for panel in recovery[:3]] == [
        "roll:absolute",
        "roll:vs-identity",
        "roll:vs-classical",
    ]
    assert recovery[0].x_unit == "degrees"
    assert recovery[9].x_unit == "metres"
    assert recovery[0].y_unit == "percent"
    assert recovery[1].y_unit == "percentage points"
    assert recovery[0].ticks == tuple((float(x), f"{x:g}") for x in ROTATIONS)
    assert {p.series for p in recovery[0].points} == set(METHODS)
    assert {p.series for p in recovery[1].points} == {*METHODS[1:], "learned-fixed-three-seed-mean"}
    assert {p.series for p in recovery[2].points} == {*METHODS[2:], "learned-fixed-three-seed-mean"}
    assert all(any(p.x == 0 for p in panel.points) for panel in recovery)
    assert {p.series for p in bev[0].points} == set(METHODS)
    assert bev[0].ticks == tuple((float(i), name) for i, name in enumerate(BINS))
    assert all("time" not in panel.key for panel in (*recovery, *bev))


def test_null_support_interval_and_exact_claims_survive_mapping(mapped_source) -> None:  # type: ignore[no-untyped-def]
    from bevcalib.report.figure_data import figure_panels

    result = figure_panels(*mapped_source)
    missing = next(p for p in result["bev-error-by-range"][0].points if p.x == 4)
    assert missing.value is None
    assert missing.reason == "no supported objects"
    assert [(cell.label, cell.value, cell.claim_id) for cell in missing.cells[:1]] == [
        ("value", None, None)
    ]
    assert missing.cells[-1].value == 0
    assert missing.cells[-1].claim_id is not None
    paired = result["recovery-by-fault-level"][1].points[0]
    assert (paired.value, paired.low, paired.high) == (25, -5, 40)
    assert paired.document_sha256 == "c" * 64
    improvement = next(cell for cell in paired.cells if cell.label == "improvement")
    assert improvement.pointer == "/comparisons/identity->classical/roll:-2/improvement"
    assert improvement.claim_id == claim_id("recovery", improvement.pointer)


@pytest.mark.parametrize("case", ["missing-row", "duplicate-row", "missing-claim", "wrong-value"])
def test_incomplete_or_ambiguous_figure_data_is_refused(mapped_source, case: str) -> None:  # type: ignore[no-untyped-def]
    from bevcalib.report.figure_data import figure_panels

    rows, hashes, claims = mapped_source
    if case == "missing-row":
        rows.pop(0)
    elif case == "duplicate-row":
        rows.append(rows[0])
    elif case == "missing-claim":
        del claims["recovery", rows[0].cells[0][1]]
    else:
        name, pointer, value = rows[0].cells[0]
        rows[0] = replace(rows[0], cells=((name, pointer, value + 1), *rows[0].cells[1:]))
    with pytest.raises(ValueError):
        figure_panels(rows, hashes, claims)


def test_mapping_is_order_independent_and_does_not_mutate_inputs(mapped_source) -> None:  # type: ignore[no-untyped-def]
    from bevcalib.report.figure_data import figure_panels

    rows, hashes, claims = mapped_source
    original = tuple(rows)
    assert figure_panels(rows, hashes, claims) == figure_panels(reversed(rows), hashes, claims)
    assert tuple(rows) == original


def test_full_synthetic_panel_inventory_renders_without_losing_points(mapped_source) -> None:  # type: ignore[no-untyped-def]
    from bevcalib.report.figure_data import figure_panels
    from bevcalib.report.svg import render_svg

    mapped = figure_panels(*mapped_source)
    ns = {"s": "http://www.w3.org/2000/svg"}
    for name, panels in mapped.items():
        svg = render_svg(name, panels, evidence_type="synthetic")
        assert svg == render_svg(name, panels, evidence_type="synthetic")
        root = ET.fromstring(svg)
        assert len(root.findall(".//s:g[@data-panel]", ns)) == len(panels)
        points = root.findall(".//s:g[@data-series]", ns)
        assert len(points) == sum(len(panel.points) for panel in panels)
        expected = {
            cell.claim_id
            for panel in panels
            for point in panel.points
            for cell in point.cells
            if cell.claim_id
        }
        displayed = "".join(root.itertext())
        assert all(identifier in displayed for identifier in expected)
        if name == "bev-error-by-range":
            assert len(root.findall(".//s:g[@data-unavailable='true']", ns)) == 300
