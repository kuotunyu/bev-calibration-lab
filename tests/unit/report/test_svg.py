"""SVG geometry and provenance checks with explicit tiny synthetic panels."""

import json
from dataclasses import replace
from xml.etree import ElementTree as ET

import pytest

from bevcalib.report.figure_data import FigureCell, FigurePanel, FigurePoint

NS = {"s": "http://www.w3.org/2000/svg"}


def element(root: ET.Element, path: str) -> ET.Element:
    result = root.find(path, NS)
    assert result is not None
    return result


def content(root: ET.Element, path: str) -> str:
    return "".join(element(root, path).itertext())


def panel() -> FigurePanel:
    cells = (
        FigureCell("value", "/runs/identity/yaw:0/value", 25, "formal.value"),
        FigureCell("scenes", "/runs/identity/yaw:0/support/scenes", 30, "formal.scenes"),
    )
    first = FigurePoint("identity", -1, 25, 10, 40, "recovery", "c" * 64, cells, None)
    missing = replace(
        first,
        x=0,
        value=None,
        low=None,
        high=None,
        cells=(FigureCell("value", "/missing", None, None),),
        reason="unsupported <80+> & no objects",
    )
    last = replace(
        first, x=1, value=75, low=60, high=90, cells=(replace(cells[0], value=75), cells[1])
    )
    return FigurePanel(
        "yaw:absolute",
        "degrees",
        "percent",
        ((-1, "-1"), (0, "0"), (1, "1")),
        (first, missing, last),
    )


def test_svg_is_deterministic_valid_xml_with_exact_point_provenance() -> None:
    from bevcalib.report.svg import render_svg

    panels = (panel(),)
    text = render_svg("recovery-by-fault-level", panels, evidence_type="synthetic")
    assert text == render_svg("recovery-by-fault-level", panels, evidence_type="synthetic")
    root = ET.fromstring(text)
    assert root.attrib["role"] == "img"
    assert "Synthetic" in content(root, "s:title")
    metadata = json.loads(content(root, "s:metadata"))
    assert metadata["evidence_type"] == "synthetic"
    assert metadata["document_sha256"] == {"recovery": "c" * 64}
    points = root.findall(".//s:g[@data-series]", NS)
    assert len(points) == 3
    assert points[0].attrib["data-claim"] == "formal.value"
    tooltip = content(points[0], "s:title")
    assert "scenes: 30" in tooltip and "formal.scenes" in tooltip
    assert "/runs/identity/yaw:0/value" in tooltip
    assert "c" * 64 in tooltip
    assert "unsupported <80+> & no objects" in content(points[1], "s:title")
    assert root.findall(".//s:g[@data-unavailable='true']", NS)
    assert "http" not in text.replace("http://www.w3.org/2000/svg", "")


def test_missing_observation_breaks_line_and_is_not_a_zero_measurement() -> None:
    from bevcalib.report.svg import render_svg

    root = ET.fromstring(
        render_svg("recovery-by-fault-level", (panel(),), evidence_type="observed")
    )
    path = element(root, ".//s:path[@data-curve='identity']").attrib["d"]
    assert path.count("M") == 2 and "L" not in path
    missing = element(root, ".//s:g[@data-unavailable='true']")
    assert missing.find("s:circle", NS) is None
    assert "Observed" in content(root, "s:title")


def test_error_bars_and_increasing_values_use_correct_vertical_direction() -> None:
    from bevcalib.report.svg import render_svg

    root = ET.fromstring(
        render_svg("recovery-by-fault-level", (panel(),), evidence_type="synthetic")
    )
    points = root.findall(".//s:g[@data-series]", NS)
    first = element(points[0], "s:circle")
    last = element(points[2], "s:circle")
    assert float(first.attrib["cx"]) < float(last.attrib["cx"])
    assert float(first.attrib["cy"]) > float(last.attrib["cy"])
    bar = element(points[0], "s:line[@data-interval]")
    assert float(bar.attrib["y1"]) > float(bar.attrib["y2"])


def test_contiguous_points_connect_and_mean_has_distinct_stroke() -> None:
    from bevcalib.report.svg import render_svg

    source = panel()
    points = tuple(
        replace(
            p,
            series="learned-fixed-three-seed-mean",
            value=-25 + i * 25,
            low=None,
            high=None,
            reason=None,
            cells=(
                FigureCell("improvement", "/comparisons/value", -25 + i * 25, "formal.improvement"),
            ),
        )
        for i, p in enumerate(source.points)
    )
    source = replace(source, key="yaw:vs-identity", y_unit="percentage points", points=points)
    root = ET.fromstring(
        render_svg("recovery-by-fault-level", (source,), evidence_type="synthetic")
    )
    curve = element(root, ".//s:path[@data-curve]")
    assert curve.attrib["d"].count("L") == 2
    assert curve.attrib["stroke-dasharray"] == "6 4"
    assert "percentage points" in "".join(root.itertext())


def test_bev_all_missing_remains_visible_and_has_shared_nonzero_scale() -> None:
    from bevcalib.report.svg import render_svg

    source = panel()
    missing = source.points[1]
    source = replace(
        source, key="yaw:0", x_unit="GT range bin (metres)", y_unit="metres", points=(missing,)
    )
    root = ET.fromstring(render_svg("bev-error-by-range", (source,) * 4, evidence_type="synthetic"))
    assert len(root.findall(".//s:g[@data-panel]", NS)) == 4
    assert len(root.findall(".//s:g[@data-unavailable='true']", NS)) == 4
    assert "Lower BEV error is better" in "".join(root.itertext())
    assert float(root.attrib["height"]) > 600


def test_overlapping_markers_keep_every_methods_provenance_accessible() -> None:
    from bevcalib.report.svg import render_svg

    source = panel()
    identity = source.points[0]
    classical = replace(
        identity,
        series="classical",
        cells=(replace(identity.cells[0], claim_id="formal.classical"),),
    )
    # Nearby values can also overlap at display resolution without being equal.
    learned = replace(
        identity,
        series="learned-17",
        value=26,
        cells=(replace(identity.cells[0], value=26, claim_id="formal.learned"),),
    )
    far = replace(
        identity,
        x=1,
        value=99,
        cells=(replace(identity.cells[0], value=99, claim_id="formal.far"),),
    )
    source = replace(source, points=(identity, classical, learned, far))
    root = ET.fromstring(
        render_svg("recovery-by-fault-level", (source,), evidence_type="synthetic")
    )
    topmost = element(root, ".//s:g[@data-series='learned-17']")
    tooltip = content(topmost, "s:title")
    assert "formal.value" in tooltip
    assert "formal.classical" in tooltip
    assert "formal.learned" in tooltip
    assert "formal.far" not in tooltip
    assert "Overlapping markers" in tooltip


@pytest.mark.parametrize("case", ["name", "evidence", "empty", "inconsistent-sha"])
def test_invalid_figure_identity_is_refused(case: str) -> None:
    from bevcalib.report.svg import render_svg

    name = "wrong" if case == "name" else "recovery-by-fault-level"
    evidence = "illustrative" if case == "evidence" else "synthetic"
    panels = () if case == "empty" else (panel(),)
    if case == "inconsistent-sha":
        source = panel()
        panels = (
            replace(
                source,
                points=(source.points[0], replace(source.points[2], document_sha256="d" * 64)),
            ),
        )
    with pytest.raises(ValueError):
        render_svg(name, panels, evidence_type=evidence)
