"""The operating-envelope SVG renders a derived document and nothing else."""

from __future__ import annotations

import json
from typing import Any
from xml.etree import ElementTree as ET

import pytest

SVG = "{http://www.w3.org/2000/svg}"


def small_document(**overrides: Any) -> dict[str, Any]:
    axes = {
        axis: {"physical": physical, "frame_axis": "camera", "unit": unit}
        for axis, physical, unit in (
            ("roll", "tilt", "degree"),
            ("pitch", "pan", "degree"),
            ("yaw", "in-plane rotation", "degree"),
            ("x", "lateral offset", "metre"),
            ("y", "vertical offset", "metre"),
            ("z", "forward offset", "metre"),
        )
    }
    keys = [
        *(
            f"{axis}:{level:g}"
            for axis in ("roll", "pitch", "yaw")
            for level in (-2.0, -1.0, 0.0, 1.0, 2.0)
        ),
        *(f"{axis}:{level:g}" for axis in ("x", "y", "z") for level in (-0.2, -0.1, 0.0, 0.1, 0.2)),
    ]
    methods = ("identity", "classical", "learned-17", "learned-42", "learned-73")
    series: dict[str, dict[str, float | None]] = {
        method: dict.fromkeys(keys, 10.0 + index) for index, method in enumerate(methods)
    }
    series["classical"]["roll:1"] = None
    verdict = dict.fromkeys(keys, "inconclusive") | {
        "roll:2": "after_better",
        "roll:-2": "after_better",
        "roll:0": "before_better",
        "yaw:1": "unavailable",
    }
    document: dict[str, Any] = {
        "evidence_type": "derived",
        "document_sha256": "d" * 64,
        "source": {"scene_count": 30, "documents": {"metrics": {"document_sha256": "a" * 64}}},
        "axes": axes,
        "break_even": {
            "identity->learned-fixed-three-seed-mean": {
                axis: {"magnitude": 2.0 if axis == "roll" else None} for axis in axes
            }
        },
        "series": {
            "pixel_frame_p50_px": series,
            "verdict": verdict,
            "pointers": {"pixel_frame_p50_px": "metrics.json#/runs/{method}/{condition}/p50"},
        },
    }
    return document | overrides


def test_figure_is_accessible_deterministic_and_carries_its_sources() -> None:
    from bevcalib.report.envelope_figure import render_envelope_svg

    document = small_document()
    first = render_envelope_svg(document)
    assert first == render_envelope_svg(document)
    root = ET.fromstring(first)
    assert root.get("role") == "img"
    assert root.find(f"{SVG}title").text.startswith("Derived")  # type: ignore[union-attr]
    metadata = json.loads(root.find(f"{SVG}metadata").text)  # type: ignore[arg-type,union-attr]
    assert metadata == {
        "document_sha256": "d" * 64,
        "evidence_type": "derived",
        "source_documents": {"metrics": "a" * 64},
    }
    panels = [group.get("data-axis") for group in root.iter(f"{SVG}g")]
    assert panels == ["roll", "pitch", "yaw", "x", "y", "z"]
    text = "".join(root.itertext())
    assert "break-even ±2°" in text
    assert "fault (m) · no break-even up to ±0.2 m" in text
    assert "fault (deg) · break-even ±2°" in text
    assert "metrics.json#/runs/learned-73/z:0.2/p50" in text


def test_missing_values_break_lines_and_unavailable_intervals_are_marked() -> None:
    from bevcalib.report.envelope_figure import render_envelope_svg

    root = ET.fromstring(render_envelope_svg(small_document()))
    roll = next(group for group in root.iter(f"{SVG}g") if group.get("data-axis") == "roll")
    classical = next(
        path for path in roll.iter(f"{SVG}path") if path.get("data-series") == "classical"
    )
    assert classical.get("d", "").count("M") == 2
    yaw = next(group for group in root.iter(f"{SVG}g") if group.get("data-axis") == "yaw")
    assert chr(0xD7) in "".join(yaw.itertext())  # the unavailable mark
    squares = list(roll.iter(f"{SVG}rect"))
    assert [rect.get("fill") for rect in squares] == [
        "#15803d",
        "#ffffff",
        "#b91c1c",
        "#ffffff",
        "#15803d",
    ]


def test_only_a_derived_document_is_rendered() -> None:
    from bevcalib.report.envelope_figure import render_envelope_svg

    with pytest.raises(ValueError, match="requires a derived document"):
        render_envelope_svg(small_document(evidence_type="observed"))
