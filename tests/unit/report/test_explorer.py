"""Analytic geometry expectations independent of the explorer renderer."""

import json
import math

import pytest

from bevcalib.report.explorer import explorer_state


@pytest.mark.parametrize("axis", ["roll", "pitch", "yaw", "x", "y", "z"])
def test_zero_preserves_fixed_observations_and_ground(axis: str) -> None:
    state = explorer_state(axis, 0.0)
    assert state["evidence_type"] == "synthetic"
    assert state["observed_uv"][1] == pytest.approx([400, 300])
    for actual, expected in zip(state["overlay_uv"], state["observed_uv"], strict=True):
        assert actual == pytest.approx(expected)
    assert state["reconstructed_xy"][1] == pytest.approx([20, 0])
    assert state["bev_errors_m"] == pytest.approx([0, 0, 0], abs=1e-12)


@pytest.mark.parametrize("level", [-0.2, 0.2])
@pytest.mark.parametrize("axis", ["x", "y", "z"])
def test_translation_uses_fixed_pixels_and_source_side_metadata(axis: str, level: float) -> None:
    state = explorer_state(axis, level)
    assert state["unit"] == "m"
    assert state["observed_uv"][1] == pytest.approx([400, 300])
    expected = {
        "x": ([400, 240 + 1200 / (20 + level)], [20 - level, 0]),
        "y": ([400 - 40 * level, 300], [20, -level]),
        "z": ([400, 300 - 40 * level], [20 * (1.5 - level) / 1.5, 0]),
    }[axis]
    assert state["overlay_uv"][1] == pytest.approx(expected[0])
    assert state["reconstructed_xy"][1] == pytest.approx(expected[1])


@pytest.mark.parametrize("level", [-2.0, 2.0])
def test_yaw_reconstruction_has_independent_analytic_solution(level: float) -> None:
    state = explorer_state("yaw", level)
    theta = math.radians(level)
    assert state["unit"] == "degrees"
    assert state["reconstructed_xy"][1] == pytest.approx(
        [20 * math.cos(theta), -20 * math.sin(theta)]
    )
    assert state["bev_errors_m"][1] == pytest.approx(40 * abs(math.sin(theta / 2)))


@pytest.mark.parametrize("axis,level", [("time", 0), ("yaw", 3), ("x", 0.03), ("z", float("nan"))])
def test_invalid_condition_is_refused(axis: str, level: float) -> None:
    with pytest.raises(ValueError, match="declared extrinsic"):
        explorer_state(axis, level)


def test_missing_ground_intersection_remains_explicit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("bevcalib.report.explorer.reconstruct_ground_contact", lambda *args: None)
    state = explorer_state("yaw", 1.0)
    assert state["reconstructed_xy"] == [None, None, None]
    assert state["bev_errors_m"] == [None, None, None]


@pytest.mark.parametrize("level", [-2.0, 2.0])
def test_roll_oblique_point_matches_ray_plane_intersection(level: float) -> None:
    theta = math.radians(level)
    sine, cosine = math.sin(theta), math.cos(theta)
    distance = 1.5 * cosine / (1.5 * cosine - 2 * sine)
    state = explorer_state("roll", level)
    assert state["reconstructed_xy"][0] == pytest.approx(
        [10 * distance, 1.5 * sine + distance * (-2 * cosine - 1.5 * sine)]
    )
    assert state["overlay_uv"][0] == pytest.approx(
        [400 + 160 * cosine, 240 + 80 * (1.5 + 2 * sine)]
    )


@pytest.mark.parametrize("level", [-2.0, 2.0])
def test_pitch_center_point_matches_ray_plane_intersection(level: float) -> None:
    theta = math.radians(level)
    sine, cosine = math.sin(theta), math.cos(theta)
    distance = 1.5 * cosine / (1.5 * cosine - 20 * sine)
    state = explorer_state("pitch", level)
    assert state["reconstructed_xy"][1] == pytest.approx(
        [-1.5 * sine + distance * (20 * cosine + 1.5 * sine), 0]
    )
    assert state["overlay_uv"][1] == pytest.approx(
        [400, 240 + 800 * (1.5 + 20 * sine) / (20 * cosine)]
    )


def test_every_declared_state_is_finite_deterministic_and_keeps_observations() -> None:
    from bevcalib.artifacts.result_documents import condition_inventory

    reference = explorer_state("yaw", 0)["observed_uv"]
    conditions = condition_inventory("classical")
    assert len(conditions) == 60
    for axis, level in conditions:
        state = explorer_state(axis, level)
        assert state["observed_uv"] == reference
        assert all(state["overlay_valid"])
        assert all(value is not None and value >= 0 for value in state["bev_errors_m"])
        assert json.dumps(state, allow_nan=False, sort_keys=True) == json.dumps(
            explorer_state(axis, level), allow_nan=False, sort_keys=True
        )


def test_axis_buttons_replace_slider_and_reset_to_zero() -> None:
    from bevcalib.report.explorer import explorer_figure

    figure = explorer_figure()
    buttons = figure["layout"]["updatemenus"][0]["buttons"]
    assert [button["label"] for button in buttons] == ["roll", "pitch", "yaw", "x", "y", "z"]
    for button in buttons:
        axis = button["label"]
        data, layout = button["args"]
        slider = layout["sliders"][0]
        assert slider["steps"][slider["active"]]["label"] == "0"
        assert data["x"][0] == data["x"][1]
        for step in slider["steps"]:
            changed, changed_layout = step["args"]
            state = explorer_state(axis, float(step["label"]))
            assert changed["x"][1] == [uv[0] for uv in state["overlay_uv"]]
            assert changed["y"][3] == [xy[0] for xy in state["reconstructed_xy"]]
            assert axis in changed_layout["title"]["text"]
            assert state["unit"] in changed_layout["title"]["text"]
            assert "BEV" in changed_layout["title"]["text"]


def test_every_slider_state_keeps_all_ground_points_in_view() -> None:
    from bevcalib.report.explorer import explorer_figure

    buttons = explorer_figure()["layout"]["updatemenus"][0]["buttons"]
    for button in buttons:
        for step in button["args"][1]["sliders"][0]["steps"]:
            data, layout = step["args"]
            for key, coordinate in (("xaxis2", "x"), ("yaxis2", "y")):
                low, high = layout[key]["range"]
                assert all(low <= value <= high for value in data[coordinate][3])


def test_html_is_byte_reproducible_and_self_contained() -> None:
    from bevcalib.report.explorer import build_explorer

    first = build_explorer()
    assert first == build_explorer()
    assert '<html lang="en">' in first
    assert "<script src=" not in first
    assert 'id="calibration-explorer"' in first
    assert "synthetic" in first
    assert "MIT" in first
    assert "nuScenes" in first


def test_footer_names_only_the_licence_marks_the_bundle_keeps() -> None:
    """The bundle keeps MapLibre GL JS's identifier and a link, not its licence text."""
    from bevcalib.report.explorer import build_explorer

    html = build_explorer()
    footer = html.split("<footer>", 1)[1].split("</footer>", 1)[0]
    assert "keeps its copyright line and MIT licence identifier" in footer
    assert "the BSD-3-Clause identifier of the MapLibre GL JS code it includes" in footer
    assert "a link to the full MapLibre GL JS licence text" in footer
    assert "notice" not in footer
    assert "Plotly, Inc." in html and "Licensed under the MIT license" in html
    assert "@license 3-Clause BSD. Full text of license: https://github.com/maplibre/" in html


def test_figure_keeps_unavailable_reconstruction_explicit(monkeypatch: pytest.MonkeyPatch) -> None:
    from bevcalib.report.explorer import explorer_figure

    monkeypatch.setattr("bevcalib.report.explorer.reconstruct_ground_contact", lambda *args: None)
    figure = explorer_figure()
    assert figure["data"][3]["x"] == [None, None, None]
    assert "unavailable" in figure["layout"]["title"]["text"]


def test_coincident_markers_are_distinguishable_without_moving_coordinates() -> None:
    from bevcalib.report.explorer import explorer_figure

    traces = explorer_figure()["data"]
    for fixed, projected in ((traces[0], traces[1]), (traces[2], traces[3])):
        assert fixed["x"] == projected["x"]
        assert fixed["y"] == pytest.approx(projected["y"])
        assert fixed["marker"]["symbol"] == "circle-open"
        assert fixed["marker"]["size"] >= projected["marker"]["size"] + 8


def test_explorer_controls_and_readout_are_outside_two_plot_containers() -> None:
    from html.parser import HTMLParser

    from bevcalib.report.explorer import build_explorer

    class Elements(HTMLParser):
        def __init__(self) -> None:
            super().__init__()
            self.ids: dict[str, tuple[str, dict[str, str | None]]] = {}

        def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
            attributes = dict(attrs)
            identity = attributes.get("id")
            if identity:
                self.ids[identity] = (tag, attributes)

    html = build_explorer()
    elements = Elements()
    elements.feed(html)
    for identity in ("camera-plot", "ground-plot", "explorer-summary", "point-values"):
        assert identity in elements.ids
    tag, attributes = elements.ids["fault-level"]
    assert tag == "input" and attributes["type"] == "range"
    assert elements.ids["reset-fault"][0] == "button"
    assert elements.ids["show-fixed"][1]["type"] == "checkbox"
    assert elements.ids["show-projected"][1]["type"] == "checkbox"
    for point in range(3):
        assert f"point-{point}-ground" in elements.ids
    assert 'id="explorer-states" type="application/json"' in html
    assert "切換軸會回到零故障" in html
    assert "重合" in html


def test_page_is_english_first_with_chinese_labels_and_states_the_axis_mapping() -> None:
    """The formal study names camera optical-frame axes; this explorer names vehicle axes."""
    from bevcalib.report.explorer import build_explorer

    html = build_explorer()
    note = html.split('<aside class="axis-note"', 1)[1].split("</aside>", 1)[0]
    for mapping in (
        "Formal roll (tilt) \u2248 explorer pitch",
        "formal pitch (pan) = explorer yaw",
        "formal yaw (in-plane rotation) \u2248 explorer roll with the opposite sign",
        "Formal x (lateral) = explorer y",
        "formal y (vertical) = explorer z",
        "formal z (forward) = explorer x with the opposite sign",
    ):
        assert mapping in note
    assert 'lang="zh-Hant"' in note
    for axis, meaning in (
        ("roll", "in-plane"),
        ("pitch", "tilt"),
        ("yaw", "pan"),
        ("x", "forward"),
        ("y", "lateral"),
        ("z", "vertical"),
    ):
        assert f'data-axis="{axis}"' in html
        assert f">{axis} \u00b7 {meaning}</button>" in html


@pytest.mark.parametrize(
    ("formal", "explorer", "sign", "tolerance_px"),
    [
        ("pitch", "yaw", 1, 1e-9),
        ("x", "y", 1, 1e-9),
        ("y", "z", 1, 1e-9),
        ("z", "x", -1, 1e-9),
        ("roll", "pitch", 1, 1.0),
        ("yaw", "roll", -1, 5.0),
    ],
)
def test_axis_note_matches_a_formal_camera_side_fault(
    formal: str, explorer: str, sign: int, tolerance_px: float
) -> None:
    """A formal fault composed on the camera side lands where the note says it does."""
    import numpy as np

    from bevcalib.artifacts.result_documents import fault_for_condition
    from bevcalib.geometry.projection import project_camera
    from bevcalib.geometry.quaternions import matrix_to_quaternion
    from bevcalib.geometry.se3 import SE3, inverse, transform_points
    from bevcalib.perturbations.apply import apply_metadata_fault

    level = 2.0 if formal in ("roll", "pitch", "yaw") else 0.2
    points = np.array([[10.0, -2.0, 0.0], [20.0, 0.0, 0.0], [40.0, 2.0, 0.0]])
    intrinsic = np.array([[800.0, 0.0, 400.0], [0.0, 800.0, 240.0], [0.0, 0.0, 1.0]])
    camera_from_global = SE3(
        matrix_to_quaternion(np.array([[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]])),
        (0.0, 1.5, 0.0),
    )
    assumed = apply_metadata_fault(inverse(camera_from_global), fault_for_condition(formal, level))
    formal_uv = project_camera(transform_points(inverse(assumed), points), intrinsic, (800, 480)).uv
    explorer_uv = np.array(explorer_state(explorer, sign * level)["overlay_uv"])

    assert np.abs(explorer_uv - formal_uv).max() <= tolerance_px
    other = np.array(explorer_state(explorer, -sign * level)["overlay_uv"])
    assert np.abs(other - formal_uv).max() > 5.0
