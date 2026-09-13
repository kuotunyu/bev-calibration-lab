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
    assert '<html lang="zh-Hant">' in first
    assert "<script src=" not in first
    assert 'id="calibration-explorer"' in first
    assert "synthetic" in first
    assert "MIT" in first
    assert "nuScenes" in first


def test_figure_keeps_unavailable_reconstruction_explicit(monkeypatch: pytest.MonkeyPatch) -> None:
    from bevcalib.report.explorer import explorer_figure

    monkeypatch.setattr("bevcalib.report.explorer.reconstruct_ground_contact", lambda *args: None)
    figure = explorer_figure()
    assert figure["data"][3]["x"] == [None, None, None]
    assert "unavailable" in figure["layout"]["title"]["text"]
