"""Fixed first-party plane observations for an offline calibration explorer."""

from __future__ import annotations

from typing import Any

import numpy as np

from bevcalib.artifacts.result_documents import condition_inventory, fault_for_condition
from bevcalib.geometry.projection import project_camera
from bevcalib.geometry.quaternions import matrix_to_quaternion
from bevcalib.geometry.se3 import SE3, transform_points
from bevcalib.operators.ground_contact import reconstruct_ground_contact
from bevcalib.perturbations.apply import apply_metadata_fault


def explorer_state(axis: str, level: float) -> dict[str, Any]:
    """Change assumed metadata while retaining the same true pixels and plane.

    Global X is forward, Y left, Z up. The camera is 1.5 metres above z=0,
    looking along +X, with optical axes right/down/forward. The fault acts on
    the global/source side here, explicitly not on a nuScenes sensor frame.
    """
    condition = next(
        (item for item in condition_inventory("classical") if item == (axis, level)), None
    )
    if condition is None:
        raise ValueError("explorer requires a declared extrinsic condition")
    points = np.array([[10.0, -2.0, 0.0], [20.0, 0.0, 0.0], [40.0, 2.0, 0.0]])
    intrinsic = np.array([[800.0, 0.0, 400.0], [0.0, 800.0, 240.0], [0.0, 0.0, 1.0]])
    camera_from_global = SE3(
        matrix_to_quaternion(np.array([[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]])),
        (0.0, 1.5, 0.0),
    )
    assumed = apply_metadata_fault(camera_from_global, fault_for_condition(*condition))
    observed = project_camera(transform_points(camera_from_global, points), intrinsic, (800, 480))
    overlay = project_camera(transform_points(assumed, points), intrinsic, (800, 480))
    contacts = [
        reconstruct_ground_contact((float(uv[0]), float(uv[1])), assumed, intrinsic, 0.0)
        for uv in observed.uv
    ]
    errors = [
        None if contact is None else float(np.linalg.norm(np.asarray(contact) - point[:2]))
        for contact, point in zip(contacts, points, strict=True)
    ]
    return {
        "evidence_type": "synthetic",
        "axis": axis,
        "level": level,
        "unit": "degrees" if axis in ("roll", "pitch", "yaw") else "m",
        "observed_uv": observed.uv.tolist(),
        "overlay_uv": overlay.uv.tolist(),
        "overlay_valid": overlay.valid.tolist(),
        "true_xy": points[:, :2].tolist(),
        "reconstructed_xy": [None if contact is None else list(contact) for contact in contacts],
        "bev_errors_m": errors,
    }


def _update(state: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    contacts = state["reconstructed_xy"]
    errors = state["bev_errors_m"]
    texts = ["unavailable" if value is None else f"BEV error: {value:.3f} m" for value in errors]
    mean = "unavailable" if None in errors else f"{sum(errors) / len(errors):.3f} m"
    data = {
        "x": [
            [uv[0] for uv in state["observed_uv"]],
            [uv[0] for uv in state["overlay_uv"]],
            [xy[1] for xy in state["true_xy"]],
            [None if xy is None else xy[1] for xy in contacts],
        ],
        "y": [
            [uv[1] for uv in state["observed_uv"]],
            [uv[1] for uv in state["overlay_uv"]],
            [xy[0] for xy in state["true_xy"]],
            [None if xy is None else xy[0] for xy in contacts],
        ],
        "text": [["Fixed observation"] * 3, texts, ["Known plane"] * 3, texts],
    }
    title = (
        f"{state['axis']} {state['level']:+g} {state['unit']} · Mean BEV error: {mean} · synthetic"
    )
    lateral = [*data["x"][2], *(value for value in data["x"][3] if value is not None)]
    forward = [*data["y"][2], *(value for value in data["y"][3] if value is not None)]
    return data, {
        "title": {"text": title, "font": {"size": 17}},
        "xaxis2": {
            "domain": [0.59, 1],
            "range": [min(-5, min(lateral) - 1), max(5, max(lateral) + 1)],
            "title": {"text": "Lateral Y (m)"},
        },
        "yaxis2": {
            "anchor": "x2",
            "range": [min(0, min(forward) - 1), max(45, max(forward) + 1)],
            "title": {"text": "Forward X (m)"},
        },
    }


def explorer_figure() -> dict[str, Any]:
    """Use declarative Plotly updates; all geometry is computed in Python."""
    buttons: list[dict[str, Any]] = []
    for axis in ("roll", "pitch", "yaw", "x", "y", "z"):
        levels = [
            level for candidate, level in condition_inventory("classical") if candidate == axis
        ]
        slider = {
            "active": levels.index(0.0),
            "currentvalue": {"prefix": f"{axis}: "},
            "pad": {"t": 55},
            "steps": [
                {
                    "label": f"{level:g}",
                    "method": "update",
                    "args": list(_update(explorer_state(axis, level))),
                }
                for level in levels
            ],
        }
        data, layout = _update(explorer_state(axis, 0.0))
        layout["sliders"] = [slider]
        buttons.append({"label": axis, "method": "update", "args": [data, layout]})
    initial_data, initial_layout = buttons[0]["args"]
    traces = []
    for index, (name, color, symbol) in enumerate(
        (
            ("Fixed camera observations", "#087e8b", "circle-open"),
            ("Assumed projection", "#c34b27", "x"),
            ("True ground contacts", "#087e8b", "circle-open"),
            ("Reconstructed contacts", "#c34b27", "x"),
        )
    ):
        traces.append(
            {
                "type": "scatter",
                "mode": "markers",
                "name": name,
                "x": initial_data["x"][index],
                "y": initial_data["y"][index],
                "text": initial_data["text"][index],
                "marker": {
                    "color": color,
                    "symbol": symbol,
                    "size": 20 if index % 2 == 0 else 10,
                    "line": {"width": 2},
                },
                "xaxis": "x" if index < 2 else "x2",
                "yaxis": "y" if index < 2 else "y2",
                "hovertemplate": "%{text}<br>(%{x:.3f}, %{y:.3f})<extra>%{fullData.name}</extra>",
            }
        )
    return {
        "data": traces,
        "layout": initial_layout
        | {
            "height": 650,
            "paper_bgcolor": "#ffffff",
            "plot_bgcolor": "#eef4f7",
            "font": {"family": "Arial, sans-serif", "color": "#203852"},
            "margin": {"l": 65, "r": 35, "t": 105, "b": 100},
            "legend": {"orientation": "h", "y": -0.2},
            "xaxis": {"domain": [0, 0.43], "range": [0, 800], "title": {"text": "Image u (px)"}},
            "yaxis": {"range": [480, 0], "title": {"text": "Image v (px)"}},
            "updatemenus": [
                {"type": "buttons", "direction": "right", "x": 0, "y": 1.15, "buttons": buttons}
            ],
        },
    }


def build_explorer() -> str:
    """Return one deterministic offline document with the bundled library notice."""
    from jinja2 import Environment, PackageLoader
    from plotly.offline import get_plotlyjs

    environment = Environment(loader=PackageLoader("bevcalib.report", "templates"), autoescape=True)
    return environment.get_template("explorer.html.j2").render(
        figure=explorer_figure(), plotly_js=get_plotlyjs()
    )
