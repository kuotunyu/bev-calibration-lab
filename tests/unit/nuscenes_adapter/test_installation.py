"""Synthetic native nuScenes tables: provenance and actual file availability."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image


@pytest.fixture
def installation_root(tmp_path: Path) -> Path:
    root = tmp_path / "data"
    directory = root / "v1.0-mini"
    directory.mkdir(parents=True)
    pose = {"rotation": [1, 0, 0, 0], "translation": [0, 0, 0]}
    tables: dict[str, list[dict[str, Any]]] = {
        "scene": [{"token": "scene", "name": "scene-0061", "log_token": "log"}],
        "log": [{"token": "log", "location": "boston-seaport"}],
        "sample": [{"token": "sample", "timestamp": 1_000_000, "scene_token": "scene"}],
        "sensor": [
            {"token": "camera-sensor", "channel": "CAM_FRONT", "modality": "camera"},
            {"token": "lidar-sensor", "channel": "LIDAR_TOP", "modality": "lidar"},
        ],
        "calibrated_sensor": [
            pose
            | {
                "token": "camera-cal",
                "sensor_token": "camera-sensor",
                "camera_intrinsic": [[4, 0, 3], [0, 4, 2], [0, 0, 1]],
            },
            pose | {"token": "lidar-cal", "sensor_token": "lidar-sensor", "camera_intrinsic": []},
        ],
        "ego_pose": [
            pose | {"token": "camera-pose", "timestamp": 1_020_000, "translation": [2, 0, 0]},
            pose | {"token": "lidar-pose", "timestamp": 1_000_000},
            pose | {"token": "selected-pose", "timestamp": 1_115_000, "translation": [11, 0, 0]},
        ],
        "sample_data": [
            {
                "token": token,
                "sample_token": "sample",
                "timestamp": timestamp,
                "calibrated_sensor_token": f"{sensor}-cal",
                "ego_pose_token": f"{which}-pose",
                "filename": f"samples/{channel}/{token}.{extension}",
                "is_key_frame": keyframe,
                "width": 6 if sensor == "camera" else 0,
                "height": 4 if sensor == "camera" else 0,
            }
            for token, sensor, which, timestamp, channel, extension, keyframe in (
                ("camera", "camera", "camera", 1_020_000, "CAM_FRONT", "png", True),
                ("lidar", "lidar", "lidar", 1_000_000, "LIDAR_TOP", "bin", True),
                ("selected", "lidar", "selected", 1_115_000, "LIDAR_TOP", "bin", False),
            )
        ],
        "sample_annotation": [],
    }
    for name, rows in tables.items():
        (directory / f"{name}.json").write_text(json.dumps(rows), encoding="utf-8")
    for row in tables["sample_data"]:
        path = root / row["filename"]
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix == ".png":
            Image.new("RGB", (6, 4), "red").save(path)
        else:
            np.array([[0, 0, 10, 0.5, 1], [0.1, 0.1, 20, 0.2, 1]], dtype=np.float32).tofile(path)
    return root


def rewrite(root: Path, table: str, change) -> None:  # type: ignore[no-untyped-def]
    path = root / "v1.0-mini" / f"{table}.json"
    rows = json.loads(path.read_text(encoding="utf-8"))
    change(rows)
    path.write_text(json.dumps(rows), encoding="utf-8")


def test_resolves_native_tables_and_preserves_fixed_camera_timing(installation_root: Path) -> None:
    from bevcalib.nuscenes_adapter.installation import resolve_installation

    installation = resolve_installation(installation_root, "v1.0-mini")
    camera = installation.packet("camera", "CAM_FRONT")
    selection = installation.select_timing("camera", 100)
    assert selection.selected_sample_data_token == "selected"
    assert selection.realized_offset_ms == 95
    assert selection.absolute_error_ms == 5
    assert camera.timestamp_us == 1_020_000
    assert camera.ego_pose.value.translation_xyz_m == (2, 0, 0)
    assert installation.packet("selected", "LIDAR_TOP").ego_pose.value.translation_xyz_m == (
        11,
        0,
        0,
    )
    records = installation.scene_records()
    assert len(records) == 1
    assert records[0].camera_sample_data_tokens == ("camera",)
    assert records[0].lidar_sample_data_tokens == ("lidar",)
    assert records[0].official_split == "train"


def test_metadata_without_payload_is_not_timing_availability(installation_root: Path) -> None:
    from bevcalib.nuscenes_adapter.installation import resolve_installation

    (installation_root / "samples/LIDAR_TOP/selected.bin").unlink()
    installation = resolve_installation(installation_root, "v1.0-mini")
    result = installation.select_timing("camera", 100)
    assert not result.valid
    assert result.selected_sample_data_token == "lidar"
    assert result.absolute_error_ms == 120
    (installation_root / "samples/LIDAR_TOP/lidar.bin").unlink()
    result = installation.select_timing("camera", -50)
    assert result.reason == "no_available_lidar"
    assert result.realized_offset_ms is None


@pytest.mark.parametrize("version", ["v1.0-test", "unknown"])
def test_unsupported_version_is_refused(installation_root: Path, version: str) -> None:
    from bevcalib.nuscenes_adapter.installation import resolve_installation

    with pytest.raises(ValueError, match="version"):
        resolve_installation(installation_root, version)


def test_missing_root_and_table_directory_are_refused(tmp_path: Path) -> None:
    from bevcalib.nuscenes_adapter.installation import resolve_installation

    with pytest.raises(FileNotFoundError):
        resolve_installation(tmp_path / "missing", "v1.0-mini")
    with pytest.raises(FileNotFoundError):
        resolve_installation(tmp_path, "v1.0-mini")


@pytest.mark.parametrize(
    "field,value", [("calibrated_sensor_token", "camera-pose"), ("ego_pose_token", "camera-cal")]
)
def test_reversed_pose_lookup_is_refused(installation_root: Path, field: str, value: str) -> None:
    from bevcalib.nuscenes_adapter.installation import resolve_installation

    rewrite(installation_root, "sample_data", lambda rows: rows[0].update({field: value}))
    with pytest.raises(ValueError, match="unresolved"):
        resolve_installation(installation_root, "v1.0-mini")


def test_wrong_sensor_and_other_scene_cannot_supply_timing(installation_root: Path) -> None:
    from bevcalib.nuscenes_adapter.installation import resolve_installation

    rewrite(
        installation_root,
        "scene",
        lambda rows: rows.append(
            {"token": "other-scene", "name": "scene-0103", "log_token": "log"}
        ),
    )
    rewrite(
        installation_root,
        "sample",
        lambda rows: rows.append(
            {"token": "other-sample", "timestamp": 1_115_000, "scene_token": "other-scene"}
        ),
    )
    rewrite(
        installation_root, "sample_data", lambda rows: rows[2].update(sample_token="other-sample")
    )
    installation = resolve_installation(installation_root, "v1.0-mini")
    assert installation.select_timing("camera", 100).selected_sample_data_token == "lidar"
    with pytest.raises(ValueError, match="channel"):
        installation.packet("lidar", "CAM_FRONT")


def test_escaped_payload_and_timestamp_mismatch_fail_closed(installation_root: Path) -> None:
    from bevcalib.nuscenes_adapter.installation import resolve_installation

    rewrite(
        installation_root, "sample_data", lambda rows: rows[0].update(filename="../escaped.png")
    )
    with pytest.raises(ValueError, match="escapes"):
        resolve_installation(installation_root, "v1.0-mini")
    rewrite(
        installation_root,
        "sample_data",
        lambda rows: rows[0].update(filename="safe.png", timestamp=12),
    )
    with pytest.raises(ValueError, match="timestamp"):
        resolve_installation(installation_root, "v1.0-mini")


def test_preflight_reports_each_offset_available_and_invalid_counts(
    installation_root: Path,
) -> None:
    from bevcalib.nuscenes_adapter.installation import resolve_installation

    report = resolve_installation(installation_root, "v1.0-mini").preflight()
    assert report["dataset_version"] == "v1.0-mini"
    assert report["timing"]["100"] == {
        "total": 1,
        "valid": 1,
        "invalid": 0,
        "valid_fraction": 1.0,
        "reasons": {},
    }
    assert report["timing"]["200"]["reasons"] == {"outside_tolerance": 1}


@pytest.mark.parametrize(
    "table,change,message",
    [
        ("sample", lambda rows: rows.append(rows[0]), "duplicate"),
        ("scene", lambda rows: rows[0].update(name="not-an-official-scene"), "official"),
        ("sample_data", lambda rows: rows[2].update(is_key_frame=True), "duplicate keyframe"),
        ("sample_data", lambda rows: rows[0].update(is_key_frame=False), "paired keyframe"),
    ],
)
def test_ambiguous_or_incomplete_tables_are_refused(
    installation_root: Path, table, change, message
) -> None:  # type: ignore[no-untyped-def]
    from bevcalib.nuscenes_adapter.installation import resolve_installation

    rewrite(installation_root, table, change)
    with pytest.raises(ValueError, match=message):
        resolve_installation(installation_root, "v1.0-mini").scene_records()


def test_empty_installation_reports_no_measurable_timing(installation_root: Path) -> None:
    from bevcalib.nuscenes_adapter.installation import TABLES, resolve_installation

    for table in TABLES:
        rewrite(installation_root, table, lambda rows: rows.clear())
    assert (
        resolve_installation(installation_root, "v1.0-mini").preflight()["timing"]["0"][
            "valid_fraction"
        ]
        is None
    )


def test_trainval_split_and_unused_sensor_are_handled(installation_root: Path) -> None:
    from bevcalib.nuscenes_adapter.installation import resolve_installation

    rewrite(
        installation_root,
        "sensor",
        lambda rows: rows.append(
            {"token": "rear-sensor", "channel": "CAM_BACK", "modality": "camera"}
        ),
    )
    rewrite(
        installation_root,
        "calibrated_sensor",
        lambda rows: rows.append(rows[0] | {"token": "rear-cal", "sensor_token": "rear-sensor"}),
    )
    rewrite(
        installation_root,
        "sample_data",
        lambda rows: rows.append(
            rows[0] | {"token": "rear", "calibrated_sensor_token": "rear-cal"}
        ),
    )
    (installation_root / "v1.0-mini").rename(installation_root / "v1.0-trainval")
    installation = resolve_installation(installation_root, "v1.0-trainval")
    assert installation.scene_records()[0].official_split == "train"
    with pytest.raises(ValueError, match="channel"):
        installation.packet("rear", "CAM_BACK")


def test_preprocessing_uses_config_geometry_and_assumed_depth(installation_root: Path) -> None:
    from bevcalib.artifacts.results import CalibrationFaultModel
    from bevcalib.nuscenes_adapter.installation import resolve_installation
    from bevcalib.preprocessing import load_observation, prepare_input
    from bevcalib.training.engine import load_corrector_config

    installation = resolve_installation(installation_root, "v1.0-mini")
    observation = load_observation(installation, installation.scene_records()[0], 0)
    original = observation.rgb.copy(), observation.points.copy()
    config = load_corrector_config(Path("configs/correctors/convnextv2_tiny_v1.yaml"))
    zero = CalibrationFaultModel(
        rotation_rpy_deg=(0, 0, 0), translation_xyz_m=(0, 0, 0), requested_time_offset_ms=0
    )
    tensor = prepare_input(observation, zero, config.input_height, config.input_width)
    assert tensor.shape == (5, 448, 800)
    assert tensor.dtype == np.float32
    np.testing.assert_allclose(
        tensor[:3, 0, 0], [(1 - 0.485) / 0.229, -0.456 / 0.224, -0.406 / 0.225], rtol=1e-6
    )
    assert tensor[4].sum() == 2
    changed = prepare_input(
        observation, zero.model_copy(update={"translation_xyz_m": (0.2, 0, 0)}), 448, 800
    )
    assert not np.array_equal(tensor[4], changed[4])
    np.testing.assert_array_equal(original[0], observation.rgb)
    np.testing.assert_array_equal(original[1], observation.points)


@pytest.mark.parametrize(
    "change,message",
    [
        (lambda rows: rows[0].update(timestamp=1_000_001), "manifest"),
        (lambda rows: rows[0].update(width=9), "dimensions"),
    ],
)
def test_observation_refuses_manifest_and_image_drift(
    installation_root: Path, change, message
) -> None:  # type: ignore[no-untyped-def]
    from bevcalib.nuscenes_adapter.installation import resolve_installation
    from bevcalib.preprocessing import load_observation

    installation = resolve_installation(installation_root, "v1.0-mini")
    scene = installation.scene_records()[0]
    change(list(installation.tables["sample_data"].values()))
    with pytest.raises(ValueError, match=message):
        load_observation(installation, scene, 0)


def test_selected_lidar_and_global_boxes_are_read_without_camera_motion(
    installation_root: Path,
) -> None:
    from bevcalib.nuscenes_adapter.installation import resolve_installation
    from bevcalib.preprocessing import load_observation

    rewrite(
        installation_root,
        "sample_annotation",
        lambda rows: rows.append(
            {
                "token": "box",
                "sample_token": "sample",
                "translation": [7, 8, 9],
                "size": [2, 4, 2],
                "rotation": [1, 0, 0, 0],
            }
        ),
    )
    installation = resolve_installation(installation_root, "v1.0-mini")
    observation = load_observation(
        installation, installation.scene_records()[0], 0, lidar_token="selected"
    )
    assert observation.lidar.sample_data_token == "selected"
    assert observation.camera.sample_data_token == "camera"
    assert observation.boxes[0]["translation"] == [7, 8, 9]


def test_observation_rejects_selected_lidar_from_a_different_scene(installation_root: Path) -> None:
    from bevcalib.nuscenes_adapter.installation import resolve_installation
    from bevcalib.preprocessing import load_observation

    installation = resolve_installation(installation_root, "v1.0-mini")
    scene = installation.scene_records()[0]
    installation.tables["sample"]["other-sample"] = {"scene_token": "other-scene"}
    installation.tables["scene"]["other-scene"] = {"token": "other-scene", "log_token": "log"}
    installation.tables["sample_data"]["selected"]["sample_token"] = "other-sample"
    with pytest.raises(ValueError, match="selected LiDAR"):
        load_observation(installation, scene, 0, lidar_token="selected")


def test_preflight_includes_missing_payload_inventory(installation_root: Path) -> None:
    from bevcalib.nuscenes_adapter.installation import resolve_installation

    (installation_root / "samples/LIDAR_TOP/selected.bin").unlink()
    assert resolve_installation(installation_root, "v1.0-mini").preflight()["missing_payloads"] == [
        "selected"
    ]
