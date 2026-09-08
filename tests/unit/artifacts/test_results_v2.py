"""V2 keeps actual operator signs, missing observations and timestamp provenance."""

from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

from bevcalib.operators.lidar_edges import trimmed_distance_transform_score


def row_document() -> dict:
    edges = np.zeros((4, 6), dtype=bool)
    edges[:, 0] = True
    score = trimmed_distance_transform_score(np.array([[2.0, 1.0]]), edges)
    return {
        "schema_version": "bev-calibration-result/v2",
        "scene_token": "scene",
        "sample_token": "sample",
        "fault_axis": "x",
        "fault_level": 0.1,
        "fault": {
            "rotation_rpy_deg": [0, 0, 0],
            "translation_xyz_m": [0.1, 0, 0],
            "requested_time_offset_ms": 0,
        },
        "estimate": {
            "rotation_rpy_deg": [0, 0, 0],
            "translation_xyz_m": [0, 0, 0],
            "requested_time_offset_ms": 0,
        },
        "camera": {"token": "camera", "timestamp_us": 1_020_000, "channel": "CAM_FRONT"},
        "lidar": {"token": "lidar", "timestamp_us": 1_000_000, "channel": "LIDAR_TOP"},
        "timing": {
            "requested_offset_ms": 0,
            "realized_offset_ms": -20,
            "absolute_error_ms": 20,
            "reason": "nominal_pair",
        },
        "pose": {
            "rotation_rpy_error_deg": [-1, 0, 0],
            "translation_xyz_error_m": [-0.1, 0, 0],
            "rotation_geodesic_error_deg": 1,
            "translation_error_m": 0.1,
        },
        "pixel_errors_px": [1, 2],
        "projection_count": 3,
        "edge_alignment_score": score,
        "ground_contacts": [
            {"box_token": "box", "range_m": 15, "error_m": 2, "invalid_reason": None}
        ],
        "valid": True,
        "invalid_reason": None,
    }


@pytest.mark.parametrize("requested_ms", [-200, -100, -50, 0, 50, 100, 200])
@pytest.mark.parametrize("error_us", [-25001, -25000, -1001, 1001, 25000, 25001, 26001])
def test_native_fractional_timing_selection_round_trips_exactly(
    requested_ms: int, error_us: int
) -> None:
    from bevcalib.artifacts.results import CalibrationResultV2
    from bevcalib.geometry.frames import FramedTransform
    from bevcalib.geometry.se3 import SE3
    from bevcalib.nuscenes_adapter.frames import SensorPacket
    from bevcalib.nuscenes_adapter.sweeps import choose_nearest_sweep

    reference = 1_538_984_233_547_259
    target = reference + requested_ms * 1000
    identity = SE3(rotation_wxyz=(1, 0, 0, 0), translation_xyz_m=(0, 0, 0))
    packet = SensorPacket(
        sample_token="sample",
        sample_data_token="lidar",
        timestamp_us=target + error_us,
        calibrated_sensor=FramedTransform(
            target="lidar_ego", source="lidar_sensor", value=identity
        ),
        ego_pose=FramedTransform(target="global", source="lidar_ego", value=identity),
        file_relative_path="samples/LIDAR_TOP/fractional.bin",
    )
    selection = choose_nearest_sweep((packet,), target, requested_offset_ms=requested_ms)
    doc = row_document()
    doc.update(fault_axis="time", fault_level=requested_ms)
    doc["fault"].update(translation_xyz_m=[0, 0, 0], requested_time_offset_ms=requested_ms)
    doc["camera"]["timestamp_us"] = reference
    doc["lidar"]["timestamp_us"] = packet.timestamp_us
    doc["timing"] = {
        "requested_offset_ms": requested_ms,
        "realized_offset_ms": selection.realized_offset_ms,
        "absolute_error_ms": selection.absolute_error_ms,
        "reason": selection.reason,
    }
    if not selection.valid:
        doc.update(
            estimate=None,
            pose=None,
            pixel_errors_px=[],
            projection_count=0,
            edge_alignment_score=None,
            ground_contacts=[],
            valid=False,
            invalid_reason=selection.reason,
        )
    row = CalibrationResultV2.model_validate(doc)
    assert row.timing.absolute_error_ms == abs(error_us) / 1000
    assert row.valid is (abs(error_us) <= 25000)


def test_actual_negative_edge_operator_and_signed_pose_round_trip() -> None:
    from bevcalib.artifacts.results import CalibrationResultV2

    row = CalibrationResultV2.model_validate(row_document())
    loaded = CalibrationResultV2.model_validate_json(row.model_dump_json())
    assert loaded.edge_alignment_score == -2
    assert loaded.pose.translation_xyz_error_m == (-0.1, 0, 0)
    assert loaded.pose.rotation_rpy_error_deg == (-1, 0, 0)
    assert loaded.projection_count == 3
    assert len(loaded.pixel_errors_px) == 2


def test_missing_timing_has_null_measurements_and_explanation() -> None:
    from bevcalib.artifacts.results import CalibrationResultV2

    doc = row_document()
    doc.update(
        fault_axis="time",
        fault_level=100,
        fault={
            "rotation_rpy_deg": [0, 0, 0],
            "translation_xyz_m": [0, 0, 0],
            "requested_time_offset_ms": 100,
        },
        lidar=None,
        timing={
            "requested_offset_ms": 100,
            "realized_offset_ms": None,
            "absolute_error_ms": None,
            "reason": "no_available_lidar",
        },
        estimate=None,
        pose=None,
        pixel_errors_px=[],
        projection_count=0,
        edge_alignment_score=None,
        ground_contacts=[],
        valid=False,
        invalid_reason="no_available_lidar",
    )
    row = CalibrationResultV2.model_validate(doc)
    assert row.lidar is None
    assert row.edge_alignment_score is None
    assert not row.valid


@pytest.mark.parametrize(
    "mutation",
    [
        {"edge_alignment_score": 2},
        {"edge_alignment_score": float("nan")},
        {"edge_alignment_score": None},
        {"pose": None},
        {"estimate": None},
        {"pixel_errors_px": []},
        {"pixel_errors_px": [-1]},
        {"projection_count": 1},
        {"ground_contacts": []},
        {"invalid_reason": "missing"},
        {"valid": False},
        {"camera": {"token": "camera", "timestamp_us": 1_020_000, "channel": "LIDAR_TOP"}},
        {"lidar": {"token": "camera", "timestamp_us": 1_000_000, "channel": "LIDAR_TOP"}},
        {"lidar": {"token": "lidar", "timestamp_us": 0, "channel": "LIDAR_TOP"}},
        {
            "timing": {
                "requested_offset_ms": 1,
                "realized_offset_ms": -20,
                "absolute_error_ms": 20,
                "reason": "nominal_pair",
            }
        },
    ],
)
def test_schema_valid_looking_incoherence_is_refused(mutation: dict) -> None:
    from bevcalib.artifacts.results import CalibrationResultV2

    with pytest.raises(ValueError):
        CalibrationResultV2.model_validate(row_document() | mutation)


def test_outside_tolerance_retains_selection_but_cannot_be_valid() -> None:
    from bevcalib.artifacts.results import CalibrationResultV2

    doc = row_document()
    doc["fault_axis"] = "time"
    doc["fault_level"] = 100
    doc["fault"] = {
        "rotation_rpy_deg": [0, 0, 0],
        "translation_xyz_m": [0, 0, 0],
        "requested_time_offset_ms": 100,
    }
    doc["timing"] = {
        "requested_offset_ms": 100,
        "realized_offset_ms": -20,
        "absolute_error_ms": 120,
        "reason": "outside_tolerance",
    }
    with pytest.raises(ValueError, match="timing"):
        CalibrationResultV2.model_validate(doc)
    doc.update(
        estimate=None,
        pose=None,
        pixel_errors_px=[],
        projection_count=0,
        edge_alignment_score=None,
        ground_contacts=[],
    )
    with pytest.raises(ValueError, match="invalid timing"):
        CalibrationResultV2.model_validate(doc)
    doc.update(valid=False, invalid_reason="outside_tolerance")
    assert CalibrationResultV2.model_validate(doc).lidar.token == "lidar"


@pytest.mark.parametrize(
    "entry",
    [
        {"box_token": "box", "range_m": 90, "error_m": None, "invalid_reason": "out_of_range"},
        {"box_token": "box", "range_m": 80, "error_m": 0, "invalid_reason": None},
    ],
)
def test_each_ground_contact_preserves_range_and_invalidity(entry: dict) -> None:
    from bevcalib.artifacts.results import GroundContactResult

    result = GroundContactResult.model_validate(entry)
    assert result.model_dump() == entry
    bad = deepcopy(entry)
    bad["invalid_reason"] = None if entry["invalid_reason"] else "invalid"
    with pytest.raises(ValueError):
        GroundContactResult.model_validate(bad)


@pytest.mark.parametrize("case", ["missing", "nominal", "tolerance", "boxes"])
def test_untruthful_selection_reason_or_duplicate_object_is_refused(case: str) -> None:
    from bevcalib.artifacts.results import CalibrationResultV2

    doc = row_document()
    if case == "missing":
        doc["lidar"] = None
    elif case == "nominal":
        doc["timing"]["reason"] = "valid"
    elif case == "tolerance":
        doc["fault_axis"] = "time"
        doc["timing"]["reason"] = "outside_tolerance"
    else:
        doc["ground_contacts"] *= 2
    with pytest.raises(ValueError):
        CalibrationResultV2.model_validate(doc)


@pytest.mark.parametrize("reason", ["no_available_lidar", "outside_tolerance"])
@pytest.mark.parametrize(
    "field",
    [
        "estimate",
        "pose",
        "pixel_errors_px",
        "projection_count",
        "edge_alignment_score",
        "ground_contacts",
    ],
)
def test_unavailable_timing_rejects_each_populated_measurement(reason: str, field: str) -> None:
    from bevcalib.artifacts.results import CalibrationResultV2

    original = row_document()
    doc = deepcopy(original)
    doc.update(
        fault_axis="time",
        fault_level=100,
        fault={
            "rotation_rpy_deg": [0, 0, 0],
            "translation_xyz_m": [0, 0, 0],
            "requested_time_offset_ms": 100,
        },
        estimate=None,
        pose=None,
        pixel_errors_px=[],
        projection_count=0,
        edge_alignment_score=None,
        ground_contacts=[],
        valid=False,
        invalid_reason=reason,
    )
    doc["timing"] = {
        "requested_offset_ms": 100,
        "realized_offset_ms": -20 if reason == "outside_tolerance" else None,
        "absolute_error_ms": 120 if reason == "outside_tolerance" else None,
        "reason": reason,
    }
    if reason == "no_available_lidar":
        doc["lidar"] = None
    doc[field] = original[field]
    if field == "pixel_errors_px":
        doc["projection_count"] = len(doc[field])
    with pytest.raises(ValueError, match="unavailable timing"):
        CalibrationResultV2.model_validate(doc)
