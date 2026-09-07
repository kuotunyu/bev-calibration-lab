"""Contracts for the `verifying-nuscenes-transforms` skill and trace validator."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import cast

import pytest
import yaml
from pydantic import ValidationError

ROOT = Path(__file__).resolve().parents[3]
SKILL_DIR = ROOT / ".agents" / "skills" / "verifying-nuscenes-transforms"
SKILL_FILE = SKILL_DIR / "SKILL.md"
VALIDATOR = SKILL_DIR / "scripts" / "validate_transform_chain.py"


def clean_trace() -> dict[str, object]:
    """A complete, public-safe record of the four required parity cases."""

    return {
        "schema_version": "bev-transform-verification/v1",
        "native_lookup": {
            "sample_pair_source": "scene_records",
            "camera_channel": "CAM_FRONT",
            "lidar_channel": "LIDAR_TOP",
        },
        "timestamps_us": {
            "lidar_timestamp_us": 1_000_000,
            "camera_timestamp_us": 1_050_000,
        },
        "point_array_shape": ["N", 3],
        "chain": [
            {
                "name": "T_lidar_ego_lidar_sensor",
                "target": "lidar_ego",
                "source": "lidar_sensor",
                "timestamp_field": None,
            },
            {
                "name": "T_global_lidar_ego",
                "target": "global",
                "source": "lidar_ego",
                "timestamp_field": "lidar_timestamp_us",
            },
            {
                "name": "T_camera_ego_global",
                "target": "camera_ego",
                "source": "global",
                "timestamp_field": "camera_timestamp_us",
            },
            {
                "name": "T_camera_sensor_camera_ego",
                "target": "camera_sensor",
                "source": "camera_ego",
                "timestamp_field": None,
            },
        ],
        "behind_camera_test": {
            "camera_point_xyz": [0.0, 0.0, -2.0],
            "in_front": False,
            "in_image": True,
            "valid": False,
        },
        "parity": {
            "absolute_tolerance": 1e-6,
            "relative_tolerance": 0.0,
            "cases": [
                {
                    "case_id": "lidar_projection:sample_0",
                    "comparison_count": 100,
                    "max_abs_error": 2e-10,
                },
                {
                    "case_id": "lidar_projection:sample_1",
                    "comparison_count": 120,
                    "max_abs_error": 3e-10,
                },
                {
                    "case_id": "box_centers:sample_0",
                    "comparison_count": 4,
                    "max_abs_error": 4e-10,
                },
                {
                    "case_id": "box_centers:sample_1",
                    "comparison_count": 3,
                    "max_abs_error": 5e-10,
                },
            ],
        },
    }


def parse(document: dict[str, object]):  # type: ignore[no-untyped-def]
    from bevcalib.verification.transform_trace import TransformVerificationTrace

    return TransformVerificationTrace.model_validate(document)


def run_validator(tmp_path: Path, document: dict[str, object]) -> subprocess.CompletedProcess[str]:
    trace = tmp_path / "trace.json"
    trace.write_text(json.dumps(document), encoding="utf-8")
    return subprocess.run(
        [sys.executable, str(VALIDATOR), "--trace", str(trace)],
        capture_output=True,
        text=True,
        check=False,
    )


def test_clean_trace_passes_the_model_and_command(tmp_path: Path) -> None:
    parsed = parse(clean_trace())
    result = run_validator(tmp_path, clean_trace())

    assert parsed.schema_version == "bev-transform-verification/v1"
    assert result.returncode == 0, result.stderr
    assert "4 parity cases" in result.stdout


def test_a_reversed_transform_edge_is_refused(tmp_path: Path) -> None:
    trace = clean_trace()
    edge = trace["chain"][0]  # type: ignore[index]
    edge["target"], edge["source"] = edge["source"], edge["target"]  # type: ignore[index]

    with pytest.raises(ValidationError, match="directed four-link chain"):
        parse(trace)
    result = run_validator(tmp_path, trace)
    assert result.returncode == 1
    assert "directed four-link chain" in result.stderr


def test_a_missing_camera_timestamp_is_refused(tmp_path: Path) -> None:
    trace = clean_trace()
    cast(dict[str, object], trace["timestamps_us"]).pop("camera_timestamp_us")

    with pytest.raises(ValidationError, match="camera_timestamp_us"):
        parse(trace)
    result = run_validator(tmp_path, trace)
    assert result.returncode == 1
    assert "camera_timestamp_us" in result.stderr


def test_a_mismatched_frame_name_is_refused(tmp_path: Path) -> None:
    trace = clean_trace()
    trace["chain"][0]["source"] = "lidar"  # type: ignore[index]

    with pytest.raises(ValidationError, match=r"chain\.0\.source"):
        parse(trace)
    result = run_validator(tmp_path, trace)
    assert result.returncode == 1
    assert "chain.0.source" in result.stderr


def test_a_parity_gap_just_over_one_e_minus_six_is_refused(tmp_path: Path) -> None:
    trace = clean_trace()
    trace["parity"]["cases"][0]["max_abs_error"] = 1.000001e-6  # type: ignore[index]

    with pytest.raises(ValidationError, match="exceeds absolute tolerance"):
        parse(trace)
    result = run_validator(tmp_path, trace)
    assert result.returncode == 1
    assert "exceeds absolute tolerance" in result.stderr


def test_a_trace_without_a_behind_camera_test_is_refused(tmp_path: Path) -> None:
    trace = clean_trace()
    trace.pop("behind_camera_test")

    with pytest.raises(ValidationError, match="behind_camera_test"):
        parse(trace)
    result = run_validator(tmp_path, trace)
    assert result.returncode == 1
    assert "behind_camera_test" in result.stderr


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("shape", "point_array_shape"),
        ("absolute_tolerance", "absolute tolerance"),
        ("relative_tolerance", "relative tolerance"),
        ("missing_case", "exactly the four pinned parity cases"),
        ("empty_case", "comparison_count"),
        ("native_lookup", "scene_records"),
        ("behind_camera_visible", "behind-camera"),
    ],
)
def test_other_trace_contract_gaps_are_refused(tmp_path: Path, mutation: str, message: str) -> None:
    trace = clean_trace()
    if mutation == "shape":
        trace["point_array_shape"] = [3, "N"]
    elif mutation == "absolute_tolerance":
        trace["parity"]["absolute_tolerance"] = 1e-5  # type: ignore[index]
    elif mutation == "relative_tolerance":
        trace["parity"]["relative_tolerance"] = 1e-7  # type: ignore[index]
    elif mutation == "missing_case":
        trace["parity"]["cases"].pop()  # type: ignore[index]
    elif mutation == "empty_case":
        trace["parity"]["cases"][0]["comparison_count"] = 0  # type: ignore[index]
    elif mutation == "native_lookup":
        trace["native_lookup"]["sample_pair_source"] = "sample.data"  # type: ignore[index]
    else:
        trace["behind_camera_test"]["valid"] = True  # type: ignore[index]

    with pytest.raises(ValidationError, match=message):
        parse(trace)
    result = run_validator(tmp_path, trace)
    assert result.returncode == 1
    assert message in result.stderr


def test_the_skill_has_valid_discoverable_frontmatter() -> None:
    text = SKILL_FILE.read_text(encoding="utf-8")
    _, frontmatter, _ = text.split("---", 2)
    metadata = yaml.safe_load(frontmatter)

    assert metadata["name"] == "verifying-nuscenes-transforms"
    assert metadata["description"].startswith("Use when ")
    assert "nuScenes" in metadata["description"]


def test_the_skill_names_its_validator_and_native_interface() -> None:
    text = SKILL_FILE.read_text(encoding="utf-8")

    assert "validate_transform_chain.py" in text
    assert "scene_records" in text
    assert VALIDATOR.is_file()


def test_inputs_that_cannot_be_checked_exit_two(tmp_path: Path) -> None:
    missing = subprocess.run(
        [sys.executable, str(VALIDATOR), "--trace", str(tmp_path / "missing.json")],
        capture_output=True,
        text=True,
        check=False,
    )
    malformed = tmp_path / "malformed.json"
    malformed.write_text("{", encoding="utf-8")
    invalid = subprocess.run(
        [sys.executable, str(VALIDATOR), "--trace", str(malformed)],
        capture_output=True,
        text=True,
        check=False,
    )

    assert missing.returncode == 2 and "cannot read trace" in missing.stderr
    assert invalid.returncode == 2 and "valid JSON" in invalid.stderr
