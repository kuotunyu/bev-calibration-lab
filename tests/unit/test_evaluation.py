"""Real identity/classical/learned evaluation on native synthetic observations."""

import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from tests.unit.nuscenes_adapter.test_installation import installation_root as installation_root
from tests.unit.nuscenes_adapter.test_installation import rewrite
from tests.unit.training.test_engine import PROVENANCE, cohort_document


@pytest.fixture
def evaluation_workspace(installation_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from bevcalib.cohort.manifest import manifest_hash
    from bevcalib.geometry.quaternions import matrix_to_quaternion
    from bevcalib.nuscenes_adapter.installation import resolve_installation

    rotation = np.array([[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]])
    rewrite(
        installation_root,
        "calibrated_sensor",
        lambda rows: rows[0].update(
            rotation=list(matrix_to_quaternion(rotation.T)), translation=[0.0, 0.0, 1.0]
        ),
    )
    rewrite(installation_root, "scene", lambda rows: rows[0].update(name="scene-0103"))
    rewrite(
        installation_root,
        "sample_annotation",
        lambda rows: rows.append(
            {
                "token": "box",
                "sample_token": "sample",
                "translation": [10.0, 0.0, 1.0],
                "size": [2.0, 4.0, 2.0],
                "rotation": [1.0, 0.0, 0.0, 0.0],
            }
        ),
    )
    rgb = np.zeros((4, 6, 3), dtype=np.uint8)
    rgb[:, 3:] = 255
    Image.fromarray(rgb).save(installation_root / "samples/CAM_FRONT/camera.png")
    for token in ("lidar", "selected"):
        np.array([[10.0, 0.0, 0.0, 0.5, 1.0], [20.0, 0.1, 0.0, 0.2, 1.0]], dtype=np.float32).tofile(
            installation_root / f"samples/LIDAR_TOP/{token}.bin"
        )
    (installation_root / "v1.0-mini").rename(installation_root / "v1.0-trainval")
    scene = resolve_installation(installation_root, "v1.0-trainval").scene_records()[0]
    body = cohort_document("evaluation", "evaluation", 1)
    body["scenes"] = [dataclasses.asdict(scene)]
    body["manifest_sha256"] = manifest_hash(body)
    manifest = tmp_path / "evaluation.json"
    manifest.write_text(json.dumps(body), encoding="utf-8")
    monkeypatch.setenv("BEVCALIB_RUN_PROVENANCE", json.dumps(PROVENANCE))
    return installation_root, manifest


@pytest.mark.parametrize("method", ["identity", "classical"])
def test_real_service_preserves_fixed_observations_all_rows_and_negative_scores(
    evaluation_workspace, tmp_path: Path, method: str
) -> None:
    from bevcalib.artifacts.result_documents import load_result_run
    from bevcalib.cohort.manifest import load_manifest
    from bevcalib.evaluation import evaluate_calibration

    root, manifest = evaluation_workspace
    result = evaluate_calibration(
        Path("configs/protocols/nuscenes_calibration_v1.yaml"),
        manifest,
        method,
        tmp_path / "runs",
        dataroot=root,
        synthetic_fixture=True,
    )
    rows = load_result_run(result.directory, result.identity, load_manifest(manifest))
    assert len(rows) == (67 if method == "identity" else 60)
    assert {row.camera.token for row in rows} == {"camera"}
    assert {row.camera.timestamp_us for row in rows} == {1_020_000}
    assert {row.lidar.token for row in rows if row.fault_axis != "time"} == {"lidar"}
    if method == "identity":
        timing = next(row for row in rows if row.fault_axis == "time" and row.fault_level == 100)
        assert timing.lidar.token == "selected" and timing.timing.absolute_error_ms == 5
        outside = next(row for row in rows if row.fault_axis == "time" and row.fault_level == 200)
        assert not outside.valid and outside.edge_alignment_score is None
        assert outside.invalid_reason == "outside_tolerance"
    else:
        assert all(row.fault_axis != "time" for row in rows)
    baseline = next(row for row in rows if row.fault_axis == "yaw" and row.fault_level == 0)
    assert baseline.valid
    assert baseline.pose.rotation_geodesic_error_deg == 0
    assert baseline.ground_contacts[0].error_m == pytest.approx(0.0, abs=1e-12)
    assert all(
        row.edge_alignment_score <= 0 for row in rows if row.edge_alignment_score is not None
    )
    assert result.identity.measurements.images["camera"].edge_threshold > 0
    assert result.identity.measurements.table_sha256
    assert (result.directory / "run_complete.json").is_file()


def test_default_formal_service_refuses_underfill_before_dataroot(
    evaluation_workspace, tmp_path: Path
) -> None:
    from bevcalib.evaluation import evaluate_calibration

    _, manifest = evaluation_workspace
    with pytest.raises(ValueError, match="exactly 30"):
        evaluate_calibration(
            Path("configs/protocols/nuscenes_calibration_v1.yaml"),
            manifest,
            "identity",
            tmp_path / "absent",
            dataroot=tmp_path / "missing",
        )
    assert not (tmp_path / "absent").exists()


def test_learned_requires_checkpoint_before_any_dataset_access(tmp_path: Path) -> None:
    from bevcalib.evaluation import evaluate_calibration

    with pytest.raises(ValueError, match="learned requires checkpoint"):
        evaluate_calibration(
            Path("configs/protocols/nuscenes_calibration_v1.yaml"),
            tmp_path / "missing.json",
            "learned",
            tmp_path / "absent",
            dataroot=tmp_path / "missing",
        )
    assert not (tmp_path / "absent").exists()
