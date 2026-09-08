"""Refusals, missing measurements and actual learned service round trips."""

import dataclasses
import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from PIL import Image
from tests.unit.nuscenes_adapter.test_installation import installation_root as installation_root
from tests.unit.test_evaluation import evaluation_workspace as evaluation_workspace
from tests.unit.training.test_engine import cohort_document
from tests.unit.training.test_torch_backend import tiny_model

PROTOCOL = Path("configs/protocols/nuscenes_calibration_v1.yaml")


@pytest.mark.parametrize(
    "case,message",
    [
        ("method", "unknown"),
        ("checkpoint", "only valid"),
        ("role", "verified V2"),
        ("metadata", "frozen cohort"),
        ("payload", "No such file"),
    ],
)
def test_refusals_precede_output(
    evaluation_workspace, tmp_path: Path, case: str, message: str
) -> None:
    from bevcalib.cohort.manifest import manifest_hash
    from bevcalib.evaluation import evaluate_calibration

    root, manifest = evaluation_workspace
    method, checkpoint = "identity", None
    if case == "method":
        method = "unknown"
    elif case == "checkpoint":
        checkpoint = tmp_path / "absent.pt"
    elif case == "role":
        body = cohort_document("development", "development", 1)
        manifest.write_text(json.dumps(body), encoding="utf-8")
    elif case == "metadata":
        body = json.loads(manifest.read_text(encoding="utf-8"))
        body["scenes"][0]["camera_timestamps"][0] += 1
        body["manifest_sha256"] = manifest_hash(body)
        manifest.write_text(json.dumps(body), encoding="utf-8")
    else:
        (root / "samples/CAM_FRONT/camera.png").unlink()
    with pytest.raises((ValueError, FileNotFoundError), match=message):
        evaluate_calibration(
            PROTOCOL,
            manifest,
            method,
            tmp_path / "out",
            dataroot=root,
            checkpoint=checkpoint,
            synthetic_fixture=True,
        )
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize(
    "case,reason",
    [
        ("constant", "no_image_edges"),
        ("behind", "no_paired_projection"),
        ("flat", "no_projected_lidar_edges"),
        ("no_boxes", "no_valid_ground_contact"),
        ("far_box", "no_valid_ground_contact"),
    ],
)
def test_actual_operator_unavailability_remains_null(
    evaluation_workspace, tmp_path: Path, case: str, reason: str
) -> None:
    from bevcalib.artifacts.result_documents import load_result_run
    from bevcalib.cohort.manifest import load_manifest
    from bevcalib.evaluation import evaluate_calibration

    root, manifest = evaluation_workspace
    if case == "constant":
        Image.new("RGB", (6, 4), "red").save(root / "samples/CAM_FRONT/camera.png")
    elif case in ("behind", "flat"):
        points = (
            [[-10.0, 0.0, 0.0, 0.5, 1.0], [-20.0, 0.1, 0.0, 0.2, 1.0]]
            if case == "behind"
            else [[10.0, 0.0, 0.0, 0.5, 1.0], [10.0, 0.1, 0.0, 0.2, 1.0]]
        )
        np.asarray(points, dtype=np.float32).tofile(root / "samples/LIDAR_TOP/lidar.bin")
    else:
        path = root / "v1.0-trainval/sample_annotation.json"
        boxes = json.loads(path.read_text(encoding="utf-8"))
        if case == "no_boxes":
            boxes = []
        else:
            boxes[0]["translation"] = [100.0, 0.0, 1.0]
        path.write_text(json.dumps(boxes), encoding="utf-8")
    result = evaluate_calibration(
        PROTOCOL, manifest, "classical", tmp_path / "out", dataroot=root, synthetic_fixture=True
    )
    rows = load_result_run(result.directory, result.identity, load_manifest(manifest))
    zero = next(row for row in rows if row.fault_axis == "yaw" and row.fault_level == 0)
    assert not zero.valid and reason in zero.invalid_reason
    if case in ("constant", "behind", "flat"):
        assert zero.edge_alignment_score is None
    if case == "behind":
        assert zero.pixel_errors_px == ()
    if case == "far_box":
        assert zero.ground_contacts[0].error_m is None


def test_empty_timing_snapshot_has_no_fabricated_measurements(evaluation_workspace) -> None:
    from bevcalib.evaluation_measurements import invalid_timing_result
    from bevcalib.nuscenes_adapter.installation import resolve_installation
    from bevcalib.preprocessing import load_observation

    root, _ = evaluation_workspace
    snapshot = resolve_installation(root, "v1.0-trainval")
    scene = snapshot.scene_records()[0]
    observation = load_observation(snapshot, scene, 0)
    for token in ("lidar", "selected"):
        (root / f"samples/LIDAR_TOP/{token}.bin").unlink()
    empty = resolve_installation(root, "v1.0-trainval").select_timing("camera", 100)
    row = invalid_timing_result(observation, scene.scene_token, empty, None)
    assert row.lidar is None and row.pose is None and row.edge_alignment_score is None
    assert row.invalid_reason == "no_available_lidar"


def test_existing_run_cannot_be_overwritten(evaluation_workspace, tmp_path: Path) -> None:
    from bevcalib.evaluation import evaluate_calibration

    root, manifest = evaluation_workspace
    evaluate_calibration(
        PROTOCOL, manifest, "identity", tmp_path / "out", dataroot=root, synthetic_fixture=True
    )
    with pytest.raises(FileExistsError, match="overwrite"):
        evaluate_calibration(
            PROTOCOL, manifest, "identity", tmp_path / "out", dataroot=root, synthetic_fixture=True
        )


def test_actual_trained_checkpoint_runs_learned_service_without_evaluation_training(
    evaluation_workspace, tmp_path: Path
) -> None:
    from bevcalib.artifacts.result_documents import load_result_run
    from bevcalib.cohort.manifest import load_manifest, manifest_hash
    from bevcalib.evaluation import evaluate_calibration
    from bevcalib.nuscenes_adapter.installation import resolve_installation
    from bevcalib.training.engine import train_learned_corrector
    from bevcalib.training.torch_backend import TorchBackend

    root, evaluation = evaluation_workspace
    for table in ("scene", "log", "sample", "sample_data", "sample_annotation"):
        path = root / "v1.0-trainval" / (table + ".json")
        originals = json.loads(path.read_text(encoding="utf-8"))
        rows = list(originals)
        for suffix, name in (("dev", "scene-0061"), ("cal", "scene-0001")):
            for row in originals:
                new = dict(row)
                for field in ("token", "sample_token", "scene_token", "log_token"):
                    if field in new:
                        new[field] += "-" + suffix
                if table == "scene":
                    new["name"] = name
                rows.append(new)
        path.write_text(json.dumps(rows), encoding="utf-8")
    records = resolve_installation(root, "v1.0-trainval").scene_records()
    manifests = []
    for role, suffix in (("development", "dev"), ("calibration", "cal")):
        body = cohort_document(role, role, 1)
        body["scenes"] = [
            dataclasses.asdict(
                next(scene for scene in records if scene.scene_token.endswith(suffix))
            )
        ]
        body["manifest_sha256"] = manifest_hash(body)
        path = tmp_path / (role + ".json")
        path.write_text(json.dumps(body), encoding="utf-8")
        manifests.append(path)
    shutil.copytree(Path("configs"), tmp_path / "configs")
    config = tmp_path / "configs/correctors/convnextv2_tiny_v1.yaml"
    body = yaml.safe_load(config.read_text(encoding="utf-8"))
    body.update(epochs=2, warmup_epochs=1, batch_size=1)
    config.write_text(yaml.safe_dump(body), encoding="utf-8")
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        trained = train_learned_corrector(
            config,
            *manifests,
            tmp_path / "train",
            17,
            backend=TorchBackend(root, synthetic_fixture=True, model_factory=tiny_model),
            synthetic_fixture=True,
        )
        result = evaluate_calibration(
            PROTOCOL,
            evaluation,
            "learned",
            tmp_path / "eval",
            dataroot=root,
            checkpoint=trained.checkpoint_path,
            synthetic_fixture=True,
            model_factory=tiny_model,
        )
        rows = load_result_run(result.directory, result.identity, load_manifest(evaluation))
        assert len(rows) == 60 and all(row.fault_axis != "time" for row in rows)
        assert result.identity.seed == 17
        assert result.identity.producer.hardware["inference_device"] == "cpu"
        assert all(row.estimate is not None for row in rows)
        assert {
            scene["scene_token"]
            for scene in json.loads(
                (trained.run_record_path.parent / "training_provenance.json").read_text(
                    encoding="utf-8"
                )
            )["development"]["scenes"]
        } == {"scene-dev"}
    finally:
        torch.set_num_threads(previous)


def test_camera_drift_between_identity_and_measurement_is_refused(
    evaluation_workspace, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import bevcalib.evaluation as service

    root, manifest = evaluation_workspace
    original = service.image_edge_evidence

    def mutate_after_threshold(rgb, **kwargs):
        evidence = original(rgb, **kwargs)
        if kwargs.get("with_distance_field") is False:
            Image.new("RGB", (6, 4), "blue").save(root / "samples/CAM_FRONT/camera.png")
        return evidence

    monkeypatch.setattr(service, "image_edge_evidence", mutate_after_threshold)
    with pytest.raises(ValueError, match="payload drift"):
        service.evaluate_calibration(
            PROTOCOL, manifest, "identity", tmp_path / "out", dataroot=root, synthetic_fixture=True
        )
    assert not (tmp_path / "out").exists()


def test_service_retains_empty_selector_evidence_without_placeholder_scores(
    evaluation_workspace, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from bevcalib.artifacts.result_documents import fault_for_condition, load_result_run
    from bevcalib.cohort.manifest import load_manifest
    from bevcalib.evaluation import evaluate_calibration
    from bevcalib.nuscenes_adapter.installation import NuScenesInstallation
    from bevcalib.perturbations.timing import select_lidar_for_timing_fault

    root, manifest = evaluation_workspace
    # Isolate the no-candidate response at the selector boundary; ordinary runs above use native files.
    monkeypatch.setattr(
        NuScenesInstallation,
        "select_timing",
        lambda self, camera, offset: select_lidar_for_timing_fault(
            (), self.packet(camera, "CAM_FRONT").timestamp_us, fault_for_condition("time", offset)
        ),
    )
    result = evaluate_calibration(
        PROTOCOL, manifest, "identity", tmp_path / "out", dataroot=root, synthetic_fixture=True
    )
    rows = load_result_run(result.directory, result.identity, load_manifest(manifest))
    timing = [row for row in rows if row.fault_axis == "time"]
    assert len(timing) == 7 and all(
        row.lidar is None
        and row.pose is None
        and row.edge_alignment_score is None
        and row.invalid_reason == "no_available_lidar"
        for row in timing
    )


def test_artifact_root_binds_one_verified_manifest(evaluation_workspace, tmp_path: Path) -> None:
    from bevcalib.cohort.manifest import load_manifest
    from bevcalib.evaluation import evaluate_calibration

    root, manifest = evaluation_workspace
    result = evaluate_calibration(
        PROTOCOL, manifest, "identity", tmp_path / "out", dataroot=root, synthetic_fixture=True
    )
    sidecar = result.directory.parent / "evaluation_manifest.json"
    assert load_manifest(sidecar) == load_manifest(manifest)
    sidecar.write_text(json.dumps(cohort_document("evaluation", "other", 1)), encoding="utf-8")
    with pytest.raises(ValueError, match=r"artifact root.*cohort"):
        evaluate_calibration(
            PROTOCOL, manifest, "classical", tmp_path / "out", dataroot=root, synthetic_fixture=True
        )
    assert not list((tmp_path / "out").glob("classical-*"))


def test_compatible_methods_reuse_the_exact_manifest_sidecar(
    evaluation_workspace, tmp_path: Path
) -> None:
    from bevcalib.evaluation import evaluate_calibration

    root, manifest = evaluation_workspace
    first = evaluate_calibration(
        PROTOCOL, manifest, "identity", tmp_path / "out", dataroot=root, synthetic_fixture=True
    )
    original = (first.directory.parent / "evaluation_manifest.json").read_bytes()
    second = evaluate_calibration(
        PROTOCOL, manifest, "classical", tmp_path / "out", dataroot=root, synthetic_fixture=True
    )
    assert first.directory != second.directory
    assert (second.directory.parent / "evaluation_manifest.json").read_bytes() == original
