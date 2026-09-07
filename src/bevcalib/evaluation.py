"""Measured fixed-observation evaluation over a verified cohort and complete fault inventory."""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bevcalib.artifacts.measurements import EvaluationMeasurements, ImageMeasurement
from bevcalib.artifacts.result_documents import (
    EvaluationIdentity,
    condition_inventory,
    finalize_run,
    save_scene,
)
from bevcalib.artifacts.run_record import RunRecordV1, load_run_provenance
from bevcalib.cohort.manifest import (
    CohortManifestV2,
    load_formal_manifest,
    load_manifest,
    save_manifest,
)
from bevcalib.cohort.protocol import resolve_protocol
from bevcalib.evaluation_measurements import invalid_timing_result, measure_condition
from bevcalib.nuscenes_adapter.installation import resolve_installation
from bevcalib.operators.image_edges import image_edge_evidence
from bevcalib.operators.lidar_edges import lidar_depth_edges
from bevcalib.preprocessing import load_observation


@dataclass(frozen=True)
class EvaluationResult:
    directory: Path
    identity: EvaluationIdentity


def evaluate_calibration(
    protocol_path: Path,
    manifest_path: Path,
    method: str,
    output_dir: Path,
    *,
    dataroot: Path,
    checkpoint: Path | None = None,
    synthetic_fixture: bool = False,
    model_factory: Callable[[], Any] | None = None,
) -> EvaluationResult:
    if method not in ("identity", "classical", "learned"):
        raise ValueError("unknown evaluation method")
    if method == "learned" and checkpoint is None:
        raise ValueError("learned requires checkpoint")
    if method != "learned" and checkpoint is not None:
        raise ValueError("checkpoint is only valid for learned evaluation")
    protocol = resolve_protocol(protocol_path)
    if synthetic_fixture:
        manifest = load_manifest(manifest_path)
        if (
            not isinstance(manifest, CohortManifestV2)
            or manifest.role != "evaluation"
            or manifest.protocol_hash != protocol.protocol_hash
            or manifest.dataset_version != protocol.dataset_version
            or not manifest.scenes
            or any(scene.official_split != "val" for scene in manifest.scenes)
        ):
            raise ValueError(
                "synthetic evaluation still requires verified V2 evaluation role/protocol/dataset/split"
            )
    else:
        manifest = load_formal_manifest(
            manifest_path,
            expected_role="evaluation",
            protocol_hash=protocol.protocol_hash,
            dataset_version=protocol.dataset_version,
        )
    producer = load_run_provenance()
    predictor = None
    seed = None
    if checkpoint is not None:
        from bevcalib.training.checkpoints import load_learned_checkpoint

        record = RunRecordV1.model_validate_json(
            (checkpoint.parent / "run_record.json").read_bytes()
        )
        seed = record.seed
        predictor = load_learned_checkpoint(
            checkpoint,
            expected_protocol_hash=protocol.protocol_hash,
            expected_seed=seed,
            allow_synthetic=synthetic_fixture,
            model_factory=model_factory,
        )
        predictor.validate_evaluation(manifest)
    installation = resolve_installation(dataroot, protocol.dataset_version)
    actual_scenes = {scene.scene_token: scene for scene in installation.scene_records()}
    if any(actual_scenes.get(scene.scene_token) != scene for scene in manifest.scenes):
        raise ValueError("evaluation frozen cohort differs from original dataset metadata")
    images = {}
    for scene in manifest.scenes:
        for index, token in enumerate(scene.camera_sample_data_tokens):
            observation = load_observation(installation, scene, index)
            edge = image_edge_evidence(observation.rgb, with_distance_field=False)
            images[token] = ImageMeasurement(
                rgb_sha256=hashlib.sha256(observation.rgb.tobytes()).hexdigest(),
                width=observation.rgb.shape[1],
                height=observation.rgb.shape[0],
                edge_threshold=edge.threshold,
            )
    identity = EvaluationIdentity(
        method=method,
        seed=seed,
        checkpoint_sha256=None if predictor is None else predictor.checkpoint_sha256,
        protocol_hash=protocol.protocol_hash,
        dataset_manifest_hash=manifest.manifest_sha256,
        dataset_version=manifest.dataset_version,
        evidence_type="synthetic" if synthetic_fixture else "observed",
        producer=producer,
        measurements=EvaluationMeasurements(table_sha256=installation.table_sha256, images=images),
    )
    directory = output_dir / f"{method}-{identity.run_id}"
    if directory.exists():
        raise FileExistsError("refusing to overwrite an existing evaluation run")
    sidecar = output_dir / "evaluation_manifest.json"
    if sidecar.exists() and load_manifest(sidecar) != manifest:
        raise ValueError("artifact root is already bound to a different cohort")
    for scene in manifest.scenes:
        rows = []
        for index, camera_token in enumerate(scene.camera_sample_data_tokens):
            nominal = load_observation(installation, scene, index)
            edges = image_edge_evidence(nominal.rgb)
            if hashlib.sha256(nominal.rgb.tobytes()).hexdigest() != images[camera_token].rgb_sha256:
                raise ValueError("camera payload drift after measurement identity")
            lidar_edges = {
                nominal.lidar.sample_data_token: lidar_depth_edges(nominal.points).points_lidar_n3
            }
            for axis, level in condition_inventory(method):
                observation = nominal
                if axis == "time":
                    selected = installation.select_timing(camera_token, int(level))
                    if not selected.valid:
                        rows.append(
                            invalid_timing_result(
                                nominal,
                                scene.scene_token,
                                selected,
                                None
                                if selected.selected_sample_data_token is None
                                else installation.packet(
                                    selected.selected_sample_data_token, "LIDAR_TOP"
                                ),
                            )
                        )
                        continue
                    observation = load_observation(
                        installation, scene, index, lidar_token=selected.selected_sample_data_token
                    )
                token = observation.lidar.sample_data_token
                if token not in lidar_edges:
                    lidar_edges[token] = lidar_depth_edges(observation.points).points_lidar_n3
                rows.append(
                    measure_condition(
                        observation,
                        scene_token=scene.scene_token,
                        axis=axis,
                        level=level,
                        method=method,
                        edges=edges,
                        edge_points=lidar_edges[token],
                        predictor=predictor,
                    )
                )
        if not sidecar.exists():
            output_dir.mkdir(parents=True, exist_ok=True)
            save_manifest(manifest, sidecar)
        save_scene(directory, identity, manifest, tuple(rows))
    finalize_run(directory, identity, manifest)
    return EvaluationResult(directory, identity)
