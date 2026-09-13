"""Publish paired scene-mean calibration errors through the optional envelope."""

from __future__ import annotations

import hashlib
import math
import subprocess
from collections.abc import Sequence
from pathlib import Path

from bevcalib.artifacts.envelope import PortfolioArtifactEnvelopeV1, canonical_json_bytes
from bevcalib.artifacts.result_documents import condition_inventory
from bevcalib.artifacts.result_sets import load_run_set
from bevcalib.artifacts.results import PoseErrors

ARTIFACT_TYPE = "calibration-error-distribution/v1"
# Shared interchange bounds, independently enforced by the consumer as well.
MAX_ROTATION_DEG = 30.0
MAX_TRANSLATION_M = 2.0


def _producer_commit() -> str:
    root = Path(__file__).resolve().parents[3]
    try:
        commit = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        source_name = Path(__file__).resolve().relative_to(root).as_posix()
        subprocess.run(
            ["git", "-C", str(root), "cat-file", "-e", f"{commit}:{source_name}"],
            check=True,
            capture_output=True,
            text=True,
        )
        dirty = subprocess.run(
            [
                "git",
                "-C",
                str(root),
                "status",
                "--porcelain",
                "--untracked-files=all",
                "--",
                "src",
                "configs",
                "pyproject.toml",
                "uv.lock",
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError("export requires an identifiable committed source checkout") from exc
    if dirty:
        raise ValueError("export requires committed source, configuration and lock")
    return commit


def _scene_mean(poses: Sequence[PoseErrors]) -> tuple[list[float], list[float]]:
    samples = []
    for pose in poses:
        rotation = pose.rotation_rpy_error_deg
        translation = pose.translation_xyz_error_m
        if len(rotation) != 3 or len(translation) != 3:
            raise ValueError("paired pose samples must have three components each")
        values = (*rotation, *translation)
        if any(not math.isfinite(value) for value in values):
            raise ValueError("pose samples must be finite")
        samples.append(values)
    mean = [math.fsum(row[index] for row in samples) / len(samples) for index in range(6)]
    if any(abs(value) > MAX_ROTATION_DEG for value in mean[:3]) or any(
        abs(value) > MAX_TRANSLATION_M for value in mean[3:]
    ):
        raise ValueError("scene mean exceeds the P3 interchange bound; no clipping is permitted")
    return mean[:3], mean[3:]


def export_calibration_distribution(
    input_root: Path,
    output_path: Path,
    *,
    method: str,
    seed: int | None,
    axis: str,
    level: float,
    producer_release: str,
    created_at_utc: str,
) -> Path:
    """Reduce one predefined condition after complete source validation.

    No sensor data or sibling package is imported. Pose eligibility is independent
    of global row validity. Unsupported scenes remain in the support accounting;
    supported scenes each contribute one paired six-component sample.
    """
    if output_path.exists():
        raise FileExistsError("refusing to overwrite calibration distribution")
    if axis == "time" or (axis, level) not in condition_inventory(method):
        raise ValueError("export requires a declared extrinsic condition, not timing")
    commit = _producer_commit()
    manifest, runs = load_run_set(input_root, require_all_seeds=True)
    selected = [
        run
        for run in runs
        if (run.marker.identity.method, run.marker.identity.seed) == (method, seed)
    ]
    if len(selected) != 1:
        raise ValueError("requested method and seed do not identify one complete source run")
    run = selected[0]
    rotations = []
    translations = []
    frame_support = []
    total_frames = 0
    valid_frames = 0
    for scene in manifest.scenes:
        rows = run.scene_rows(manifest, scene.scene_token)
        chosen = [row for row in rows if (row.fault_axis, row.fault_level) == (axis, level)]
        total_frames += len(chosen)
        poses = [row.pose for row in chosen if row.pose is not None]
        valid_frames += len(poses)
        if poses:
            rotation, translation = _scene_mean(poses)
            rotations.append(rotation)
            translations.append(translation)
            frame_support.append(len(poses))
        del rows, chosen, poses
    if not rotations:
        raise ValueError("distribution has no pose-valid scene support")
    identity = run.marker.identity
    payload = {
        "schema_version": ARTIFACT_TYPE,
        "sampling_unit": "scene-mean over pose-valid frames",
        "evidence_type": "observed",
        "condition": {"method": method, "seed": seed, "axis": axis, "level": level},
        "units": {"rotation": "degrees", "translation": "metres"},
        "rotation_samples_rpy_deg": rotations,
        "translation_samples_xyz_m": translations,
        "sample_pose_valid_frames": frame_support,
        "support": {
            "total_scenes": len(manifest.scenes),
            "valid_scenes": len(rotations),
            "excluded_scenes": len(manifest.scenes) - len(rotations),
            "total_frames": total_frames,
            "pose_valid_frames": valid_frames,
            "excluded_frames": total_frames - valid_frames,
        },
        "source": {
            "run_identity_sha256": identity.run_id,
            "source_complete_sha256": run.source_complete_sha256,
            "evaluator_commit": identity.producer.commit,
            "evaluator_lock_sha256": identity.producer.lock_sha256,
        },
    }
    envelope = PortfolioArtifactEnvelopeV1(
        schema_version="portfolio-artifact-envelope/v1",
        producer_repository="bev-calibration-lab",
        producer_release=producer_release,
        producer_commit=commit,
        artifact_type=ARTIFACT_TYPE,
        protocol_hash=manifest.protocol_hash,
        dataset_manifest_hash=manifest.manifest_sha256,
        created_at_utc=created_at_utc,
        payload_sha256=hashlib.sha256(canonical_json_bytes(payload)).hexdigest(),
        payload=payload,
    )
    contents = canonical_json_bytes(envelope.model_dump(mode="json")) + b"\n"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("xb") as handle:
        handle.write(contents)
    return output_path
