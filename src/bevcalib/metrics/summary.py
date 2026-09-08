"""Token-free descriptive summaries from compatible, complete private result runs."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from bevcalib.artifacts.result_documents import (
    condition_inventory,
    digest,
    load_result_run,
)
from bevcalib.artifacts.result_sets import FORMAL_RUNS as FORMAL_RUNS
from bevcalib.artifacts.result_sets import SYNTHETIC_PAIR as SYNTHETIC_PAIR
from bevcalib.artifacts.result_sets import load_run_set
from bevcalib.artifacts.results import CalibrationResultV2
from bevcalib.metrics.calibration import recovered
from bevcalib.metrics.reprojection import RANGE_BINS, range_bin


def statistic(values: Sequence[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": len(values),
        "mean": float(array.mean()) if values else None,
        "median": float(np.median(array)) if values else None,
        "p90": float(np.quantile(array, 0.9)) if values else None,
    }


def condition_key(axis: str, level: float) -> str:
    return f"{axis}:{level:g}"


def summarize_condition(rows: Sequence[CalibrationResultV2]) -> dict[str, Any]:
    valid = sum(row.valid for row in rows)
    poses = [row.pose for row in rows if row.pose is not None]
    pose = {
        "rotation_rpy_error_deg": [
            statistic([p.rotation_rpy_error_deg[index] for p in poses]) for index in range(3)
        ],
        "translation_xyz_error_m": [
            statistic([p.translation_xyz_error_m[index] for p in poses]) for index in range(3)
        ],
        "rotation_geodesic_error_deg": statistic([p.rotation_geodesic_error_deg for p in poses]),
        "translation_error_m": statistic([p.translation_error_m for p in poses]),
    }
    contacts = [contact for row in rows for contact in row.ground_contacts]
    bins = {}
    for name in RANGE_BINS:
        selected = [contact for contact in contacts if range_bin(contact.range_m) == name]
        measured = [contact.error_m for contact in selected if contact.error_m is not None]
        bins[name] = statistic(measured) | {
            "total": len(selected),
            "invalid": len(selected) - len(measured),
            "reasons": dict(
                sorted(
                    Counter(
                        contact.invalid_reason for contact in selected if contact.error_m is None
                    ).items()
                )
            ),
        }
    successes = sum(recovered(p.rotation_geodesic_error_deg, p.translation_error_m) for p in poses)
    return {
        "fault_axis": rows[0].fault_axis,
        "fault_level": rows[0].fault_level,
        "validity": {
            "total": len(rows),
            "valid": valid,
            "invalid": len(rows) - valid,
            "invalid_rate": (len(rows) - valid) / len(rows),
            "reasons": dict(
                sorted(Counter(row.invalid_reason for row in rows if not row.valid).items())
            ),
        },
        "pose": pose,
        "pixel_error_px": statistic([value for row in rows for value in row.pixel_errors_px])
        | {"projection_count": sum(row.projection_count for row in rows)},
        "edge_alignment_score": statistic(
            [row.edge_alignment_score for row in rows if row.edge_alignment_score is not None]
        ),
        "ground_contact_by_range": bins,
        "pose_recovery": None
        if rows[0].fault_axis == "time"
        else {
            "valid": len(poses),
            "recovered": successes,
            "rate": successes / len(poses) if poses else None,
        },
    }


def summarize_result_runs(
    artifacts_dir: Path, *, synthetic_fixture: bool = False
) -> dict[str, Any]:
    manifest, verified_runs = load_run_set(artifacts_dir, synthetic_fixture=synthetic_fixture)
    measurements = verified_runs[0].marker.identity.measurements
    runs = {}
    for run in verified_runs:
        identity = run.marker.identity
        rows = load_result_run(run.directory, identity, manifest)
        label = run.label
        conditions = {}
        for axis, level in condition_inventory(identity.method):
            selected = [row for row in rows if row.fault_axis == axis and row.fault_level == level]
            conditions[condition_key(axis, level)] = summarize_condition(selected)
        runs[label] = {
            "method": identity.method,
            "seed": identity.seed,
            "checkpoint_sha256": identity.checkpoint_sha256,
            "run_identity_sha256": identity.run_id,
            "source_complete_sha256": run.source_complete_sha256,
            "conditions": conditions,
        }
    body = {
        "schema_version": "bev-calibration-summary/v1",
        "protocol_hash": manifest.protocol_hash,
        "dataset_manifest_hash": manifest.manifest_sha256,
        "evidence_type": "synthetic" if synthetic_fixture else "observed",
        "measurement_policy": measurements.model_dump(
            mode="json", exclude={"images", "table_sha256"}
        ),
        "measurement_identity_sha256": digest(measurements.model_dump(mode="json")),
        "runs": runs,
    }
    return body | {"document_sha256": digest(body)}
