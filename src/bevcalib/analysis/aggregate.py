"""Complete private V2 scenes to five portable, predeclared formal documents."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from bevcalib.analysis.estimands import (
    CompactFrame,
    ScenePair,
    compact_frame,
    finish_estimate,
    finish_pair,
    scene_pairs,
)
from bevcalib.analysis.policy import (
    COMPARISON_METRICS,
    COMPARISONS,
    ESTIMAND_DESCRIPTION,
    METRIC_UNITS,
)
from bevcalib.artifacts.documents import DOCUMENT_TYPES, FormalArtifactSet, FormalIdentity
from bevcalib.artifacts.envelope import canonical_json_bytes
from bevcalib.artifacts.result_documents import condition_inventory, digest
from bevcalib.artifacts.result_sets import load_run_set
from bevcalib.artifacts.results import TimingEvidence
from bevcalib.metrics.summary import condition_key, statistic
from bevcalib.perturbations.schedule import TIMING_OFFSET_MS


def aggregate_formal_results(
    artifacts_dir: Path, output_dir: Path, *, synthetic_fixture: bool = False
) -> FormalArtifactSet:
    """Publish only after every expected source scene has passed identity/hash checks.

    Raw point arrays live for one run-scene read. Compact contacts/quantiles live
    for one five-method scene. Only scene estimands and counts survive that scene.
    """
    if output_dir.exists():
        raise FileExistsError("refusing to overwrite formal evidence")
    manifest, runs = load_run_set(
        artifacts_dir, synthetic_fixture=synthetic_fixture, require_all_seeds=True
    )
    identity = FormalIdentity(
        protocol_hash=manifest.protocol_hash,
        dataset_manifest_hash=manifest.manifest_sha256,
        dataset_version=manifest.dataset_version,
        evidence_type="synthetic" if synthetic_fixture else "observed",
        measurement_identity_sha256=digest(
            runs[0].marker.identity.measurements.model_dump(mode="json")
        ),
        source_runs={
            run.label: {
                "method": run.marker.identity.method,
                "seed": run.marker.identity.seed,
                "checkpoint_sha256": run.marker.identity.checkpoint_sha256,
                "run_identity_sha256": run.marker.identity.run_id,
                "source_complete_sha256": run.source_complete_sha256,
                "producer_commit": run.marker.identity.producer.commit,
                "producer_lock_sha256": run.marker.identity.producer.lock_sha256,
            }
            for run in runs
        },
        scene_count=len(manifest.scenes),
        sample_count=sum(len(scene.sample_tokens) for scene in manifest.scenes),
        estimands=ESTIMAND_DESCRIPTION,
        units=METRIC_UNITS,
    )
    singles: dict[tuple[str, str, str], list[ScenePair]] = defaultdict(list)
    pairs: dict[tuple[str, str, str], list[ScenePair]] = defaultdict(list)
    counts: dict[tuple[str, str], dict[str, Any]] = {}
    timing: dict[int, list[TimingEvidence]] = defaultdict(list)
    for scene in manifest.scenes:
        compact: dict[str, dict[str, list[CompactFrame]]] = {}
        for run in runs:
            conditions: dict[str, list[CompactFrame]] = defaultdict(list)
            raw_rows = run.scene_rows(manifest, scene.scene_token)
            for row in raw_rows:
                key = condition_key(row.fault_axis, row.fault_level)
                conditions[key].append(compact_frame(row))
                count = counts.setdefault(
                    (run.label, key),
                    {
                        "total": 0,
                        "valid": 0,
                        "invalid": 0,
                        "reasons": Counter(),
                        "projection_input_points": 0,
                        "within_row_pixel_error_count": 0,
                    },
                )
                count["total"] += 1
                count["valid"] += row.valid
                count["invalid"] += not row.valid
                if not row.valid:
                    count["reasons"][row.invalid_reason] += 1
                count["projection_input_points"] += row.projection_count
                count["within_row_pixel_error_count"] += len(row.pixel_errors_px)
                if row.fault_axis == "time":
                    timing[int(row.fault_level)].append(row.timing)
            del row, raw_rows
            compact[run.label] = conditions
            for key, frames in conditions.items():
                for metric in METRIC_UNITS:
                    singles[run.label, key, metric].extend(
                        scene_pairs({run.label: frames}, run.label, (run.label,), metric)
                    )
        for axis, level in condition_inventory("classical"):
            key = condition_key(axis, level)
            selected = {label: conditions[key] for label, conditions in compact.items()}
            for comparison, (before, after) in COMPARISONS.items():
                for metric in COMPARISON_METRICS:
                    pairs[comparison, key, metric].extend(
                        scene_pairs(selected, before, after, metric)
                    )
        del compact, selected, conditions, frames
    metrics: dict[str, dict[str, Any]] = defaultdict(dict)
    for (label, key, metric), measurements in singles.items():
        metrics[label].setdefault(key, {})[metric] = finish_estimate(measurements)
    intervals: dict[str, dict[str, Any]] = defaultdict(dict)
    for (comparison, key, metric), measurements in pairs.items():
        intervals[comparison].setdefault(key, {})[metric] = finish_pair(measurements, metric)
    exclusions: dict[str, dict[str, Any]] = defaultdict(dict)
    for (label, key), count in counts.items():
        exclusions[label][key] = count | {
            "operators": {metric: value.support for metric, value in metrics[label][key].items()}
        }
    paired_support = {
        label: {
            key: {metric: value.support for metric, value in values.items()}
            for key, values in conditions.items()
        }
        for label, conditions in intervals.items()
    }
    offsets = {}
    for offset in TIMING_OFFSET_MS:
        selections = timing[offset]
        valid = sum(selection.reason == "valid" for selection in selections)
        realized = [
            selection.realized_offset_ms
            for selection in selections
            if selection.realized_offset_ms is not None
        ]
        errors = [
            selection.absolute_error_ms
            for selection in selections
            if selection.absolute_error_ms is not None
        ]
        offsets[str(offset)] = {
            "requested_offset_ms": offset,
            "total": len(selections),
            "selected": len(realized),
            "valid": valid,
            "invalid": len(selections) - valid,
            "valid_fraction": valid / len(selections),
            "reasons": dict(Counter(selection.reason for selection in selections)),
            "realized_offset_ms": statistic(realized),
            "absolute_error_ms": statistic(errors),
            "metrics": metrics["identity"][condition_key("time", offset)],
        }
    payloads = {
        "metrics": {"runs": metrics},
        "intervals": {"comparisons": intervals},
        "recovery": {
            "runs": {
                label: {
                    key: values["recovery_rate_pct"]
                    for key, values in conditions.items()
                    if not key.startswith("time:")
                }
                for label, conditions in metrics.items()
            },
            "comparisons": {
                label: {key: values["recovery_rate_pct"] for key, values in conditions.items()}
                for label, conditions in intervals.items()
            },
        },
        "timing": {"offsets": offsets},
        "exclusions": {"runs": exclusions, "comparisons": paired_support},
    }
    documents = {}
    for name, payload in payloads.items():
        body = json.loads(
            json.dumps(
                {"schema_version": f"bev-calibration-{name}/v1", "identity": identity, **payload},
                default=lambda value: value.model_dump(mode="json"),
            )
        )
        documents[name] = DOCUMENT_TYPES[name].model_validate(
            body | {"document_sha256": digest(body)}
        )
    result = FormalArtifactSet.model_validate(documents)
    output_dir.mkdir(parents=True)
    for name, document in documents.items():
        with (output_dir / f"{name}.json").open("xb") as handle:
            handle.write(canonical_json_bytes(document.model_dump(mode="json")) + b"\n")
    return result
