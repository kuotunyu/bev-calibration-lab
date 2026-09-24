"""Derived operating envelope of the v1 correctors, computed from the frozen evidence.

The five formal documents answer "how large is each error" one condition at a time.
This module answers a question they do not state directly: across the whole fault
grid, where does applying a corrector beat leaving the calibration alone, and how
large is the error a corrector leaves behind? It only counts, orders and copies
values that already exist in `metrics.json`, `intervals.json` and `recovery.json`.
Nothing is re-estimated, no interval is recomputed and no threshold is chosen from
the result, so the output is labelled `derived` and carries its own source identity.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from bevcalib.analysis.formal_claims import load_publication, pointer_token
from bevcalib.artifacts.documents import FormalArtifactSet
from bevcalib.artifacts.result_documents import condition_inventory, digest
from bevcalib.metrics.calibration import RECOVERY_ROTATION_THRESHOLD_DEG
from bevcalib.metrics.summary import condition_key

SCHEMA_VERSION = "bev-calibration-operating-envelope/v1"
SOURCES = ("metrics", "intervals", "recovery")
# The released v1 evidence. A derived analysis names the exact documents it read;
# pointing it at other evidence is refused unless the caller names those instead.
V1_SOURCE_SHA256 = {
    "metrics": "45c5754cb9d2611a63bb2b96c11c72c42829369c51f063a11d7021c8ba4a9088",
    "intervals": "a69a95b255c203eb47b4d364dbaf831883227f9d5e2c1d79037b24aed4623c72",
    "recovery": "648d8f33801745d08b3cad912432c5ef3aa08519930e4be6533ac986133987d7",
}
METHODS = ("identity", "classical", "learned-17", "learned-42", "learned-73")
LEARNED = ("learned-17", "learned-42", "learned-73")
# Fault axes are composed on the camera side of the CAM_FRONT extrinsic, so each
# formal axis is an axis of the camera optical frame (x right, y down, z forward).
AXES = {
    "roll": ("tilt", "rotation about camera x (right)", "degree"),
    "pitch": ("pan", "rotation about camera y (down)", "degree"),
    "yaw": ("in-plane rotation", "rotation about camera z (optical axis)", "degree"),
    "x": ("lateral offset", "translation along camera x (right)", "metre"),
    "y": ("vertical offset", "translation along camera y (down)", "metre"),
    "z": ("forward offset", "translation along camera z (optical axis)", "metre"),
}
POSE_AND_PIXEL = (
    "rotation_geodesic_deg",
    "translation_norm_cm",
    "pixel_frame_p50_px",
    "pixel_frame_p90_px",
)
COUNTED = {
    "classical->learned-fixed-three-seed-mean": (
        *POSE_AND_PIXEL,
        "edge_score_px",
        "recovery_rate_pct",
    ),
    # Recovery against identity is not counted: identity's recovery at the grid
    # levels equal to the rotation threshold is a floating-point boundary artifact.
    "identity->learned-fixed-three-seed-mean": POSE_AND_PIXEL,
    "identity->classical": POSE_AND_PIXEL,
}
BREAK_EVEN_METRIC = "pixel_frame_p50_px"
DEFINITIONS = {
    "interval_counts": (
        "per comparison and estimand, the number of the sixty extrinsic conditions whose "
        "paired 95% scene-bootstrap interval of the improvement lies entirely above zero "
        "(after method better), entirely below zero (before method better), contains zero, "
        "or is unavailable; improvement directions are those of the source documents"
    ),
    "break_even": (
        "per axis, the smallest grid magnitude at which the after method is better on "
        "pixel_frame_p50_px at both signs and at every larger grid magnitude; null when "
        "the largest magnitude does not qualify"
    ),
    "residual": (
        "minimum and maximum over the sixty extrinsic conditions of each method's "
        "scene-mean rotation_geodesic_deg and translation_norm_cm"
    ),
    "pan_residual": "each method's rotation_abs_pitch_deg at every pitch (pan) condition",
    "recovery_boundary": (
        "identity at rotation grid levels equal to the recovery threshold: its geodesic "
        "error, its recovery rate, and how many identity->* recovery intervals lie above zero"
    ),
    "series": "pixel_frame_p50_px per method and condition, and the identity->"
    "learned-fixed-three-seed-mean interval verdict per condition, for the figure",
}


def _verdict(interval: Any) -> str:
    if interval is None:
        return "unavailable"
    if interval.low > 0:
        return "after_better"
    if interval.high < 0:
        return "before_better"
    return "inconclusive"


def _conditions() -> list[tuple[str, float, str]]:
    return [
        (axis, level, condition_key(axis, level))
        for axis, level in condition_inventory("classical")
    ]


def _counts(estimates: Mapping[str, Any]) -> dict[str, Any]:
    verdicts = {key: _verdict(value.interval) for key, value in estimates.items()}
    result: dict[str, Any] = {
        name: sum(verdict == name for verdict in verdicts.values())
        for name in ("after_better", "before_better", "inconclusive", "unavailable")
    }
    result["total"] = len(verdicts)
    result["after_better_conditions"] = sorted(
        key for key, verdict in verdicts.items() if verdict == "after_better"
    )
    return result


def _break_even(estimates: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for axis, (_, _, unit) in AXES.items():
        magnitudes = sorted(
            {abs(level) for candidate, level, _ in _conditions() if candidate == axis} - {0.0}
        )
        qualifying = None
        for magnitude in reversed(magnitudes):
            keys = (condition_key(axis, magnitude), condition_key(axis, -magnitude))
            if all(_verdict(estimates[key].interval) == "after_better" for key in keys):
                qualifying = magnitude
            else:
                break
        result[axis] = {"magnitude": qualifying, "unit": unit}
    return result


def _extreme(values: Mapping[str, float | None]) -> dict[str, Any]:
    present = {key: value for key, value in values.items() if value is not None}
    if not present:
        return {"min": None, "min_condition": None, "max": None, "max_condition": None}
    low = min(present, key=lambda key: (present[key], key))
    high = max(present, key=lambda key: (present[key], key))
    return {
        "min": present[low],
        "min_condition": low,
        "max": present[high],
        "max_condition": high,
    }


def analyse(
    artifacts: FormalArtifactSet,
    source_paths: Mapping[str, str],
    *,
    expected_sha256: Mapping[str, str] = V1_SOURCE_SHA256,
) -> dict[str, Any]:
    """Return the derived document for one validated formal artifact set."""

    documents = {name: getattr(artifacts, name) for name in SOURCES}
    actual = {name: document.document_sha256 for name, document in documents.items()}
    if actual != dict(expected_sha256):
        raise ValueError("operating envelope source documents differ from the declared evidence")
    identity = artifacts.metrics.identity
    runs = artifacts.metrics.runs
    comparisons = artifacts.intervals.comparisons
    recovery = artifacts.recovery.comparisons
    conditions = _conditions()

    def paired(comparison: str, metric: str) -> dict[str, Any]:
        if metric == "recovery_rate_pct":
            return dict(recovery[comparison])
        return {key: comparisons[comparison][key][metric] for _, _, key in conditions}

    boundary = [
        key
        for axis, level, key in conditions
        if axis in ("roll", "pitch", "yaw") and abs(level) == RECOVERY_ROTATION_THRESHOLD_DEG
    ]
    boundary_comparisons = [
        recovery[name][key]
        for name in sorted(recovery)
        if name.startswith("identity->")
        for key in boundary
    ]
    mean = "identity->learned-fixed-three-seed-mean"
    body: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "evidence_type": "derived",
        "producer": "bevcalib.analysis.operating_envelope",
        "definitions": DEFINITIONS,
        "source": {
            "evidence_type": identity.evidence_type,
            "protocol_hash": identity.protocol_hash,
            "dataset_manifest_hash": identity.dataset_manifest_hash,
            "scene_count": identity.scene_count,
            "documents": {
                name: {"path": source_paths[name], "document_sha256": actual[name]}
                for name in SOURCES
            },
        },
        "axes": {
            axis: {"physical": physical, "frame_axis": frame_axis, "unit": unit}
            for axis, (physical, frame_axis, unit) in AXES.items()
        },
        "interval_counts": {
            comparison: {metric: _counts(paired(comparison, metric)) for metric in metrics}
            for comparison, metrics in COUNTED.items()
        },
        "break_even": {
            comparison: _break_even(paired(comparison, BREAK_EVEN_METRIC))
            for comparison in ("identity->learned-fixed-three-seed-mean", "identity->classical")
        },
        "residual": {
            method: {
                metric: _extreme({key: runs[method][key][metric].value for _, _, key in conditions})
                for metric in ("rotation_geodesic_deg", "translation_norm_cm")
            }
            for method in ("classical", *LEARNED)
        },
        "pan_residual": {
            method: {
                key: runs[method][key]["rotation_abs_pitch_deg"].value
                for axis, _, key in conditions
                if axis == "pitch"
            }
            for method in METHODS
        },
        "recovery_boundary": {
            "threshold_deg": RECOVERY_ROTATION_THRESHOLD_DEG,
            "conditions": boundary,
            "identity_rotation_geodesic_deg": {
                key: runs["identity"][key]["rotation_geodesic_deg"].value for key in boundary
            },
            "identity_recovery_rate_pct": {
                key: artifacts.recovery.runs["identity"][key].value for key in boundary
            },
            "identity_comparisons": len(boundary_comparisons),
            "identity_comparisons_above_zero": sum(
                _verdict(value.interval) == "after_better" for value in boundary_comparisons
            ),
        },
        "series": {
            "pixel_frame_p50_px": {
                method: {
                    key: runs[method][key][BREAK_EVEN_METRIC].value for _, _, key in conditions
                }
                for method in METHODS
            },
            "verdict": {
                key: _verdict(comparisons[mean][key][BREAK_EVEN_METRIC].interval)
                for _, _, key in conditions
            },
            "pointers": {
                "pixel_frame_p50_px": "metrics.json#/runs/{method}/{condition}/"
                + pointer_token(BREAK_EVEN_METRIC)
                + "/value",
                "verdict": "intervals.json#/comparisons/"
                + pointer_token(mean)
                + "/{condition}/"
                + pointer_token(BREAK_EVEN_METRIC)
                + "/interval",
            },
        },
    }
    return body | {"document_sha256": digest(body)}


def encode(document: Mapping[str, Any]) -> bytes:
    """Deterministic, human-readable bytes; NaN is refused rather than written."""

    return (json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def build_operating_envelope(
    artifacts_dir: Path,
    output_dir: Path,
    *,
    repository_root: Path,
    expected_sha256: Mapping[str, str] = V1_SOURCE_SHA256,
) -> tuple[Path, Path]:
    """Write the derived document and its figure into a new directory."""

    from bevcalib.report.envelope_figure import render_envelope_svg

    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    artifacts = load_publication(artifacts_dir, repository_root)
    root = repository_root.resolve()
    paths = {
        name: (artifacts_dir / f"{name}.json").resolve().relative_to(root).as_posix()
        for name in SOURCES
    }
    document = analyse(artifacts, paths, expected_sha256=expected_sha256)
    figure = render_envelope_svg(document)
    output_dir.mkdir(parents=True)
    data = output_dir / "operating-envelope.json"
    data.write_bytes(encode(document))
    svg = output_dir / "operating-envelope.svg"
    svg.write_text(figure, encoding="utf-8", newline="\n")
    return data, svg


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args(argv)
    for path in build_operating_envelope(
        arguments.artifacts_dir, arguments.output_dir, repository_root=Path.cwd()
    ):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
