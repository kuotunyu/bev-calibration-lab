"""Compact frame reductions and operator-specific, scene-paired estimands."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import cast

import numpy as np

from bevcalib.analysis.policy import GT_RANGE_ABSOLUTE_TOLERANCE_M, HIGHER_BETTER
from bevcalib.artifacts.results import CalibrationResultV2, GroundContactResult
from bevcalib.artifacts.statistics import Estimate, PairedEstimate, Support
from bevcalib.metrics.bootstrap import paired_scene_bootstrap
from bevcalib.metrics.calibration import recovered
from bevcalib.metrics.reprojection import range_bin
from bevcalib.operators.ground_contact import MAX_RANGE_M


@dataclass(frozen=True)
class CompactFrame:
    scene: str
    sample: str
    values: dict[str, float | None]
    contacts: dict[str, GroundContactResult]


def compact_frame(row: CalibrationResultV2) -> CompactFrame:
    pose = row.pose
    values: dict[str, float | None] = {
        "rotation_geodesic_deg": None if pose is None else pose.rotation_geodesic_error_deg,
        "translation_norm_cm": None if pose is None else pose.translation_error_m * 100,
        "pixel_frame_p50_px": float(np.quantile(row.pixel_errors_px, 0.5))
        if row.pixel_errors_px
        else None,
        "pixel_frame_p90_px": float(np.quantile(row.pixel_errors_px, 0.9))
        if row.pixel_errors_px
        else None,
        "edge_score_px": row.edge_alignment_score,
        "recovery_rate_pct": None
        if pose is None or row.fault_axis == "time"
        else 100.0 * recovered(pose.rotation_geodesic_error_deg, pose.translation_error_m),
    }
    for prefix, axes, components, factor, unit in (
        (
            "rotation",
            ("roll", "pitch", "yaw"),
            None if pose is None else pose.rotation_rpy_error_deg,
            1,
            "deg",
        ),
        (
            "translation",
            ("x", "y", "z"),
            None if pose is None else pose.translation_xyz_error_m,
            100,
            "cm",
        ),
    ):
        for index, axis in enumerate(axes):
            value = None if components is None else components[index] * factor
            values[f"{prefix}_bias_{axis}_{unit}"] = value
            values[f"{prefix}_abs_{axis}_{unit}"] = None if value is None else abs(value)
    return CompactFrame(
        row.scene_token, row.sample_token, values, {c.box_token: c for c in row.ground_contacts}
    )


@dataclass(frozen=True)
class ScenePair:
    scene: str
    before: float | None
    after: float | None
    total_frames: int
    frames: int
    total_objects: int | None
    objects: int | None
    exclusions: dict[str, int]


def scene_pairs(
    runs: Mapping[str, Sequence[CompactFrame]], before: str, after: tuple[str, ...], metric: str
) -> list[ScenePair]:
    labels = tuple(dict.fromkeys((before, *after)))
    indices = {label: {(row.scene, row.sample): row for row in runs[label]} for label in labels}
    keys = set(indices[before])
    if any(
        set(index) != keys or len(index) != len(runs[label]) for label, index in indices.items()
    ):
        raise ValueError("paired frame inventory differs or is duplicated")
    by_scene = defaultdict(list)
    for key in sorted(keys):
        by_scene[key[0]].append(key)
    result = []
    for scene, scene_keys in by_scene.items():
        values = []
        exclusions: Counter[str] = Counter()
        total_objects = objects = 0
        bev = metric.startswith("bev_frame_mean_m/")
        for key in scene_keys:
            frames = [indices[label][key] for label in labels]
            if bev:
                all_objects = set().union(*(frame.contacts for frame in frames))
                selected = []
                for token in sorted(all_objects):
                    contacts = [frame.contacts.get(token) for frame in frames]
                    ranges = {contact.range_m for contact in contacts if contact is not None}
                    # Geometry arithmetic can differ across CPU builds. Bound the
                    # entire span, with no relative scaling or value rewriting;
                    # discrete bin and operator-cutoff decisions must stay exact.
                    bins = {range_bin(value) for value in ranges}
                    if (
                        max(ranges) - min(ranges) > GT_RANGE_ABSOLUTE_TOLERANCE_M
                        or len(bins) != 1
                        or len({value <= MAX_RANGE_M for value in ranges}) != 1
                    ):
                        raise ValueError("paired object GT range differs between methods")
                    if next(iter(bins)) != metric.split("/", 1)[1]:
                        continue
                    total_objects += 1
                    if all(
                        contact is not None and contact.error_m is not None for contact in contacts
                    ):
                        selected.append(
                            [cast(GroundContactResult, contact).error_m for contact in contacts]
                        )
                    else:
                        exclusions["object_missing_or_invalid_in_common_support"] += 1
                objects += len(selected)
                current = np.mean(selected, axis=0).tolist() if selected else [None] * len(labels)
            else:
                current = [frame.values[metric] for frame in frames]
            if any(value is None for value in current):
                for label, value in zip(labels, current, strict=True):
                    if value is None:
                        exclusions[f"{label}:operator_unavailable"] += 1
                continue
            by_label = dict(zip(labels, current, strict=True))
            values.append((by_label[before], float(np.mean([by_label[label] for label in after]))))
        result.append(
            ScenePair(
                scene,
                float(np.mean([value[0] for value in values])) if values else None,
                float(np.mean([value[1] for value in values])) if values else None,
                len(scene_keys),
                len(values),
                total_objects if bev else None,
                objects if bev else None,
                dict(exclusions),
            )
        )
    return result


def support_for(pairs: Sequence[ScenePair]) -> Support:
    total = sum(pair.total_frames for pair in pairs)
    frames = sum(pair.frames for pair in pairs)
    bev = bool(pairs) and pairs[0].objects is not None
    total_objects = sum(cast(int, pair.total_objects) for pair in pairs) if bev else None
    objects = sum(cast(int, pair.objects) for pair in pairs) if bev else None
    reasons: Counter[str] = Counter()
    for pair in pairs:
        reasons.update(pair.exclusions)
    return Support(
        total_frames=total,
        frames=frames,
        excluded_frames=total - frames,
        total_scenes=len(pairs),
        scenes=sum(pair.frames > 0 for pair in pairs),
        total_objects=total_objects,
        objects=objects,
        excluded_objects=cast(int, total_objects) - cast(int, objects) if bev else None,
        exclusions=dict(reasons),
    )


def finish_estimate(pairs: Sequence[ScenePair]) -> Estimate:
    values = [cast(float, pair.before) for pair in pairs if pair.frames]
    return Estimate(
        value=float(np.mean(values)) if values else None,
        support=support_for(pairs),
        reason=None if values else "no_operator_valid_frames",
    )


def finish_pair(pairs: Sequence[ScenePair], metric: str) -> PairedEstimate:
    measured = [pair for pair in pairs if pair.frames]
    direction = -1 if metric in HIGHER_BETTER else 1
    interval = (
        paired_scene_bootstrap(
            {
                pair.scene: (
                    direction * cast(float, pair.before),
                    direction * cast(float, pair.after),
                )
                for pair in measured
            }
        )
        if measured
        else None
    )
    return PairedEstimate(
        before=float(np.mean([cast(float, pair.before) for pair in measured]))
        if measured
        else None,
        after=float(np.mean([cast(float, pair.after) for pair in measured])) if measured else None,
        improvement=None if interval is None else interval.estimate,
        interval=interval,
        support=support_for(pairs),
        reason=None if measured else "no_common_operator_valid_frames",
    )


def paired_estimate(
    runs: Mapping[str, Sequence[CalibrationResultV2]],
    before: str,
    after: tuple[str, ...],
    metric: str,
) -> PairedEstimate:
    compact = {label: [compact_frame(row) for row in rows] for label, rows in runs.items()}
    return finish_pair(scene_pairs(compact, before, after, metric), metric)
