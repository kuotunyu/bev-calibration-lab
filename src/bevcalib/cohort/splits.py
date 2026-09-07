"""Deterministic stratified allocation with polynomial joint log-capacity lookahead."""

from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict

from .manifest import ROLE_COUNTS, CohortManifestV2, Role, manifest_hash
from .records import LOCATIONS, Location, SceneRecord, validate_records


def _ordered(records: Sequence[SceneRecord], protocol_hash: str) -> list[SceneRecord]:
    return sorted(
        records,
        key=lambda s: (
            hashlib.sha256(f"{protocol_hash}:{s.scene_token}".encode()).hexdigest(),
            s.scene_token,
        ),
    )


def location_quotas(records: Sequence[SceneRecord], requested: int) -> dict[Location, int]:
    """Integer largest remainder; lexical location breaks ties; empty input is uniform."""

    weights = Counter(s.location for s in records)
    if not records:
        weights.update(LOCATIONS)
    total = sum(weights.values())
    quotas = {loc: requested * weights[loc] // total for loc in LOCATIONS}
    ranked = sorted(LOCATIONS, key=lambda loc: (-(requested * weights[loc] % total), loc))
    for loc in ranked[: requested - sum(quotas.values())]:
        quotas[loc] += 1
    return quotas


def allocate_training_location(
    records: Sequence[SceneRecord], *, development: int, calibration: int, protocol_hash: str
) -> tuple[list[SceneRecord], list[SceneRecord]]:
    """Reserve distinct calibration logs without starving an achievable development quota.

    A reservation costs every scene in that log. The cheapest k remaining logs
    give the exact minimum cost of finishing k reservations, so each hash-first
    candidate can be accepted iff that completion leaves development capacity.
    If full quotas are impossible, retain the attainable development quota first,
    then maximize calibration up to its quota. No cross-location substitution.
    """

    ordered = _ordered(records, protocol_hash)
    capacities = Counter(s.log_token for s in records)
    budget = len(records) - min(development, len(records))
    target = 0
    spent = 0
    for capacity in sorted(capacities.values()):
        if target == calibration or spent + capacity > budget:
            break
        target += 1
        spent += capacity
    selected: list[SceneRecord] = []
    reserved: set[str] = set()
    spent = 0
    for scene in ordered:
        if len(selected) == target:
            break
        if scene.log_token in reserved:
            continue
        remaining = sorted(
            size
            for log, size in capacities.items()
            if log not in reserved and log != scene.log_token
        )
        required = target - len(selected) - 1
        cost = capacities[scene.log_token]
        if spent + cost + sum(remaining[:required]) <= budget:
            selected.append(scene)
            reserved.add(scene.log_token)
            spent += cost
    return [s for s in ordered if s.log_token not in reserved][:development], selected


def freeze_scene_cohort(
    records: Sequence[SceneRecord], *, protocol_hash: str, dataset_version: str
) -> dict[str, CohortManifestV2]:
    """Freeze 100/20/30 role manifests without reading data or hiding shortages."""

    unique = validate_records(records)
    train = [s for s in unique if s.official_split == "train"]
    val = [s for s in unique if s.official_split == "val"]
    quotas = {
        role: location_quotas(val if role == "evaluation" else train, count)
        for role, count in ROLE_COUNTS.items()
    }
    chosen: dict[Role, list[SceneRecord]] = {role: [] for role in ROLE_COUNTS}
    for location in LOCATIONS:
        development, calibration, evaluation = allocate_location(
            [s for s in unique if s.location == location],
            development=quotas["development"][location],
            calibration=quotas["calibration"][location],
            evaluation=quotas["evaluation"][location],
            protocol_hash=protocol_hash,
        )
        chosen["development"].extend(development)
        chosen["calibration"].extend(calibration)
        chosen["evaluation"].extend(evaluation)
    result: dict[str, CohortManifestV2] = {}
    for role, scenes in chosen.items():
        pool = val if role == "evaluation" else train
        diagnostics = []
        for location in LOCATIONS:
            available = [s for s in pool if s.location == location]
            selected = sum(s.location == location for s in scenes)
            requested = quotas[role][location]
            reason = None
            if selected < requested:
                reason = (
                    "insufficient_scenes" if len(available) < requested else "joint_log_capacity"
                )
            diagnostics.append(
                {
                    "location": location,
                    "requested": requested,
                    "available": len(available),
                    "available_logs": len({s.log_token for s in available}),
                    "selected": selected,
                    "shortage_reason": reason,
                }
            )
        body = {
            "schema_version": "bev-calibration-cohort/v2",
            "role": role,
            "protocol_hash": protocol_hash,
            "dataset_version": dataset_version,
            "scenes": [asdict(s) for s in scenes],
            "allocation": diagnostics,
        }
        result[role] = CohortManifestV2.model_validate(
            body | {"manifest_sha256": manifest_hash(body)}
        )
    return result


Capacity = tuple[int, int, int]


def _options(train: int, val: int) -> list[tuple[int, Capacity]]:
    options = []
    if train:
        options.extend([(1, (0, 1, 0)), (0, (train, 0, 0))])
    if val:
        options.append((2, (0, 0, val)))
    options.append((3, (0, 0, 0)))
    return options


def _mixed_location(
    records: Sequence[SceneRecord], quotas: Capacity, protocol_hash: str
) -> tuple[list[SceneRecord], list[SceneRecord], list[SceneRecord]]:
    ordered = _ordered(records, protocol_hash)
    logs = list(dict.fromkeys(s.log_token for s in ordered))
    train = Counter(s.log_token for s in records if s.official_split == "train")
    val = Counter(s.log_token for s in records if s.official_split == "val")
    options = [_options(train[log], val[log]) for log in logs]
    # At most (D+1)*(C+1)*(E+1) states per suffix, independent of 4**log_count.
    suffix: list[set[Capacity]] = [set() for _ in range(len(logs) + 1)]
    suffix[-1] = {(0, 0, 0)}
    for index in reversed(range(len(logs))):
        suffix[index] = {
            (
                min(quotas[0], state[0] + delta[0]),
                min(quotas[1], state[1] + delta[1]),
                min(quotas[2], state[2] + delta[2]),
            )
            for state in suffix[index + 1]
            for _, delta in options[index]
        }
    target = max(suffix[0])  # Explicit underfill priority: development, calibration, evaluation.
    remaining = target
    assigned: list[set[str]] = [set(), set(), set(), set()]
    for index, log in enumerate(logs):
        # Suffix feasibility guarantees at least one choice, including unused.
        # next expresses that total choice without an impossible loop-exhaustion path.
        role, remaining = next(
            (role, needed)
            for role, delta in options[index]
            if role == 3 or remaining[role] > 0
            for needed in [
                (
                    max(0, remaining[0] - delta[0]),
                    max(0, remaining[1] - delta[1]),
                    max(0, remaining[2] - delta[2]),
                )
            ]
            if any(
                all(a >= b for a, b in zip(state, needed, strict=True))
                for state in suffix[index + 1]
            )
        )
        assigned[role].add(log)
    development = [
        s for s in ordered if s.log_token in assigned[0] and s.official_split == "train"
    ][: target[0]]
    calibration = []
    for scene in ordered:
        if scene.log_token in assigned[1] and scene.official_split == "train":
            calibration.append(scene)
            assigned[1].remove(scene.log_token)
    evaluation = [s for s in ordered if s.log_token in assigned[2] and s.official_split == "val"][
        : target[2]
    ]
    return development, calibration, evaluation


def allocate_location(
    records: Sequence[SceneRecord],
    *,
    development: int,
    calibration: int,
    evaluation: int,
    protocol_hash: str,
) -> tuple[list[SceneRecord], list[SceneRecord], list[SceneRecord]]:
    """Allocate all three roles jointly, including scene-level splits sharing a log."""

    train = [s for s in records if s.official_split == "train"]
    val = [s for s in records if s.official_split == "val"]
    if {s.log_token for s in train} & {s.log_token for s in val}:
        return _mixed_location(records, (development, calibration, evaluation), protocol_hash)
    dev, cal = allocate_training_location(
        train, development=development, calibration=calibration, protocol_hash=protocol_hash
    )
    return dev, cal, _ordered(val, protocol_hash)[:evaluation]
