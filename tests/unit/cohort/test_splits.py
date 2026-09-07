"""Synthetic allocations exercise location quotas and scarce independent logs."""

from __future__ import annotations

import hashlib
import random
from collections import Counter

import pytest


def record(token: str, log: str, location: str = "boston-seaport", split: str = "train"):
    from bevcalib.cohort.records import SceneRecord

    return SceneRecord(
        scene_token=token,
        log_token=log,
        location=location,
        official_split=split,
        sample_tokens=(token + "-sample",),
        camera_sample_data_tokens=(token + "-camera",),
        lidar_sample_data_tokens=(token + "-lidar",),
        sample_timestamps=(100,),
        camera_timestamps=(101,),
        lidar_timestamps=(98,),
    )


def population():
    from bevcalib.cohort.records import LOCATIONS

    return [
        record(f"{split}-{loc}-{i}", f"{split}-{loc}-{i // 10}", loc, split)
        for split, count in [("train", 100), ("val", 30)]
        for loc in LOCATIONS
        for i in range(count)
    ]


def freeze(records, protocol_hash="a" * 64):
    from bevcalib.cohort.splits import freeze_scene_cohort

    return freeze_scene_cohort(
        records, protocol_hash=protocol_hash, dataset_version="v1.0-trainval"
    )


def test_formal_counts_strata_disjointness_and_shuffled_identity():
    result = freeze(population())
    assert {r: len(m.scenes) for r, m in result.items()} == {
        "development": 100,
        "calibration": 20,
        "evaluation": 30,
    }
    assert len({s.log_token for s in result["calibration"].scenes}) == 20
    logs = [{s.log_token for s in m.scenes} for m in result.values()]
    assert not (logs[0] & logs[1] or logs[0] & logs[2] or logs[1] & logs[2])
    assert sorted(Counter(s.location for s in result["development"].scenes).values()) == [25] * 4
    assert sorted(Counter(s.location for s in result["calibration"].scenes).values()) == [5] * 4
    assert sorted(Counter(s.location for s in result["evaluation"].scenes).values()) == [7, 7, 8, 8]
    for role, manifest in result.items():
        assert all(
            s.official_split == ("val" if role == "evaluation" else "train")
            for s in manifest.scenes
        )
    shuffled = population()
    random.Random(23).shuffle(shuffled)
    assert result == freeze(shuffled)
    assert result["development"].scenes != freeze(population(), "b" * 64)["development"].scenes


def test_small_adversary_reserves_two_singleton_logs_instead_of_first_large_log():
    from bevcalib.cohort.splits import allocate_training_location

    names = sorted(
        [str(i) for i in range(5)],
        key=lambda s: hashlib.sha256(f"{'a' * 64}:{s}".encode()).hexdigest(),
    )
    records = [record(s, "large" if i < 3 else s) for i, s in enumerate(names)]
    development, calibration = allocate_training_location(
        records, development=3, calibration=2, protocol_hash="a" * 64
    )
    assert {s.log_token for s in development} == {"large"}
    assert {s.scene_token for s in calibration} == set(names[3:])


def test_underfill_is_persisted_without_other_location_backfill():
    records = [record(f"x{i}", "one-log") for i in range(200)]
    result = freeze(records)
    assert len(result["development"].scenes) == 100
    assert result["calibration"].scenes == ()
    assert result["calibration"].allocation[0].requested == 20
    assert result["calibration"].allocation[0].shortage_reason == "joint_log_capacity"
    assert sum(d.requested for d in result["evaluation"].allocation) == 30
    assert all(d.shortage_reason == "insufficient_scenes" for d in result["evaluation"].allocation)


def test_inconsistent_duplicate_scene_or_log_is_refused():
    for bad in [record("one", "different"), record("two", "log", "singapore-onenorth")]:
        with pytest.raises(ValueError, match="inconsistent"):
            freeze([record("one", "log"), bad])


def test_identical_duplicates_do_not_change_counts_or_bytes():
    records = population()
    assert freeze(records) == freeze(records + records[:10])


@pytest.mark.parametrize("extra_val,expected", [(True, (3, 1, 1)), (False, (3, 1, 0))])
def test_shared_split_logs_use_joint_capacity_instead_of_false_shortage(extra_val, expected):
    from bevcalib.cohort.splits import allocate_location

    rows = [record(f"train-{i}", "shared") for i in range(3)] + [
        record("val-shared", "shared", split="val"),
        record("train-single", "train-only"),
    ]
    if extra_val:
        rows.append(record("val-only", "val-only", split="val"))
    result = allocate_location(
        rows, development=3, calibration=1, evaluation=1, protocol_hash="a" * 64
    )
    assert tuple(len(r) for r in result) == expected
    assert {s.log_token for s in result[0]} == {"shared"}
    assert {s.log_token for s in result[1]} == {"train-only"}
    assert all(s.official_split == "train" for s in result[0] + result[1])
    assert all(s.official_split == "val" for s in result[2])
    assert result == allocate_location(
        rows[::-1], development=3, calibration=1, evaluation=1, protocol_hash="a" * 64
    )


def test_full_shared_log_freeze_preserves_splits_and_reports_real_shortage():
    rows = [record(f"dev-{i}", "shared") for i in range(100)]
    rows += [record(f"cal-{i}", f"cal-log-{i}") for i in range(20)]
    rows += [record(f"val-{i}", "shared", split="val") for i in range(30)]
    short = freeze(rows)
    assert {r: len(m.scenes) for r, m in short.items()} == {
        "development": 100,
        "calibration": 20,
        "evaluation": 0,
    }
    assert short["evaluation"].allocation[0].shortage_reason == "joint_log_capacity"
    rows += [record(f"extra-{i}", "val-only", split="val") for i in range(30)]
    complete = freeze(rows)
    assert {r: len(m.scenes) for r, m in complete.items()} == {
        "development": 100,
        "calibration": 20,
        "evaluation": 30,
    }
    logs = [{s.log_token for s in m.scenes} for m in complete.values()]
    assert not (logs[0] & logs[1] or logs[0] & logs[2] or logs[1] & logs[2])
    assert complete == freeze(rows[::-1])


def test_small_joint_capacity_outcomes_match_exhaustive_assignments():
    """An independent tiny oracle detects feasible sets missed by lookahead or DP."""
    from itertools import product

    from bevcalib.cohort.splits import allocate_location

    for capacities in product([(0, 1), (1, 0), (1, 1), (2, 1)], repeat=3):
        records = [
            record(f"{log}-{split}-{i}", str(log), split=split)
            for log, (train, val) in enumerate(capacities)
            for split, count in [("train", train), ("val", val)]
            for i in range(count)
        ]
        outcomes = []
        for assignment in product(range(4), repeat=3):
            dev = sum(capacities[i][0] for i, r in enumerate(assignment) if r == 0)
            cal = sum(capacities[i][0] > 0 for i, r in enumerate(assignment) if r == 1)
            val = sum(capacities[i][1] for i, r in enumerate(assignment) if r == 2)
            outcomes.append((min(2, dev), min(1, cal), min(2, val)))
        actual = allocate_location(
            records, development=2, calibration=1, evaluation=2, protocol_hash="a" * 64
        )
        assert tuple(map(len, actual)) == max(outcomes)
        logs = [{s.log_token for s in rows} for rows in actual]
        assert not (logs[0] & logs[1] or logs[0] & logs[2] or logs[1] & logs[2])


def test_unequal_largest_remainders_and_empty_allocation():
    from bevcalib.cohort.splits import location_quotas

    rows = [record(f"b{i}", str(i)) for i in range(3)] + [record("s", "s", "singapore-queenstown")]
    assert location_quotas(rows, 30) == {
        "boston-seaport": 23,
        "singapore-hollandvillage": 0,
        "singapore-onenorth": 0,
        "singapore-queenstown": 7,
    }
    empty = freeze([])
    assert all(not m.scenes for m in empty.values())
    assert all(d.selected == 0 and d.available == 0 for m in empty.values() for d in m.allocation)
