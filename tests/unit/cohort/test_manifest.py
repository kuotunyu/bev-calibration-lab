"""Persist full provenance, and fail closed at the formal boundary."""

from __future__ import annotations

import json

import pytest


def test_round_trip_preserves_distinct_sensor_times_and_refuses_overwrite(tmp_path):
    from tests.unit.cohort.test_splits import freeze, population

    from bevcalib.cohort.manifest import load_manifest, save_manifest

    manifest = freeze(population())["development"]
    path = tmp_path / "manifest.json"
    save_manifest(manifest, path)
    assert b"\r" not in path.read_bytes()
    restored = load_manifest(path)
    assert restored == manifest
    assert restored.scenes[0].sample_timestamps == (100,)
    assert restored.scenes[0].camera_timestamps == (101,)
    assert restored.scenes[0].lidar_timestamps == (98,)
    with pytest.raises(FileExistsError):
        save_manifest(manifest, path)


@pytest.mark.parametrize(
    "field,value",
    [
        ("dataset_version", "v1.0-mini"),
        ("protocol_hash", "b" * 64),
        ("manifest_sha256", "0" * 64),
        ("role", "calibration"),
    ],
)
def test_schema_valid_tampering_is_rejected(tmp_path, field, value):
    from tests.unit.cohort.test_splits import freeze, population

    from bevcalib.cohort.manifest import load_manifest

    doc = freeze(population())["development"].model_dump(mode="json")
    doc[field] = value
    path = tmp_path / "tampered.json"
    path.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="hash"):
        load_manifest(path)


def test_legacy_is_readable_without_fabricated_provenance_but_still_hash_checked(tmp_path):
    from tests.unit.training.test_engine import rehash

    from bevcalib.cohort.manifest import CohortManifestV1, load_manifest

    doc = {"schema_version": "bev-calibration-cohort/v1", "role": "development", "scenes": []}
    rehash(doc)
    path = tmp_path / "legacy.json"
    path.write_text(json.dumps(doc))
    restored = load_manifest(path)
    assert isinstance(restored, CohortManifestV1)
    assert "dataset_version" not in restored.model_dump()
    doc["role"] = "calibration"
    path.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="hash"):
        load_manifest(path)


@pytest.mark.parametrize(
    "mutation,match",
    [
        ("duplicate", "duplicate scene"),
        ("locations", "every location"),
        ("requested", "requested count"),
        ("selected", "counts disagree"),
        ("availability", "counts disagree"),
        ("available_logs", "counts disagree"),
        ("missing_logs", "counts disagree"),
        ("quota_overflow", "counts disagree"),
        ("reason", "shortage reason"),
    ],
)
def test_rehashed_semantically_inconsistent_documents_are_refused(tmp_path, mutation, match):
    from tests.unit.training.test_engine import cohort_document, rehash

    from bevcalib.cohort.manifest import load_manifest

    doc = cohort_document("development", "synthetic", 100)
    if mutation == "duplicate":
        doc["scenes"].append(doc["scenes"][0])
    elif mutation == "locations":
        doc["allocation"].reverse()
    elif mutation == "requested":
        doc["allocation"][0]["requested"] = 99
    elif mutation == "selected":
        doc["allocation"][0]["selected"] = 99
    elif mutation == "availability":
        doc["allocation"][0]["available"] = 99
    elif mutation == "available_logs":
        doc["allocation"][0]["available_logs"] = 101
    elif mutation == "missing_logs":
        doc["allocation"][0]["available_logs"] = 0
    elif mutation == "quota_overflow":
        doc["allocation"][0]["requested"] = 99
        doc["allocation"][1]["requested"] = 1
    else:
        doc["allocation"][0]["shortage_reason"] = "joint_log_capacity"
    rehash(doc)
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match=match):
        load_manifest(path)


def test_shuffled_freezes_write_identical_bytes_and_hash_diagnostics(tmp_path):
    from tests.unit.cohort.test_splits import freeze, population

    from bevcalib.cohort.manifest import load_manifest, save_manifest

    one = freeze(population())
    two = freeze(population()[::-1])
    for role in one:
        left, right = tmp_path / f"{role}-a.json", tmp_path / f"{role}-b.json"
        save_manifest(one[role], left)
        save_manifest(two[role], right)
        assert left.read_bytes() == right.read_bytes()
    doc = json.loads(left.read_text())
    doc["allocation"][0]["available"] += 1
    left.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="hash"):
        load_manifest(left)


def test_saving_a_modified_model_cannot_bypass_hash_verification(tmp_path):
    from tests.unit.cohort.test_splits import freeze, population

    from bevcalib.cohort.manifest import save_manifest

    model = freeze(population())["development"].model_copy(update={"dataset_version": "v1.0-mini"})
    path = tmp_path / "not-written.json"
    with pytest.raises(ValueError, match="hash"):
        save_manifest(model, path)
    assert not path.exists()
