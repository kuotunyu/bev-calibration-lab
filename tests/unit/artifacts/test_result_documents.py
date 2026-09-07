"""Scene writes and completion require the full expected run inventory."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from tests.unit.artifacts.test_results_v2 import row_document
from tests.unit.training.test_engine import cohort_document


def inputs():  # type: ignore[no-untyped-def]
    from bevcalib.artifacts.result_documents import EvaluationIdentity
    from bevcalib.cohort.manifest import CohortManifestV2

    manifest = CohortManifestV2.model_validate(cohort_document("evaluation", "eval", 1))
    identity = EvaluationIdentity(
        method="identity",
        seed=None,
        checkpoint_sha256=None,
        protocol_hash=manifest.protocol_hash,
        dataset_manifest_hash=manifest.manifest_sha256,
        dataset_version=manifest.dataset_version,
        evidence_type="synthetic",
        producer={
            "commit": "a" * 40,
            "lock_sha256": "b" * 64,
            "hardware": {"runtime": "synthetic"},
        },
    )
    return identity, manifest


def rows_for(manifest):  # type: ignore[no-untyped-def]
    from bevcalib.artifacts.result_documents import fault_for_condition
    from bevcalib.artifacts.results import CalibrationResultV2
    from bevcalib.perturbations.schedule import formal_single_axis_faults

    rows = []
    scene = manifest.scenes[0]
    for index, sample in enumerate(scene.sample_tokens):
        for axis, level in formal_single_axis_faults():
            doc = row_document()
            doc.update(
                scene_token=scene.scene_token,
                sample_token=sample,
                fault_axis=axis,
                fault_level=level,
                fault=fault_for_condition(axis, level).model_dump(),
                camera={
                    "token": scene.camera_sample_data_tokens[index],
                    "timestamp_us": scene.camera_timestamps[index],
                    "channel": "CAM_FRONT",
                },
                lidar={
                    "token": scene.lidar_sample_data_tokens[index],
                    "timestamp_us": scene.lidar_timestamps[index],
                    "channel": "LIDAR_TOP",
                },
                timing={
                    "requested_offset_ms": int(level) if axis == "time" else 0,
                    "realized_offset_ms": -0.003,
                    "absolute_error_ms": abs(-0.003 - level) if axis == "time" else 0.003,
                    "reason": ("valid" if abs(-0.003 - level) <= 25 else "outside_tolerance")
                    if axis == "time"
                    else "nominal_pair",
                },
            )
            if axis == "time" and level != 0:
                doc.update(valid=False, invalid_reason="outside_tolerance")
            rows.append(CalibrationResultV2.model_validate(doc))
    return tuple(rows)


def test_atomic_scene_retains_all_samples_faults_then_completes(tmp_path: Path) -> None:
    from bevcalib.artifacts.result_documents import finalize_run, load_result_run, save_scene

    identity, manifest = inputs()
    rows = rows_for(manifest)
    scene_path = save_scene(tmp_path, identity, manifest, rows)
    assert len(json.loads(scene_path.read_text())["rows"]) == 134
    assert not (tmp_path / "run_complete.json").exists()
    finalize_run(tmp_path, identity, manifest)
    assert load_result_run(tmp_path, identity, manifest) == rows
    with pytest.raises(FileExistsError):
        save_scene(tmp_path, identity, manifest, rows)


@pytest.mark.parametrize("change", [lambda rows: rows[:-1], lambda rows: rows + rows[:1]])
def test_incomplete_or_duplicate_rows_do_not_create_any_file(tmp_path: Path, change) -> None:  # type: ignore[no-untyped-def]
    from bevcalib.artifacts.result_documents import save_scene

    identity, manifest = inputs()
    with pytest.raises(ValueError, match="inventory"):
        save_scene(tmp_path, identity, manifest, change(rows_for(manifest)))
    assert not list(tmp_path.iterdir())


def test_missing_scene_or_tampered_schema_valid_metric_refuses_completion(tmp_path: Path) -> None:
    from bevcalib.artifacts.result_documents import finalize_run, save_scene

    identity, manifest = inputs()
    with pytest.raises(ValueError, match="inventory"):
        finalize_run(tmp_path, identity, manifest)
    path = save_scene(tmp_path, identity, manifest, rows_for(manifest))
    document = json.loads(path.read_text())
    document["rows"][0]["edge_alignment_score"] = -9
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="hash"):
        finalize_run(tmp_path, identity, manifest)
    assert not (tmp_path / "run_complete.json").exists()


def test_completed_run_rejects_changed_bytes_identity_and_legacy(tmp_path: Path) -> None:
    from bevcalib.artifacts.result_documents import finalize_run, load_result_run, save_scene

    identity, manifest = inputs()
    path = save_scene(tmp_path, identity, manifest, rows_for(manifest))
    finalize_run(tmp_path, identity, manifest)
    wrong = identity.model_copy(update={"protocol_hash": "c" * 64})
    with pytest.raises(ValueError, match="identity"):
        load_result_run(tmp_path, wrong, manifest)
    path.write_text(path.read_text() + " ", encoding="utf-8")
    with pytest.raises(ValueError, match="hash"):
        load_result_run(tmp_path, identity, manifest)
    (tmp_path / "legacy.json").write_text('{"schema_version":"bev-calibration-result/v1"}')
    with pytest.raises(ValueError, match="inventory"):
        load_result_run(tmp_path, identity, manifest)


@pytest.mark.parametrize("seed", [17, 42, 73])
def test_learned_seeds_and_checkpoints_have_distinct_bound_run_ids(seed: int) -> None:
    from bevcalib.artifacts.result_documents import EvaluationIdentity

    identity, _ = inputs()
    learned = EvaluationIdentity.model_validate(
        identity.model_dump()
        | {"method": "learned", "seed": seed, "checkpoint_sha256": str(seed)[0] * 64}
    )
    assert learned.run_id != identity.run_id
    changed = learned.model_copy(update={"checkpoint_sha256": "f" * 64})
    assert learned.run_id != changed.run_id


@pytest.mark.parametrize(
    "mutation",
    [
        {"method": "learned"},
        {"seed": 17},
        {"checkpoint_sha256": "c" * 64},
        {"method": "learned", "seed": 12, "checkpoint_sha256": "c" * 64},
    ],
)
def test_missing_or_wrong_seed_checkpoint_identity_is_refused(mutation: dict) -> None:
    from bevcalib.artifacts.result_documents import EvaluationIdentity

    identity, _ = inputs()
    with pytest.raises(ValueError):
        EvaluationIdentity.model_validate(identity.model_dump() | mutation)


def test_observed_result_rejects_underfilled_synthetic_cohort(tmp_path: Path) -> None:
    from bevcalib.artifacts.result_documents import save_scene

    identity, manifest = inputs()
    observed = identity.model_copy(update={"evidence_type": "observed"})
    with pytest.raises(ValueError, match="formal evaluation"):
        save_scene(tmp_path, observed, manifest, rows_for(manifest))


def test_v2_contract_schemas_are_registered() -> None:
    from bevcalib.artifacts.schemas import contract_schema_documents

    schemas = contract_schema_documents()
    assert {
        "calibration_result_v2.json",
        "scene_result_v2.json",
        "run_complete_v2.json",
    } <= schemas.keys()


@pytest.mark.parametrize("case", ["empty", "scene", "camera", "lidar", "identity"])
def test_wrong_row_provenance_never_reaches_disk(tmp_path: Path, case: str) -> None:
    from bevcalib.artifacts.result_documents import save_scene

    identity, manifest = inputs()
    rows = list(rows_for(manifest))
    if case == "empty":
        rows = []
    elif case == "scene":
        rows[0] = rows[0].model_copy(update={"scene_token": "unknown"})
    elif case == "camera":
        rows[0] = rows[0].model_copy(
            update={"camera": rows[0].camera.model_copy(update={"token": "other"})}
        )
    elif case == "lidar":
        rows[0] = rows[0].model_copy(
            update={"lidar": rows[0].lidar.model_copy(update={"token": "other"})}
        )
    else:
        identity = identity.model_copy(update={"dataset_manifest_hash": "c" * 64})
    with pytest.raises(ValueError):
        save_scene(tmp_path, identity, manifest, tuple(rows))
    assert not list(tmp_path.iterdir())


def test_scene_from_another_method_cannot_finalize_same_run(tmp_path: Path) -> None:
    from bevcalib.artifacts.result_documents import finalize_run, save_scene

    identity, manifest = inputs()
    save_scene(tmp_path, identity, manifest, rows_for(manifest))
    with pytest.raises(ValueError, match="identity"):
        finalize_run(tmp_path, identity.model_copy(update={"method": "classical"}), manifest)


def test_unknown_formal_fault_is_not_silently_rounded() -> None:
    from bevcalib.artifacts.result_documents import fault_for_condition

    with pytest.raises(ValueError, match="inventory"):
        fault_for_condition("time", 100.5)
