"""Safe aggregate export from complete, compatible private V2 runs."""

import json
from pathlib import Path

import pytest
from tests.unit.artifacts.test_result_documents import inputs, rows_for


def run_fixture(
    tmp_path: Path, specifications=(("identity", None, None), ("classical", None, None))
):
    from bevcalib.artifacts.result_documents import EvaluationIdentity, finalize_run, save_scene
    from bevcalib.cohort.manifest import save_manifest

    identity, manifest = inputs()
    root = tmp_path / "private"
    root.mkdir()
    save_manifest(manifest, root / "evaluation_manifest.json")
    for method, seed, checkpoint in specifications:
        run_identity = EvaluationIdentity.model_validate(
            identity.model_dump()
            | {"method": method, "seed": seed, "checkpoint_sha256": checkpoint}
        )
        directory = root / f"{method}-{run_identity.run_id}"
        rows = tuple(
            row for row in rows_for(manifest) if method == "identity" or row.fault_axis != "time"
        )
        save_scene(directory, run_identity, manifest, rows)
        finalize_run(directory, run_identity, manifest)
    return root, manifest


def test_summary_preserves_signed_measurements_ranges_nulls_and_validity(tmp_path: Path) -> None:
    from bevcalib.metrics.summary import summarize_result_runs

    root, manifest = run_fixture(tmp_path)
    summary = summarize_result_runs(root, synthetic_fixture=True)
    assert summary["protocol_hash"] == manifest.protocol_hash
    assert summary["dataset_manifest_hash"] == manifest.manifest_sha256
    assert set(summary["runs"]) == {"identity", "classical"}
    assert len(summary["runs"]["identity"]["conditions"]) == 67
    assert len(summary["runs"]["classical"]["conditions"]) == 60
    condition = summary["runs"]["identity"]["conditions"]["yaw:0"]
    assert condition["edge_alignment_score"]["mean"] < 0
    assert condition["pose"]["rotation_rpy_error_deg"][0]["mean"] < 0
    assert set(condition["ground_contact_by_range"]) == {"0-10", "10-20", "20-40", "40-80", "80+"}
    assert condition["ground_contact_by_range"]["80+"]["mean"] is None
    assert condition["validity"]["total"] == 2
    serialized = json.dumps(summary)
    for scene in manifest.scenes:
        assert scene.scene_token not in serialized and scene.log_token not in serialized
        assert all(
            token not in serialized
            for token in scene.sample_tokens
            + scene.camera_sample_data_tokens
            + scene.lidar_sample_data_tokens
        )


@pytest.mark.parametrize(
    "specs",
    [
        (("identity", None, None),),
        (("identity", None, None), ("classical", None, None), ("learned", 17, "a" * 64)),
        (
            ("identity", None, None),
            ("classical", None, None),
            ("learned", 17, "a" * 64),
            ("learned", 42, "a" * 64),
            ("learned", 73, "c" * 64),
        ),
    ],
)
def test_missing_or_reused_learned_run_identity_is_refused(tmp_path: Path, specs) -> None:
    from bevcalib.metrics.summary import summarize_result_runs

    root, _ = run_fixture(tmp_path, specs)
    with pytest.raises(ValueError, match=r"run inventory|checkpoint"):
        summarize_result_runs(root, synthetic_fixture=True)


def test_partial_or_tampered_input_cannot_produce_a_summary(tmp_path: Path) -> None:
    from bevcalib.metrics.summary import summarize_result_runs

    root, _ = run_fixture(tmp_path)
    next(root.glob("classical-*/run_complete.json")).unlink()
    with pytest.raises(FileNotFoundError):
        summarize_result_runs(root, synthetic_fixture=True)


def test_three_distinct_learned_runs_keep_separate_denominators(tmp_path: Path) -> None:
    from bevcalib.metrics.summary import summarize_result_runs

    specs = (
        ("identity", None, None),
        ("classical", None, None),
        ("learned", 17, "a" * 64),
        ("learned", 42, "b" * 64),
        ("learned", 73, "c" * 64),
    )
    root, _ = run_fixture(tmp_path, specs)
    summary = summarize_result_runs(root, synthetic_fixture=True)
    assert set(summary["runs"]) == {
        "identity",
        "classical",
        "learned-17",
        "learned-42",
        "learned-73",
    }
    assert all(
        run["conditions"]["yaw:0"]["validity"]["total"] == 2 for run in summary["runs"].values()
    )
    with pytest.raises(ValueError, match="exactly 30"):
        summarize_result_runs(root)


def test_duplicate_run_directories_cannot_inflate_measurement_counts(tmp_path: Path) -> None:
    import shutil

    from bevcalib.metrics.summary import summarize_result_runs

    root, _ = run_fixture(tmp_path)
    shutil.copytree(next(root.glob("identity-*")), root / "duplicate")
    with pytest.raises(ValueError, match="run inventory"):
        summarize_result_runs(root, synthetic_fixture=True)


@pytest.mark.parametrize("change", ["measurement", "evidence"])
def test_schema_valid_runs_with_different_measurement_identity_are_refused(
    tmp_path: Path, change: str
) -> None:
    from bevcalib.artifacts.result_documents import digest
    from bevcalib.metrics.summary import summarize_result_runs

    root, _ = run_fixture(tmp_path)
    path = next(root.glob("classical-*/run_complete.json"))
    body = json.loads(path.read_text(encoding="utf-8"))
    if change == "measurement":
        first = next(iter(body["identity"]["measurements"]["images"].values()))
        first["edge_threshold"] += 1
    else:
        body["identity"]["evidence_type"] = "observed"
    if change == "measurement":
        import hashlib

        for scene_path in path.parent.glob("scene-*.json"):
            scene_doc = json.loads(scene_path.read_text(encoding="utf-8"))
            scene_doc["identity"] = body["identity"]
            scene_doc["document_sha256"] = digest(
                {key: value for key, value in scene_doc.items() if key != "document_sha256"}
            )
            scene_path.write_text(json.dumps(scene_doc), encoding="utf-8")
            body["files"][scene_path.name] = hashlib.sha256(scene_path.read_bytes()).hexdigest()
    body["document_sha256"] = digest(
        {key: value for key, value in body.items() if key != "document_sha256"}
    )
    path.write_text(json.dumps(body), encoding="utf-8")
    with pytest.raises(ValueError, match="measurement identity or evidence"):
        summarize_result_runs(root, synthetic_fixture=True)


def test_legacy_or_wrong_role_manifest_is_refused(tmp_path: Path) -> None:
    from tests.unit.training.test_engine import cohort_document

    from bevcalib.metrics.summary import summarize_result_runs

    root, _ = run_fixture(tmp_path)
    (root / "evaluation_manifest.json").write_text(
        json.dumps(cohort_document("development", "development", 1)), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="V2 evaluation"):
        summarize_result_runs(root, synthetic_fixture=True)


def test_unmeasured_operator_denominators_and_eighty_boundary_are_honest() -> None:
    from tests.unit.artifacts.test_results_v2 import row_document

    from bevcalib.artifacts.results import CalibrationResultV2
    from bevcalib.metrics.summary import summarize_condition

    body = row_document()
    body.update(
        valid=False,
        invalid_reason="no_image_edges",
        pose=None,
        pixel_errors_px=[],
        edge_alignment_score=None,
        ground_contacts=[
            {"box_token": "exact", "range_m": 80.0, "error_m": 2.0, "invalid_reason": None},
            {
                "box_token": "beyond",
                "range_m": 80.001,
                "error_m": None,
                "invalid_reason": "range_cutoff",
            },
        ],
    )
    summary = summarize_condition([CalibrationResultV2.model_validate(body)])
    assert summary["pose_recovery"] == {"valid": 0, "recovered": 0, "rate": None}
    assert summary["pixel_error_px"]["mean"] is None
    assert summary["validity"]["invalid_rate"] == 1.0
    assert summary["validity"]["reasons"] == {"no_image_edges": 1}
    far = summary["ground_contact_by_range"]["80+"]
    assert far["total"] == 2 and far["count"] == 1 and far["invalid"] == 1 and far["mean"] == 2.0
    assert far["reasons"] == {"range_cutoff": 1}
