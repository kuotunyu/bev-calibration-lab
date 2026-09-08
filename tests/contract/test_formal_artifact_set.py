"""Five formal documents require complete compatible source runs and exact set identity."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from tests.unit.metrics.test_summary import run_fixture

from bevcalib.cohort.manifest import CohortManifestV2

RUNS = (
    ("identity", None, None),
    ("classical", None, None),
    ("learned", 17, "a" * 64),
    ("learned", 42, "b" * 64),
    ("learned", 73, "c" * 64),
)


def two_scene_fixture(tmp_path: Path) -> tuple[Path, CohortManifestV2]:
    from tests.unit.artifacts.test_result_documents import inputs, rows_for
    from tests.unit.training.test_engine import cohort_document

    from bevcalib.artifacts.result_documents import EvaluationIdentity, finalize_run, save_scene
    from bevcalib.cohort.manifest import CohortManifestV2, save_manifest

    original, _ = inputs()
    manifest = CohortManifestV2.model_validate(cohort_document("evaluation", "eval", 2))
    identity = original.model_dump(mode="json")
    identity["dataset_manifest_hash"] = manifest.manifest_sha256
    identity["measurements"]["images"] = {
        token: {"rgb_sha256": "b" * 64, "width": 6, "height": 4, "edge_threshold": 1.0}
        for scene in manifest.scenes
        for token in scene.camera_sample_data_tokens
    }
    root = tmp_path / "private"
    root.mkdir()
    save_manifest(manifest, root / "evaluation_manifest.json")
    for method, seed, checkpoint in RUNS:
        run = EvaluationIdentity.model_validate(
            identity | {"method": method, "seed": seed, "checkpoint_sha256": checkpoint}
        )
        directory = root / f"{method}-{run.run_id}"
        for scene in manifest.scenes:
            rows = rows_for(manifest.model_copy(update={"scenes": (scene,)}))
            save_scene(
                directory,
                run,
                manifest,
                tuple(row for row in rows if method == "identity" or row.fault_axis != "time"),
            )
        finalize_run(directory, run, manifest)
    return root, manifest


def test_aggregation_releases_raw_scene_rows_before_reading_another_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import weakref

    from bevcalib.analysis.aggregate import aggregate_formal_results
    from bevcalib.artifacts.result_sets import VerifiedRun

    root, manifest = two_scene_fixture(tmp_path)
    original = VerifiedRun.scene_rows
    references: list[weakref.ReferenceType] = []
    calls = []

    def read(run, cohort, scene):
        assert all(reference() is None for reference in references), "previous raw V2 rows retained"
        rows = original(run, cohort, scene)
        references[:] = [weakref.ref(row) for row in rows]
        calls.append((scene, run.label))
        return rows

    monkeypatch.setattr(VerifiedRun, "scene_rows", read)
    aggregate_formal_results(root, tmp_path / "formal", synthetic_fixture=True)
    assert len(calls) == 10
    assert {scene for scene, _ in calls[:5]} == {manifest.scenes[0].scene_token}
    assert all(reference() is None for reference in references)


def test_late_corrupt_scene_refuses_after_earlier_reads_without_publishing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from bevcalib.analysis.aggregate import aggregate_formal_results
    from bevcalib.artifacts.result_documents import scene_filename
    from bevcalib.artifacts.result_sets import VerifiedRun

    root, manifest = two_scene_fixture(tmp_path)
    path = next(root.glob("identity-*")) / scene_filename(manifest.scenes[1].scene_token)
    path.write_bytes(path.read_bytes() + b" ")
    original = VerifiedRun.scene_rows
    calls = []

    def read(run, cohort, scene):
        calls.append(scene)
        return original(run, cohort, scene)

    monkeypatch.setattr(VerifiedRun, "scene_rows", read)
    with pytest.raises(ValueError, match="file hash"):
        aggregate_formal_results(root, tmp_path / "absent", synthetic_fixture=True)
    assert calls.count(manifest.scenes[0].scene_token) == 5
    assert not (tmp_path / "absent").exists()


def test_five_formal_document_schemas_are_registered() -> None:
    from bevcalib.artifacts.schemas import contract_schema_documents

    assert {
        f"formal_{name}_v1.json"
        for name in ("metrics", "intervals", "recovery", "timing", "exclusions")
    } <= contract_schema_documents().keys()


@pytest.mark.parametrize("case", ["marker_hash", "scene_file"])
def test_source_headers_reject_invalid_hash_or_missing_scene(tmp_path: Path, case: str) -> None:
    from bevcalib.analysis.aggregate import aggregate_formal_results

    root, _ = run_fixture(tmp_path, RUNS)
    if case == "marker_hash":
        path = next(root.glob("identity-*/run_complete.json"))
        body = json.loads(path.read_bytes())
        body["document_sha256"] = "f" * 64
        path.write_text(json.dumps(body), encoding="utf-8")
    else:
        next(root.glob("identity-*/scene-*.json")).unlink()
    with pytest.raises(ValueError, match=r"hash|inventory"):
        aggregate_formal_results(root, tmp_path / "absent", synthetic_fixture=True)
    assert not (tmp_path / "absent").exists()


def test_existing_output_and_missing_formal_documents_are_refused(tmp_path: Path) -> None:
    from bevcalib.analysis.aggregate import aggregate_formal_results
    from bevcalib.artifacts.documents import load_formal_artifact_set

    output = tmp_path / "existing"
    output.mkdir()
    with pytest.raises(FileExistsError, match="overwrite"):
        aggregate_formal_results(tmp_path / "missing", output)
    with pytest.raises(ValueError, match="file inventory"):
        load_formal_artifact_set(output)


def rewrite_scene(root: Path, change) -> None:
    from bevcalib.artifacts.result_documents import digest

    path = next(root.glob("classical-*/scene-*.json"))
    body = json.loads(path.read_bytes())
    change(body)
    body["document_sha256"] = digest({k: v for k, v in body.items() if k != "document_sha256"})
    path.write_text(json.dumps(body), encoding="utf-8")
    marker_path = path.parent / "run_complete.json"
    marker = json.loads(marker_path.read_bytes())
    marker["files"][path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    marker["document_sha256"] = digest({k: v for k, v in marker.items() if k != "document_sha256"})
    marker_path.write_text(json.dumps(marker), encoding="utf-8")


@pytest.mark.parametrize("change", ["method", "condition", "scene"])
def test_incomplete_formal_source_set_refused_before_any_output(
    tmp_path: Path, change: str
) -> None:
    root, _ = run_fixture(tmp_path, RUNS[:-1] if change == "method" else RUNS)
    if change == "condition":
        rewrite_scene(root, lambda body: body["rows"].pop())
    if change == "scene":
        rewrite_scene(root, lambda body: body["rows"][0].update(scene_token="other-scene"))
    from bevcalib.analysis.aggregate import aggregate_formal_results

    output = tmp_path / "formal"
    with pytest.raises(ValueError, match=r"inventory|provenance"):
        aggregate_formal_results(root, output, synthetic_fixture=True)
    assert not output.exists()


def test_five_documents_are_typed_bound_and_deterministic(tmp_path: Path) -> None:
    root, manifest = run_fixture(tmp_path, RUNS)
    from bevcalib.analysis.aggregate import aggregate_formal_results
    from bevcalib.artifacts.documents import load_formal_artifact_set

    first, second = tmp_path / "one", tmp_path / "two"
    result = aggregate_formal_results(root, first, synthetic_fixture=True)
    aggregate_formal_results(root, second, synthetic_fixture=True)
    assert {p.name for p in first.iterdir()} == {
        "metrics.json",
        "intervals.json",
        "recovery.json",
        "timing.json",
        "exclusions.json",
    }
    assert all(p.read_bytes() == (second / p.name).read_bytes() for p in first.iterdir())
    assert load_formal_artifact_set(first) == result
    assert result.metrics.identity.dataset_manifest_hash == manifest.manifest_sha256
    assert set(result.metrics.runs) == {
        "identity",
        "classical",
        "learned-17",
        "learned-42",
        "learned-73",
    }
    assert len(result.metrics.runs["identity"]) == 67
    assert len(result.metrics.runs["learned-17"]) == 60
    yaw = result.metrics.runs["identity"]["yaw:0"]
    assert yaw["rotation_bias_roll_deg"].value == -1
    assert yaw["translation_bias_x_cm"].value == -10
    assert yaw["pixel_frame_p50_px"].value == 1.5
    assert yaw["bev_frame_mean_m/80+"].value is None
    assert yaw["bev_frame_mean_m/80+"].reason == "no_operator_valid_frames"
    assert "classical->learned-42" in result.intervals.comparisons
    serialized = "".join(p.read_text(encoding="utf-8") for p in first.iterdir())
    assert manifest.scenes[0].scene_token not in serialized
    assert all(token not in serialized for token in manifest.scenes[0].sample_tokens)


def test_five_document_loader_refuses_tamper_and_cross_file_identity(tmp_path: Path) -> None:
    root, _ = run_fixture(tmp_path, RUNS)
    from bevcalib.analysis.aggregate import aggregate_formal_results
    from bevcalib.artifacts.documents import load_formal_artifact_set
    from bevcalib.artifacts.result_documents import digest

    output = tmp_path / "formal"
    aggregate_formal_results(root, output, synthetic_fixture=True)
    path = output / "timing.json"
    body = json.loads(path.read_bytes())
    body["identity"]["protocol_hash"] = "d" * 64
    path.write_text(json.dumps(body), encoding="utf-8")
    with pytest.raises(ValueError, match="hash"):
        load_formal_artifact_set(output)
    body["document_sha256"] = digest({k: v for k, v in body.items() if k != "document_sha256"})
    path.write_text(json.dumps(body), encoding="utf-8")
    with pytest.raises(ValueError, match="identity"):
        load_formal_artifact_set(output)
