"""Paired scene means; the source loader is isolated from these arithmetic tests."""

import json
import subprocess
import weakref
from pathlib import Path
from types import SimpleNamespace

import pytest

from bevcalib.artifacts.envelope import verify_envelope


@pytest.fixture
def export_inputs(monkeypatch: pytest.MonkeyPatch):  # type: ignore[no-untyped-def]
    from bevcalib.artifacts import calibration_distribution as module

    def row(rotation, translation):  # type: ignore[no-untyped-def]
        return SimpleNamespace(
            fault_axis="yaw",
            fault_level=1.0,
            pose=SimpleNamespace(
                rotation_rpy_error_deg=rotation, translation_xyz_error_m=translation
            ),
        )

    rows = {
        "private-scene-a": [row((1, 2, 3), (0.1, 0.2, 0.3)), row((3, 4, 5), (0.3, 0.4, 0.5))],
        "private-scene-b": [row((-2, -4, -6), (-0.2, -0.4, -0.6))],
    }
    manifest = SimpleNamespace(
        protocol_hash="a" * 64,
        manifest_sha256="b" * 64,
        scenes=[SimpleNamespace(scene_token=token) for token in rows],
    )
    identity = SimpleNamespace(
        method="identity",
        seed=None,
        run_id="c" * 64,
        producer=SimpleNamespace(commit="d" * 40, lock_sha256="e" * 64),
    )
    run = SimpleNamespace(
        marker=SimpleNamespace(identity=identity),
        source_complete_sha256="f" * 64,
        scene_rows=lambda manifest, token: rows[token],
    )
    calls = []

    def source_loader(path, *, require_all_seeds):  # type: ignore[no-untyped-def]
        calls.append((path, require_all_seeds))
        return manifest, (run,)

    monkeypatch.setattr(module, "load_run_set", source_loader)
    monkeypatch.setattr(module, "_producer_commit", lambda: "1" * 40)
    return module, rows, calls


def export(module, tmp_path):  # type: ignore[no-untyped-def]
    return module.export_calibration_distribution(
        tmp_path / "inputs",
        tmp_path / "distribution.json",
        method="identity",
        seed=None,
        axis="yaw",
        level=1.0,
        producer_release="v1.0.0",
        created_at_utc="2026-09-10T00:00:00Z",
    )


def test_export_keeps_signed_paired_scene_means(export_inputs, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    module, rows, calls = export_inputs
    path = export(module, tmp_path)
    result = verify_envelope(path, "calibration-error-distribution/v1")
    assert result.producer_commit == "1" * 40
    assert result.protocol_hash == "a" * 64
    assert result.payload["rotation_samples_rpy_deg"] == [[2, 3, 4], [-2, -4, -6]]
    for actual, expected in zip(
        result.payload["translation_samples_xyz_m"],
        [[0.2, 0.3, 0.4], [-0.2, -0.4, -0.6]],
        strict=True,
    ):
        assert actual == pytest.approx(expected)
    assert result.payload["support"] == {
        "total_scenes": 2,
        "valid_scenes": 2,
        "excluded_scenes": 0,
        "total_frames": 3,
        "pose_valid_frames": 3,
        "excluded_frames": 0,
    }
    assert calls == [(tmp_path / "inputs", True)]
    assert all(token not in path.read_text() for token in rows)


def test_export_releases_raw_rows_before_next_scene(
    export_inputs, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:  # type: ignore[no-untyped-def]
    module, _, _ = export_inputs
    original_loader = module.load_run_set

    class Row:
        fault_axis = "yaw"
        fault_level = 1.0
        pose = SimpleNamespace(rotation_rpy_error_deg=(0, 0, 1), translation_xyz_error_m=(0, 0, 0))

    references: list[weakref.ReferenceType[Row]] = []

    def read(manifest, token):  # type: ignore[no-untyped-def]
        assert all(reference() is None for reference in references), "previous raw scene retained"
        row = Row()
        references.append(weakref.ref(row))
        return (row,)

    def load(path, **kwargs):  # type: ignore[no-untyped-def]
        manifest, runs = original_loader(path, **kwargs)
        runs[0].scene_rows = read
        return manifest, runs

    monkeypatch.setattr(module, "load_run_set", load)
    export(module, tmp_path)
    assert len(references) == 2
    assert all(reference() is None for reference in references)


@pytest.mark.parametrize(
    "case", ["empty", "rotation_bound", "translation_bound", "nonfinite", "wrong_length"]
)
def test_export_refuses_unusable_distribution_before_output(
    export_inputs, tmp_path: Path, case: str
) -> None:  # type: ignore[no-untyped-def]
    module, rows, _ = export_inputs
    pose = rows["private-scene-a"][0].pose
    if case == "empty":
        for group in rows.values():
            for row in group:
                row.pose = None
    elif case == "rotation_bound":
        pose.rotation_rpy_error_deg = (100, 0, 0)
    elif case == "translation_bound":
        pose.translation_xyz_error_m = (10, 0, 0)
    elif case == "nonfinite":
        pose.rotation_rpy_error_deg = (float("nan"), 0, 0)
    else:
        pose.rotation_rpy_error_deg = (1, 2)
    with pytest.raises(ValueError):
        export(module, tmp_path)
    assert not (tmp_path / "distribution.json").exists()


def test_export_preserves_unsupported_scene_and_frame_counts(export_inputs, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    module, rows, _ = export_inputs
    rows["private-scene-b"][0].pose = None
    result = verify_envelope(export(module, tmp_path), "calibration-error-distribution/v1")
    assert result.payload["support"] == {
        "total_scenes": 2,
        "valid_scenes": 1,
        "excluded_scenes": 1,
        "total_frames": 3,
        "pose_valid_frames": 2,
        "excluded_frames": 1,
    }


@pytest.mark.parametrize(
    "options",
    [
        {"axis": "time", "level": 50.0},
        {"axis": "yaw", "level": 99.0},
        {"method": "missing"},
        {"seed": 17},
        {"producer_release": "v1.0.0-rc1"},
        {"created_at_utc": "not UTC"},
    ],
)
def test_export_refuses_incompatible_request(export_inputs, tmp_path: Path, options: dict) -> None:  # type: ignore[no-untyped-def]
    module, _, _ = export_inputs
    args = {
        "method": "identity",
        "seed": None,
        "axis": "yaw",
        "level": 1.0,
        "producer_release": "v1.0.0",
        "created_at_utc": "2026-09-10T00:00:00Z",
    }
    args.update(options)
    with pytest.raises(ValueError):
        module.export_calibration_distribution(tmp_path / "inputs", tmp_path / "out.json", **args)
    assert not (tmp_path / "out.json").exists()


def test_export_does_not_replace_existing_output(export_inputs, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    module, _, _ = export_inputs
    path = tmp_path / "distribution.json"
    path.write_bytes(b"preserve")
    with pytest.raises(FileExistsError):
        export(module, tmp_path)
    assert path.read_bytes() == b"preserve"


def test_fixed_time_export_is_byte_reproducible(export_inputs, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    module, _, _ = export_inputs
    first = export(module, tmp_path / "first")
    second = export(module, tmp_path / "second")
    assert first.read_bytes() == second.read_bytes()


@pytest.mark.parametrize("frame_multiplier", [1, 29])
def test_export_accepts_exact_interchange_bounds(
    export_inputs, tmp_path: Path, frame_multiplier: int
) -> None:  # type: ignore[no-untyped-def]
    module, rows, _ = export_inputs
    for group in rows.values():
        for row in group:
            row.pose.rotation_rpy_error_deg = (30, -30, 30)
            row.pose.translation_xyz_error_m = (2, -2, 2)
        group *= frame_multiplier
    result = verify_envelope(export(module, tmp_path), "calibration-error-distribution/v1")
    assert result.payload["rotation_samples_rpy_deg"] == [[30, -30, 30]] * 2
    assert result.payload["translation_samples_xyz_m"] == [[2, -2, 2]] * 2


def test_later_scene_validation_failure_leaves_no_output(
    export_inputs, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:  # type: ignore[no-untyped-def]
    module, rows, _ = export_inputs
    original_loader = module.load_run_set

    def read(manifest, token):  # type: ignore[no-untyped-def]
        if token == "private-scene-b":
            raise ValueError("source scene digest differs")
        return rows[token]

    def load(path, **kwargs):  # type: ignore[no-untyped-def]
        manifest, runs = original_loader(path, **kwargs)
        runs[0].scene_rows = read
        return manifest, runs

    monkeypatch.setattr(module, "load_run_set", load)
    with pytest.raises(ValueError, match="source scene digest differs"):
        export(module, tmp_path)
    assert not (tmp_path / "distribution.json").exists()


@pytest.mark.parametrize("change", ["clean", "modified", "untracked", "ignored_source"])
def test_producer_identity_comes_from_committed_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, change: str
) -> None:
    from bevcalib.artifacts import calibration_distribution as module

    source = tmp_path / "src/bevcalib/artifacts/calibration_distribution.py"
    source.parent.mkdir(parents=True)
    source.write_text("# committed fixture\n")

    def git(*args):  # type: ignore[no-untyped-def]
        return subprocess.run(
            ["git", "-C", str(tmp_path), *args], check=True, capture_output=True, text=True
        ).stdout.strip()

    git("init")
    git("add", "src")
    git(
        "-c",
        "user.name=Fixture",
        "-c",
        "user.email=fixture@example.invalid",
        "commit",
        "-m",
        "fixture",
    )
    expected = git("rev-parse", "HEAD")
    monkeypatch.setattr(module, "__file__", str(source))
    if change == "modified":
        source.write_text("# different source\n")
    elif change == "untracked":
        (source.parent / "uncommitted.py").write_text("# new dependency\n")
    elif change == "ignored_source":
        git("rm", "--cached", "src/bevcalib/artifacts/calibration_distribution.py")
        (tmp_path / ".gitignore").write_text("*.py\n")
        git("add", ".gitignore")
        git(
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=fixture@example.invalid",
            "commit",
            "-m",
            "remove source",
        )
    if change == "clean":
        assert module._producer_commit() == expected
    else:
        with pytest.raises(ValueError, match="committed"):
            module._producer_commit()


def test_export_real_source_loader_refuses_nonformal_cohort(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tests.unit.training.test_engine import cohort_document

    from bevcalib.artifacts import calibration_distribution as module
    from bevcalib.cohort.manifest import CohortManifestV2, save_manifest

    root = tmp_path / "inputs"
    root.mkdir()
    manifest = CohortManifestV2.model_validate(cohort_document("evaluation", "eval", 2))
    save_manifest(manifest, root / "evaluation_manifest.json")
    monkeypatch.setattr(module, "_producer_commit", lambda: "1" * 40)
    with pytest.raises(ValueError, match="exactly 30 scenes"):
        export(module, tmp_path)
    assert not (tmp_path / "distribution.json").exists()


@pytest.mark.parametrize("damage", ["missing_manifest", "missing_scene", "protocol", "scene_hash"])
def test_export_real_loader_refuses_damaged_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, damage: str
) -> None:
    from tests.unit.artifacts.test_result_documents import inputs
    from tests.unit.training.test_engine import cohort_document

    from bevcalib.artifacts import calibration_distribution as module
    from bevcalib.artifacts.result_documents import digest, scene_filename
    from bevcalib.cohort.manifest import CohortManifestV2, save_manifest

    # Intentionally invalid on-disk fixtures exercise production refusal, not
    # successful observed evidence. No source loader or scene reader is mocked.
    monkeypatch.setattr(module, "_producer_commit", lambda: "1" * 40)
    root = tmp_path / "inputs"
    root.mkdir()
    if damage == "missing_manifest":
        with pytest.raises(FileNotFoundError):
            export(module, tmp_path)
        assert not (tmp_path / "distribution.json").exists()
        return
    manifest = CohortManifestV2.model_validate(cohort_document("evaluation", "eval", 30))
    save_manifest(manifest, root / "evaluation_manifest.json")
    original, _ = inputs()
    identity = original.model_dump(mode="json")
    identity.update(dataset_manifest_hash=manifest.manifest_sha256, evidence_type="observed")
    identity["measurements"]["images"] = {
        token: {"rgb_sha256": "b" * 64, "width": 6, "height": 4, "edge_threshold": 1.0}
        for scene in manifest.scenes
        for token in scene.camera_sample_data_tokens
    }
    if damage == "protocol":
        identity["protocol_hash"] = "0" * 64
    for method, seed, checkpoint in (
        ("identity", None, None),
        ("classical", None, None),
        ("learned", 17, "a" * 64),
        ("learned", 42, "b" * 64),
        ("learned", 73, "c" * 64),
    ):
        directory = root / f"{method}-{seed}"
        directory.mkdir()
        files = {scene_filename(scene.scene_token): "f" * 64 for scene in manifest.scenes}
        for name in files:
            (directory / name).write_text("{}", encoding="utf-8")
        body = {
            "schema_version": "bev-calibration-complete/v2",
            "identity": identity
            | {"method": method, "seed": seed, "checkpoint_sha256": checkpoint},
            "files": files,
        }
        (directory / "run_complete.json").write_text(
            json.dumps(body | {"document_sha256": digest(body)}), encoding="utf-8"
        )
        if damage == "missing_scene":
            (directory / next(iter(files))).unlink()
    message = {
        "missing_scene": "scene file inventory is missing",
        "protocol": "run identity differs from verified cohort",
        "scene_hash": "completed run file hash mismatch",
    }[damage]
    with pytest.raises(ValueError, match=message):
        export(module, tmp_path)
    assert not (tmp_path / "distribution.json").exists()
