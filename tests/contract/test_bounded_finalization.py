"""Finalization validates and releases one scene at a time."""

import json
import weakref
from pathlib import Path

import pytest
from tests.contract.test_formal_artifact_set import two_scene_fixture

from bevcalib.artifacts.results import CalibrationResultV2


def test_finalization_releases_each_scene_and_preserves_completion_bytes(tmp_path, monkeypatch):
    from bevcalib.artifacts import result_documents as documents

    root, manifest = two_scene_fixture(tmp_path)
    directory = next(root.glob("identity-*"))
    marker = directory / "run_complete.json"
    expected = marker.read_bytes()
    identity = documents.RunCompleteV2.model_validate_json(expected).identity
    marker.unlink()
    references: list[weakref.ReferenceType[CalibrationResultV2]] = []
    checked = documents._checked_body
    load_scene = documents.load_result_scene
    validate = documents._validate_rows
    calls = []

    def observe_read(path):
        assert all(ref() is None for ref in references), (
            "prior scene rows are retained during finalization"
        )
        calls.append(path)
        return checked(path)

    def observe_validation(rows, *args):
        validate(rows, *args)
        references[:] = [weakref.ref(row) for row in rows]

    def observe_scene(path, *args):
        assert all(ref() is None for ref in references), (
            "prior scene rows are retained during finalization"
        )
        calls.append(path)
        return load_scene(path, *args)

    monkeypatch.setattr(documents, "_checked_body", observe_read)
    monkeypatch.setattr(documents, "_validate_rows", observe_validation)
    monkeypatch.setattr(documents, "load_result_scene", observe_scene)
    documents.finalize_run(directory, identity, manifest)
    assert len(calls) == 2
    assert all(ref() is None for ref in references)
    assert marker.read_bytes() == expected


def test_finalization_refuses_late_invalid_scene_before_completion(tmp_path):
    from bevcalib.artifacts import result_documents as documents

    root, manifest = two_scene_fixture(tmp_path)
    directory = next(root.glob("identity-*"))
    marker = directory / "run_complete.json"
    identity = documents.RunCompleteV2.model_validate_json(marker.read_bytes()).identity
    marker.unlink()
    path = directory / documents.scene_filename(manifest.scenes[-1].scene_token)
    body = json.loads(path.read_bytes())
    body["rows"].pop()
    body["document_sha256"] = documents.digest(
        {k: v for k, v in body.items() if k != "document_sha256"}
    )
    path.write_text(json.dumps(body), encoding="utf-8")
    with pytest.raises(ValueError, match="inventory"):
        documents.finalize_run(directory, identity, manifest)
    assert not marker.exists()


def test_finalization_binds_hash_to_validated_scene_bytes(tmp_path, monkeypatch):
    from bevcalib.artifacts import result_documents as documents

    root, manifest = two_scene_fixture(tmp_path)
    directory = next(root.glob("identity-*"))
    marker = directory / "run_complete.json"
    identity = documents.RunCompleteV2.model_validate_json(marker.read_bytes()).identity
    marker.unlink()
    original = Path.read_bytes
    counts: dict[Path, int] = {}

    def changing_read(path):
        payload = original(path)
        counts[path] = counts.get(path, 0) + 1
        if path.parent == directory and counts[path] == 2:
            return payload + b"\n"
        return payload

    monkeypatch.setattr(Path, "read_bytes", changing_read)
    with pytest.raises(ValueError, match="file hash mismatch"):
        documents.finalize_run(directory, identity, manifest)
    assert not marker.exists()


def test_completed_run_refuses_tampered_marker_hash(tmp_path):
    from bevcalib.artifacts import result_documents as documents

    root, manifest = two_scene_fixture(tmp_path)
    directory = next(root.glob("identity-*"))
    marker = directory / "run_complete.json"
    body = json.loads(marker.read_bytes())
    identity = documents.RunCompleteV2.model_validate(body).identity
    body["document_sha256"] = "0" * 64
    marker.write_text(json.dumps(body), encoding="utf-8")
    with pytest.raises(ValueError, match="document hash"):
        documents.load_result_run(directory, identity, manifest)
