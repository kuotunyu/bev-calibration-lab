"""Verified streaming must retain late integrity failures and legacy reader parity."""

import hashlib
import json
from pathlib import Path

import pytest
from tests.unit.artifacts.test_result_documents import inputs, rows_for


def test_verified_scene_iterator_matches_legacy_reader(tmp_path: Path) -> None:
    from bevcalib.artifacts.result_documents import finalize_run, load_result_run, save_scene

    identity, manifest = inputs()
    rows = rows_for(manifest)
    save_scene(tmp_path, identity, manifest, rows)
    finalize_run(tmp_path, identity, manifest)
    from bevcalib.artifacts.result_documents import iter_result_scenes

    pieces = iter_result_scenes(tmp_path, identity, manifest)
    assert iter(pieces) is pieces
    assert list(pieces) == [rows]
    assert load_result_run(tmp_path, identity, manifest) == rows


def test_verified_scene_iterator_refuses_payload_hash_tamper(tmp_path: Path) -> None:
    from bevcalib.artifacts.result_documents import finalize_run, save_scene

    identity, manifest = inputs()
    path = save_scene(tmp_path, identity, manifest, rows_for(manifest))
    finalize_run(tmp_path, identity, manifest)
    path.write_bytes(path.read_bytes() + b" ")
    from bevcalib.artifacts.result_documents import iter_result_scenes

    with pytest.raises(ValueError, match="file hash"):
        list(iter_result_scenes(tmp_path, identity, manifest))


@pytest.mark.parametrize("case", ["identity", "marker_files"])
def test_scene_iterator_rechecks_expected_completion_identity(tmp_path: Path, case: str) -> None:
    from bevcalib.artifacts.result_documents import (
        digest,
        finalize_run,
        iter_result_scenes,
        save_scene,
    )

    identity, manifest = inputs()
    save_scene(tmp_path, identity, manifest, rows_for(manifest))
    marker = finalize_run(tmp_path, identity, manifest)
    if case == "identity":
        identity = identity.model_copy(update={"method": "classical"})
    else:
        body = json.loads(marker.read_bytes())
        body["files"] = {}
        body["document_sha256"] = digest({k: v for k, v in body.items() if k != "document_sha256"})
        marker.write_text(json.dumps(body), encoding="utf-8")
    with pytest.raises(ValueError, match=r"complete marker|file hash"):
        list(iter_result_scenes(tmp_path, identity, manifest))


@pytest.mark.parametrize("case", ["hash", "identity"])
def test_single_scene_read_rechecks_inner_document_contract(tmp_path: Path, case: str) -> None:
    from bevcalib.artifacts.result_documents import digest, load_result_scene, save_scene

    identity, manifest = inputs()
    path = save_scene(tmp_path, identity, manifest, rows_for(manifest))
    body = json.loads(path.read_bytes())
    if case == "hash":
        body["document_sha256"] = "f" * 64
    else:
        body["identity"]["method"] = "classical"
        body["document_sha256"] = digest({k: v for k, v in body.items() if k != "document_sha256"})
    path.write_text(json.dumps(body), encoding="utf-8")
    with pytest.raises(ValueError, match=r"document hash|scene identity"):
        load_result_scene(
            path,
            identity,
            manifest,
            manifest.scenes[0].scene_token,
            hashlib.sha256(path.read_bytes()).hexdigest(),
        )
