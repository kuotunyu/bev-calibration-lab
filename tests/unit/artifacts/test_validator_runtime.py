"""Runtime identity uses actual source bytes, not a matching HEAD alone."""

import hashlib
import importlib
import json
import subprocess
from pathlib import Path

import pytest
from pydantic import ValidationError


@pytest.fixture
def checkout(tmp_path: Path) -> Path:
    for name, data in {
        "src/bevcalib/artifacts/validator_runtime.py": b"# validator\n",
        "configs/protocol.yaml": b"fixed: true\n",
        "schemas/study.json": b"{}\n",
        "pyproject.toml": b"[project]\n",
        "uv.lock": b"version = 1\n",
    }.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    return tmp_path


def module():
    return importlib.import_module("bevcalib.artifacts.validator_runtime")


def setup_runtime(monkeypatch, checkout):
    api = module()
    monkeypatch.setattr(
        api, "__file__", str(checkout / "src/bevcalib/artifacts/validator_runtime.py")
    )
    monkeypatch.setattr(api.platform, "python_version", lambda: "3.12.13")

    def git(command, **kwargs):
        assert command == ["git", "-C", str(checkout), "rev-parse", "--show-toplevel", "HEAD"]
        assert kwargs == {"check": True, "capture_output": True, "text": True, "encoding": "utf-8"}
        return subprocess.CompletedProcess(command, 0, f"{checkout.as_posix()}\n{'a' * 40}\n")

    monkeypatch.setattr(api.subprocess, "run", git)
    expected = api.ExpectedValidator(
        schema_version="bev-validator-runtime-expectations/v1",
        python_version="3.12.13",
        commit="a" * 40,
        lock_sha256=hashlib.sha256((checkout / "uv.lock").read_bytes()).hexdigest(),
        source_sha256=api.source_snapshot(checkout)["sha256"],
    )
    return api, expected


def test_snapshot_canonical_inventory_and_byte_changes(checkout):
    api = module()
    snapshot = api.source_snapshot(checkout)
    files = {
        path.relative_to(checkout).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in checkout.rglob("*")
        if path.is_file()
    }
    assert snapshot["files"] == files
    assert (
        snapshot["sha256"]
        == hashlib.sha256(
            json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
    )
    for name in ("docs/note.md", "src/bevcalib/__pycache__/cache.pyc"):
        path = checkout / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("excluded")
    assert api.source_snapshot(checkout) == snapshot
    (checkout / "configs/protocol.yaml").write_text("fixed: false\n")
    assert api.source_snapshot(checkout)["sha256"] != snapshot["sha256"]


@pytest.mark.parametrize("name", ["uv.lock", "pyproject.toml"])
def test_required_files_missing(checkout, name):
    (checkout / name).unlink()
    with pytest.raises(ValueError, match="required source file"):
        module().source_snapshot(checkout)


def test_source_directory_missing(tmp_path):
    with pytest.raises(ValueError, match="source directory"):
        module().source_snapshot(tmp_path)


def test_source_link_refused(checkout, monkeypatch):
    original = Path.is_symlink
    monkeypatch.setattr(
        Path, "is_symlink", lambda path: path.name == "protocol.yaml" or original(path)
    )
    with pytest.raises(ValueError, match="source symlink"):
        module().source_snapshot(checkout)


def test_scoped_root_symlink_ancestor_refused_before_traversal(checkout, monkeypatch):
    original = Path.is_symlink
    monkeypatch.setattr(Path, "is_symlink", lambda path: path.name == ".agents" or original(path))
    with pytest.raises(ValueError, match="source symlink"):
        module().source_snapshot(checkout)


@pytest.mark.parametrize("name", [".agents", "protocol.yaml"])
def test_windows_junctions_refused(checkout, monkeypatch, name):
    monkeypatch.setattr(Path, "is_junction", lambda path: path.name == name)
    with pytest.raises(ValueError, match="source symlink or junction"):
        module().source_snapshot(checkout)


def test_runtime_reads_actual_interpreter_and_snapshot(checkout, monkeypatch):
    api, expected = setup_runtime(monkeypatch, checkout)
    receipt = api.validate_validator_runtime(expected)
    assert receipt["python_version"] == "3.12.13"
    assert receipt["commit"] == "a" * 40
    assert receipt["source_sha256"] == expected.source_sha256
    assert receipt["lock_sha256"] == expected.lock_sha256
    assert receipt["executable"] == str(Path(api.sys.executable).resolve())


@pytest.mark.parametrize(
    "field,value",
    [
        ("python_version", "3.13.0"),
        ("commit", "b" * 40),
        ("lock_sha256", "b" * 64),
        ("source_sha256", "b" * 64),
    ],
)
def test_external_runtime_expectation_mismatch(checkout, monkeypatch, field, value):
    api, expected = setup_runtime(monkeypatch, checkout)
    wrong = api.ExpectedValidator.model_validate(expected.model_dump() | {field: value})
    with pytest.raises(ValueError, match=field):
        api.validate_validator_runtime(wrong)


@pytest.mark.parametrize(
    "name",
    [
        "src/bevcalib/new.py",
        "src/bevcalib/artifacts/validator_runtime.py",
        "scripts/check.py",
        ".agents/skills/study/validator.py",
        "schemas/study.json",
    ],
)
def test_same_head_changed_or_added_source_refused(checkout, monkeypatch, name):
    api, expected = setup_runtime(monkeypatch, checkout)
    path = checkout / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("changed source, including git-ignored files\n")
    with pytest.raises(ValueError, match="source_sha256"):
        api.validate_validator_runtime(expected)


@pytest.mark.parametrize("result", ["bad", "/wrong/root\n" + "a" * 40])
def test_wrong_git_checkout_or_malformed_identity(checkout, monkeypatch, result):
    api, expected = setup_runtime(monkeypatch, checkout)
    monkeypatch.setattr(
        api.subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, 0, result)
    )
    with pytest.raises(ValueError, match="Git checkout identity"):
        api.validate_validator_runtime(expected)


@pytest.mark.parametrize("error", [OSError("no git"), subprocess.CalledProcessError(1, "git")])
def test_git_failure_refused(checkout, monkeypatch, error):
    api, expected = setup_runtime(monkeypatch, checkout)

    def fail(*args, **kwargs):
        raise error

    monkeypatch.setattr(api.subprocess, "run", fail)
    with pytest.raises(ValueError, match="Git checkout identity"):
        api.validate_validator_runtime(expected)


@pytest.mark.parametrize("extra", [{"python_version": "3.12"}, {"commit": "HEAD"}, {"extra": 1}])
def test_runtime_expectation_schema(checkout, monkeypatch, extra):
    api, expected = setup_runtime(monkeypatch, checkout)
    with pytest.raises(ValidationError):
        api.ExpectedValidator.model_validate(expected.model_dump() | extra)
