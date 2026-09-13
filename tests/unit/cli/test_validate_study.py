"""Study validation CLI requires every identity/raw/checkpoint input explicitly."""

import importlib
import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from bevcalib.cli.app import app


def arguments(root):
    return [
        "validate-study",
        "--expectations",
        str(root / "expected.json"),
        "--artifacts-dir",
        str(root / "evidence"),
        "--claims",
        str(root / "claims.yaml"),
        "--raw-runs-dir",
        str(root / "raw"),
        "--checkpoint-17",
        str(root / "17.pt"),
        "--checkpoint-42",
        str(root / "42.pt"),
        "--checkpoint-73",
        str(root / "73.pt"),
        "--repository-root",
        str(root),
    ]


def test_cli_requires_complete_argument_inventory():
    result = CliRunner().invoke(app, ["validate-study"])
    assert result.exit_code == 2
    assert "expectations" in result.output


@pytest.mark.parametrize("failure", [False, True])
def test_cli_dispatch_and_failure_exit(tmp_path: Path, monkeypatch, failure):
    api = importlib.import_module("bevcalib.cli.validate_study")

    def validate(expected, artifacts, claims, raw, checkpoints, *, repository_root):
        assert (expected, artifacts, claims, raw, repository_root) == (
            tmp_path / "expected.json",
            tmp_path / "evidence",
            tmp_path / "claims.yaml",
            tmp_path / "raw",
            tmp_path,
        )
        assert checkpoints == {f"learned-{seed}": tmp_path / f"{seed}.pt" for seed in (17, 42, 73)}
        if failure:
            raise ValueError("independent identity mismatch")
        return {"status": "validated-inputs"}

    monkeypatch.setattr(api, "validate_fault_study", validate)
    result = CliRunner().invoke(app, arguments(tmp_path))
    if failure:
        assert result.exit_code == 1
        assert "independent identity mismatch" in result.output
        assert "validated-inputs" not in result.output
    else:
        assert result.exit_code == 0
        assert json.loads(result.output) == {"status": "validated-inputs"}
    assert list(tmp_path.iterdir()) == []  # CLI never manufactures input or output files.
