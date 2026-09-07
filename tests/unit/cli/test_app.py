"""Every declared command is discoverable and refuses unusable inputs."""

from pathlib import Path

import pytest
from tests.unit.nuscenes_adapter.test_installation import installation_root as installation_root
from typer.testing import CliRunner

from bevcalib.cli.app import app


@pytest.mark.parametrize(
    "command",
    [
        ["data"],
        ["data", "preflight"],
        ["cohort"],
        ["cohort", "freeze"],
        ["evaluate"],
        ["train"],
        ["report"],
        ["audit-claims"],
    ],
)
def test_every_command_help_is_available(command) -> None:
    result = CliRunner().invoke(app, [*command, "--help"])
    assert result.exit_code == 0, result.output
    assert "Usage" in result.output


@pytest.mark.parametrize(
    "arguments",
    [
        [
            "data",
            "preflight",
            "--dataroot",
            "missing",
            "--version",
            "v1.0-mini",
            "--output",
            "out.json",
        ],
        [
            "data",
            "preflight",
            "--dataroot",
            "missing",
            "--version",
            "unknown",
            "--output",
            "out.json",
        ],
        [
            "cohort",
            "freeze",
            "--dataroot",
            "missing",
            "--version",
            "v1.0-mini",
            "--protocol",
            "missing.yaml",
            "--output-dir",
            "out",
        ],
        [
            "evaluate",
            "--protocol",
            "missing.yaml",
            "--manifest",
            "missing.json",
            "--method",
            "unknown",
            "--output-dir",
            "out",
        ],
        [
            "evaluate",
            "--protocol",
            "missing.yaml",
            "--manifest",
            "missing.json",
            "--method",
            "learned",
            "--output-dir",
            "out",
        ],
        [
            "evaluate",
            "--protocol",
            "missing.yaml",
            "--manifest",
            "missing.json",
            "--method",
            "identity",
            "--output-dir",
            "out",
        ],
        [
            "train",
            "--config",
            "missing.yaml",
            "--development-manifest",
            "missing.json",
            "--calibration-manifest",
            "missing.json",
            "--output-dir",
            "out",
            "--seed",
            "17",
        ],
        ["report", "--claims", "missing.yaml", "--artifacts-dir", "missing", "--output-dir", "out"],
        ["audit-claims", "--claims", "missing.yaml"],
    ],
)
def test_cli_refusals_are_clear_without_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, arguments
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("NUSCENES_ROOT", raising=False)
    result = CliRunner().invoke(app, arguments)
    assert result.exit_code != 0
    assert "No such command" not in result.output
    assert "Error" in result.output
    assert not (tmp_path / "out").exists() and not (tmp_path / "out.json").exists()


@pytest.mark.parametrize("with_weights", [False, True])
def test_formal_training_env_requires_explicit_weight_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, with_weights: bool
) -> None:
    monkeypatch.setenv("NUSCENES_ROOT", str(tmp_path))
    for name, value in (
        ("BEVCALIB_PRETRAINED_WEIGHTS", "local.pt"),
        ("BEVCALIB_PRETRAINED_SOURCE", "convnextv2_tiny.fcmae_ft_in22k_in1k"),
        ("BEVCALIB_PRETRAINED_SHA256", "a" * 64),
    ):
        if with_weights:
            monkeypatch.setenv(name, value)
        else:
            monkeypatch.delenv(name, raising=False)
    result = CliRunner().invoke(
        app,
        [
            "train",
            "--config",
            str(tmp_path / "missing.yaml"),
            "--development-manifest",
            "missing.json",
            "--calibration-manifest",
            "missing.json",
            "--output-dir",
            str(tmp_path / "out"),
            "--seed",
            "17",
        ],
    )
    assert result.exit_code != 0 and "Error" in result.output
    assert ("BEVCALIB_PRETRAINED" in result.output) != with_weights
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("case", ["missing_root", "reversed_calibrated", "reversed_ego"])
def test_preflight_runtime_failures_exit_one_with_pose_diagnostic(
    installation_root: Path, tmp_path: Path, case: str
) -> None:
    from tests.unit.nuscenes_adapter.test_installation import rewrite

    root = installation_root
    expected = "nuScenes root"
    if case == "missing_root":
        root = tmp_path / "absent"
    elif case == "reversed_calibrated":
        rewrite(
            root, "sample_data", lambda rows: rows[0].update(calibrated_sensor_token="camera-pose")
        )
        expected = "unresolved calibrated_sensor token: camera-pose"
    else:
        rewrite(root, "sample_data", lambda rows: rows[0].update(ego_pose_token="camera-cal"))
        expected = "unresolved ego_pose token: camera-cal"
    output = tmp_path / "preflight.json"
    result = CliRunner().invoke(
        app,
        [
            "data",
            "preflight",
            "--dataroot",
            str(root),
            "--version",
            "v1.0-mini",
            "--output",
            str(output),
        ],
    )
    assert result.exit_code == 1, result.output
    assert expected in result.output and "Error" in result.output
    assert not output.exists()


def test_usage_errors_remain_exit_two() -> None:
    result = CliRunner().invoke(app, ["data", "preflight", "--version", "unknown"])
    assert result.exit_code == 2
