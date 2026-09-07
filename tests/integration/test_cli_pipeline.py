"""Actual CPU native adapters through commands and portable report rendering."""

import json
from pathlib import Path

import pytest
import torch
from tests.unit.analysis.test_claims import VOCABULARY
from tests.unit.nuscenes_adapter.test_installation import installation_root as installation_root
from tests.unit.report.test_builder import report_binding, report_paths
from tests.unit.test_evaluation import evaluation_workspace as evaluation_workspace
from tests.unit.training.test_torch_backend import tiny_model
from tests.unit.training.test_torch_backend import training_workspace as training_workspace
from typer.testing import CliRunner

from bevcalib.cli.app import app
from bevcalib.cli.runtime import Runtime


def invoke(arguments, runtime):
    result = CliRunner().invoke(app, [str(value) for value in arguments], obj=runtime)
    assert result.exit_code == 0, (result.output, result.exception)
    return result


def test_native_cli_evaluation_exports_to_portable_identical_reports(
    evaluation_workspace, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from bevcalib.artifacts.summary import write_safe_summary
    from bevcalib.metrics.summary import summarize_result_runs

    root, manifest = evaluation_workspace
    monkeypatch.setenv("NUSCENES_ROOT", str(root))
    options = Runtime(synthetic_fixture=True)
    private = tmp_path / "private-runs"
    for method in ("identity", "classical"):
        invoke(
            [
                "evaluate",
                "--protocol",
                "configs/protocols/nuscenes_calibration_v1.yaml",
                "--manifest",
                manifest,
                "--method",
                method,
                "--output-dir",
                private,
            ],
            options,
        )
    summary = summarize_result_runs(private, synthetic_fixture=True)
    public = tmp_path / "clean-portable"
    artifacts = public / "artifacts"
    write_safe_summary(summary, artifacts / "calibration_summary.json")
    claims = [
        {
            "claim_id": f"synthetic-{index}",
            "text": "Synthetic measured value.",
            "evidence_type": "synthetic",
            "protocol_hash": summary["protocol_hash"],
            "dataset_manifest_hash": summary["dataset_manifest_hash"],
            "artifact_path": "artifacts/calibration_summary.json",
            "metric_path": pointer,
            "report_binding": report_binding(summary, pointer),
            "status": "verified",
        }
        for index, pointer in enumerate(report_paths(summary))
    ]
    claims_path = public / "claims.yaml"
    claims_path.write_text(json.dumps(VOCABULARY | {"claims": claims}), encoding="utf-8")
    monkeypatch.delenv("NUSCENES_ROOT")
    monkeypatch.chdir(public)
    # Rendering uses the production default runtime, with no dataset or model paths.
    for destination in ("one", "two"):
        invoke(
            [
                "report",
                "--claims",
                claims_path,
                "--artifacts-dir",
                artifacts,
                "--output-dir",
                destination,
            ],
            None,
        )
    invoke(["audit-claims", "--claims", claims_path], None)
    assert (public / "one/index.html").read_bytes() == (public / "two/index.html").read_bytes()
    assert {path.name for path in artifacts.iterdir()} == {"calibration_summary.json"}
    assert b"outside_tolerance" in (public / "one/index.html").read_bytes()


def test_cli_real_training_produces_selected_checkpoint(
    training_workspace, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, config, development, calibration = training_workspace
    monkeypatch.setenv("NUSCENES_ROOT", str(root))
    original = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        invoke(
            [
                "train",
                "--config",
                config,
                "--development-manifest",
                development,
                "--calibration-manifest",
                calibration,
                "--output-dir",
                tmp_path / "trained",
                "--seed",
                17,
            ],
            Runtime(synthetic_fixture=True, model_factory=tiny_model),
        )
    finally:
        torch.set_num_threads(original)
    checkpoint = torch.load(tmp_path / "trained/selected_checkpoint.pt", weights_only=True)
    assert checkpoint["metadata"]["synthetic_fixture"] is True
    assert checkpoint["state_dict"]


def test_cli_preflight_and_freeze_emit_native_evidence_and_honest_shortages(
    evaluation_workspace, tmp_path: Path
) -> None:
    from bevcalib.cohort.manifest import load_manifest

    root, _ = evaluation_workspace
    preflight = tmp_path / "preflight.json"
    invoke(
        [
            "data",
            "preflight",
            "--dataroot",
            root,
            "--version",
            "v1.0-trainval",
            "--output",
            preflight,
        ],
        None,
    )
    evidence = json.loads(preflight.read_text(encoding="utf-8"))
    assert evidence["table_sha256"]
    frozen = tmp_path / "frozen"
    result = invoke(
        [
            "cohort",
            "freeze",
            "--dataroot",
            root,
            "--version",
            "v1.0-trainval",
            "--protocol",
            "configs/protocols/nuscenes_calibration_v1.yaml",
            "--output-dir",
            frozen,
        ],
        None,
    )
    assert "shortage=100" in result.output
    assert load_manifest(frozen / "evaluation.json").scenes
    assert not load_manifest(frozen / "development.json").scenes
