"""Explicit inference device propagation, with CPU execution and mocked CUDA plumbing."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml
from tests.unit.nuscenes_adapter.test_installation import installation_root as installation_root
from tests.unit.training.test_engine import cohort_document
from tests.unit.training.test_torch_backend import tiny_model
from tests.unit.training.test_torch_backend import trained_checkpoint as trained_checkpoint
from tests.unit.training.test_torch_backend import training_workspace as training_workspace
from typer.testing import CliRunner


def test_cli_passes_explicit_device_to_evaluation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import bevcalib.evaluation as service
    from bevcalib.cli.app import app

    captured = {}

    def evaluate(*args, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace(directory=tmp_path / "result")

    monkeypatch.setattr(service, "evaluate_calibration", evaluate)
    monkeypatch.setenv("NUSCENES_ROOT", str(tmp_path))
    monkeypatch.setenv("BEVCALIB_DEVICE", "cuda")
    result = CliRunner().invoke(
        app,
        [
            "evaluate",
            "--protocol",
            "protocol.yaml",
            "--manifest",
            "manifest.json",
            "--method",
            "learned",
            "--checkpoint",
            "selected.pt",
            "--output-dir",
            str(tmp_path / "out"),
        ],
    )
    assert result.exit_code == 0, result.output
    assert captured["device"] == "cuda"


def test_service_passes_device_before_output(
    trained_checkpoint, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import bevcalib.evaluation as service
    import bevcalib.training.checkpoints as checkpoints

    result, _ = trained_checkpoint
    manifest = tmp_path / "eval.json"
    manifest.write_text(json.dumps(cohort_document("evaluation", "eval", 1)), encoding="utf-8")
    captured = {}

    def load(*args, **kwargs):
        captured.update(kwargs)
        raise ValueError("captured loader boundary")

    monkeypatch.setattr(checkpoints, "load_learned_checkpoint", load)
    with pytest.raises(ValueError, match="captured loader"):
        service.evaluate_calibration(
            Path("configs/protocols/nuscenes_calibration_v1.yaml"),
            manifest,
            "learned",
            tmp_path / "out",
            dataroot=tmp_path,
            checkpoint=result.checkpoint_path,
            synthetic_fixture=True,
            device="cuda",
        )
    assert captured["device"] == "cuda"
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_loader_places_model_and_input_on_same_explicit_device(
    trained_checkpoint, monkeypatch: pytest.MonkeyPatch, device: str
) -> None:
    from bevcalib.training.checkpoints import load_learned_checkpoint

    result, backend = trained_checkpoint
    moved = []
    tensor_devices = []
    original_to = torch.nn.Module.to
    original_tensor = torch.as_tensor

    def move(model, target):
        moved.append(target)
        return original_to(model, "cpu")

    def tensor(data, **kwargs):
        tensor_devices.append(kwargs.get("device"))
        return original_tensor(data, **(kwargs | {"device": "cpu"}))

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.nn.Module, "to", move)
    monkeypatch.setattr(torch, "as_tensor", tensor)
    predictor = load_learned_checkpoint(
        result.checkpoint_path,
        expected_protocol_hash=backend.context["protocol_sha256"],
        expected_seed=17,
        allow_synthetic=True,
        model_factory=tiny_model,
        device=device,
    )
    tensor_devices.clear()  # Stem adaptation also constructs a CPU tensor during loading.
    config = yaml.safe_load(predictor.metadata["config_raw"])
    output = predictor.predict(
        np.zeros((5, config["input_height"], config["input_width"]), dtype=np.float32)
    )
    assert output.shape == (6,) and np.isfinite(output).all()
    assert moved == [device] and tensor_devices == [device]
    assert predictor.device == device


@pytest.mark.parametrize(
    "device,message", [("cuda", "CUDA.*unavailable"), ("mps", "unsupported inference device")]
)
def test_unavailable_or_unsupported_device_refused_before_checkpoint_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, device: str, message: str
) -> None:
    from bevcalib.training.checkpoints import load_learned_checkpoint

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(ValueError, match=message):
        load_learned_checkpoint(
            tmp_path / "absent.pt", expected_protocol_hash="a" * 64, expected_seed=17, device=device
        )
