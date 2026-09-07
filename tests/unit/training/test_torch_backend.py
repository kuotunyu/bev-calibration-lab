"""Real CPU tensors/optimizer/checkpoints on explicitly synthetic native observations."""

from __future__ import annotations

import dataclasses
import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from tests.unit.nuscenes_adapter.test_installation import installation_root as installation_root
from tests.unit.training.test_engine import PROVENANCE, cohort_document


def tiny_model():  # type: ignore[no-untyped-def]
    return torch.nn.Sequential(
        torch.nn.Conv2d(3, 4, 4, stride=4),
        torch.nn.AdaptiveAvgPool2d(1),
        torch.nn.Flatten(),
        torch.nn.Linear(4, 6),
    )


@pytest.fixture
def training_workspace(installation_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):  # type: ignore[no-untyped-def]
    from bevcalib.cohort.manifest import manifest_hash
    from bevcalib.nuscenes_adapter.installation import resolve_installation

    monkeypatch.setenv("BEVCALIB_RUN_PROVENANCE", json.dumps(PROVENANCE))
    root = installation_root
    for table in ("scene", "log", "sample", "sample_data"):
        path = root / "v1.0-mini" / f"{table}.json"
        original = json.loads(path.read_text(encoding="utf-8"))
        additional = []
        for row in original:
            new = row.copy()
            for field in ("token", "sample_token", "scene_token", "log_token"):
                if field in new:
                    new[field] += "-cal"
            if table == "scene":
                new["name"] = "scene-0001"
            additional.append(new)
        path.write_text(json.dumps(original + additional), encoding="utf-8")
    (root / "v1.0-mini").rename(root / "v1.0-trainval")
    scenes = resolve_installation(root, "v1.0-trainval").scene_records()
    configs = tmp_path / "configs"
    shutil.copytree(Path("configs"), configs)
    config_path = configs / "correctors/convnextv2_tiny_v1.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config.update(epochs=2, batch_size=1, warmup_epochs=1, learning_rate=0.001)
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    manifests = []
    for role, scene in zip(("development", "calibration"), scenes, strict=True):
        body = cohort_document(role, role, 1)
        body["scenes"] = [dataclasses.asdict(scene)]
        body["manifest_sha256"] = manifest_hash(body)
        path = tmp_path / f"{role}.json"
        path.write_text(json.dumps(body), encoding="utf-8")
        manifests.append(path)
    return root, config_path, *manifests


def test_real_backend_trains_selects_saves_and_loads_learned_inference(
    training_workspace, tmp_path: Path
) -> None:  # type: ignore[no-untyped-def]
    from bevcalib.training.engine import train_learned_corrector
    from bevcalib.training.torch_backend import TorchBackend, load_learned_checkpoint

    root, config, dev, cal = training_workspace
    original_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        backend = TorchBackend(root, synthetic_fixture=True, model_factory=tiny_model)
        result = train_learned_corrector(
            config, dev, cal, tmp_path / "run", 17, backend=backend, synthetic_fixture=True
        )
        assert result.selected_epoch in (1, 2)
        assert np.isfinite(result.calibration_loss)
        record = json.loads(result.run_record_path.read_text(encoding="utf-8"))
        assert record["run_id"].startswith("synthetic-")
        predictor = load_learned_checkpoint(
            result.checkpoint_path,
            expected_protocol_hash=record["protocol_sha256"],
            expected_seed=17,
            allow_synthetic=True,
            model_factory=tiny_model,
        )
        batch = torch.zeros((1, 5, 448, 800))
        saved = torch.load(result.checkpoint_path, weights_only=True)
        restored = tiny_model()
        from bevcalib.correctors.learned import initialize_convnextv2_five_channel

        initialize_convnextv2_five_channel(restored)
        restored.load_state_dict(saved["state_dict"])
        np.testing.assert_allclose(
            predictor.predict(batch.numpy()[0]), restored(batch).detach().numpy()[0]
        )
        assert predictor.metadata["synthetic_fixture"] is True
        assert predictor.metadata["preprocessing"]["rgb_mean"] == [0.485, 0.456, 0.406]
        assert (
            predictor.metadata["development"]["manifest_sha256"]
            != predictor.metadata["calibration"]["manifest_sha256"]
        )
        assert backend.optimizer_steps == 2
        with pytest.raises(ValueError, match="synthetic"):
            load_learned_checkpoint(
                result.checkpoint_path,
                expected_protocol_hash=record["protocol_sha256"],
                expected_seed=17,
            )
        with pytest.raises(ValueError, match="seed"):
            load_learned_checkpoint(
                result.checkpoint_path,
                expected_protocol_hash=record["protocol_sha256"],
                expected_seed=42,
                allow_synthetic=True,
            )
    finally:
        torch.set_num_threads(original_threads)


def test_formal_service_still_rejects_small_fixture_before_model(
    training_workspace, tmp_path: Path
) -> None:  # type: ignore[no-untyped-def]
    from bevcalib.training.engine import train_learned_corrector
    from bevcalib.training.torch_backend import TorchBackend

    root, config, dev, cal = training_workspace
    backend = TorchBackend(root, synthetic_fixture=True, model_factory=tiny_model)
    with pytest.raises(ValueError, match="exactly 100"):
        train_learned_corrector(config, dev, cal, tmp_path / "run", 17, backend=backend)
    assert backend.optimizer_steps == 0
    assert not (tmp_path / "run").exists()


def test_configured_architecture_local_initialization_has_correct_stem_and_output() -> None:
    from bevcalib.training.engine import load_corrector_config
    from bevcalib.training.torch_backend import create_configured_model

    config = load_corrector_config(Path("configs/correctors/convnextv2_tiny_v1.yaml"))
    original_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        model = create_configured_model(config.model_dump(), synthetic_fixture=True)
        assert model.stem[0].weight.shape == (96, 5, 4, 4)
        with torch.no_grad():
            assert model(torch.zeros(1, 5, 448, 800)).shape == (1, 6)
    finally:
        torch.set_num_threads(original_threads)


def test_production_model_cannot_silently_use_untrained_weights() -> None:
    from bevcalib.training.engine import load_corrector_config
    from bevcalib.training.torch_backend import create_configured_model

    config = load_corrector_config(Path("configs/correctors/convnextv2_tiny_v1.yaml"))
    with pytest.raises(ValueError, match="pretrained"):
        create_configured_model(config.model_dump())


def test_calibration_corruptions_are_fixed_across_training_seeds() -> None:
    from bevcalib.training.torch_backend import calibration_fault_seed

    assert calibration_fault_seed("a" * 64) == calibration_fault_seed("a" * 64)
    assert calibration_fault_seed("a" * 64) != calibration_fault_seed("b" * 64)


@pytest.fixture
def trained_checkpoint(training_workspace, tmp_path: Path):  # type: ignore[no-untyped-def]
    from bevcalib.training.engine import train_learned_corrector
    from bevcalib.training.torch_backend import TorchBackend

    root, config, dev, cal = training_workspace
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        backend = TorchBackend(root, synthetic_fixture=True, model_factory=tiny_model)
        result = train_learned_corrector(
            config, dev, cal, tmp_path / "run", 17, backend=backend, synthetic_fixture=True
        )
        yield result, backend
    finally:
        torch.set_num_threads(previous_threads)


def test_checkpoint_tensor_tampering_is_refused_by_selected_run_record(trained_checkpoint) -> None:  # type: ignore[no-untyped-def]
    from bevcalib.training.torch_backend import load_learned_checkpoint

    result, backend = trained_checkpoint
    saved = torch.load(result.checkpoint_path, weights_only=True)
    saved["state_dict"]["3.bias"] += 1
    torch.save(saved, result.checkpoint_path)
    with pytest.raises(ValueError, match="selected checkpoint"):
        load_learned_checkpoint(
            result.checkpoint_path,
            expected_protocol_hash=backend.context["protocol_sha256"],
            expected_seed=17,
            allow_synthetic=True,
            model_factory=tiny_model,
        )


def test_pretrained_source_hash_and_profile_are_required(tmp_path: Path) -> None:
    from bevcalib.training.torch_backend import PretrainedWeights

    path = tmp_path / "weights.pt"
    path.write_bytes(b"synthetic local hash probe")
    with pytest.raises(ValueError, match="profile"):
        PretrainedWeights(path, "resnet18.a1_in1k", "a" * 64).validate()
    with pytest.raises(ValueError, match="hash"):
        PretrainedWeights(path, "convnextv2_tiny.fcmae_ft_in22k_in1k", "a" * 64).validate()


def test_run_provenance_binds_fixed_calibration_objective(trained_checkpoint) -> None:  # type: ignore[no-untyped-def]
    import hashlib

    from bevcalib.training.torch_backend import CALIBRATION_POLICY, calibration_fault_seed

    result, backend = trained_checkpoint
    path = result.run_record_path.parent / "training_provenance.json"
    assert path.is_file()
    body = json.loads(path.read_text(encoding="utf-8"))
    assert body["backend"]["calibration_policy"] == CALIBRATION_POLICY
    assert body["backend"]["calibration_fault_seed"] == calibration_fault_seed(
        backend.context["protocol_sha256"]
    )
    record = json.loads(result.run_record_path.read_text(encoding="utf-8"))
    assert (
        record["artifacts"]["training_provenance"] == hashlib.sha256(path.read_bytes()).hexdigest()
    )
