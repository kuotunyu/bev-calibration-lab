"""Checkpoint and training boundaries on real synthetic CPU fixtures."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from tests.unit.nuscenes_adapter.test_installation import installation_root as installation_root
from tests.unit.training.test_engine import cohort_document
from tests.unit.training.test_torch_backend import (
    tiny_model,
)
from tests.unit.training.test_torch_backend import (
    trained_checkpoint as trained_checkpoint,
)
from tests.unit.training.test_torch_backend import (
    training_workspace as training_workspace,
)


@pytest.mark.parametrize("case", ["config", "seed", "role", "joint", "calibration_logs"])
def test_checkpoint_context_revalidates_training_provenance(trained_checkpoint, case: str) -> None:  # type: ignore[no-untyped-def]
    from copy import deepcopy

    from bevcalib.artifacts.result_documents import digest
    from bevcalib.cohort.manifest import manifest_hash
    from bevcalib.training.checkpoints import validate_training_context

    _, backend = trained_checkpoint
    context = deepcopy(backend.context)
    if case == "config":
        context["config_raw"] += "\n"
    elif case == "seed":
        context["seed"] = 99
    elif case == "role":
        context["development"] = context["calibration"]
    elif case == "joint":
        context["cohort_manifest_sha256"] = "c" * 64
    else:
        context["synthetic_fixture"] = False
        context["development"] = cohort_document("development", "dev", 100)
        context["calibration"] = cohort_document("calibration", "cal", 20)
        for scene in context["calibration"]["scenes"]:
            scene["log_token"] = "cal-log"
        context["calibration"]["manifest_sha256"] = manifest_hash(context["calibration"])
        context["cohort_manifest_sha256"] = digest(
            {
                "development": context["development"]["manifest_sha256"],
                "calibration": context["calibration"]["manifest_sha256"],
            }
        )
    with pytest.raises(ValueError):
        validate_training_context(context, allow_synthetic=True)


def test_backend_refuses_changed_manifest_and_nonfinite_training_loss(
    trained_checkpoint, training_workspace
) -> None:  # type: ignore[no-untyped-def]
    from bevcalib.cohort.manifest import manifest_hash

    _, backend = trained_checkpoint
    _, _, dev, _ = training_workspace
    model, optimizer = backend.create_model_and_optimizer(backend.config.model_dump())
    with torch.no_grad():
        model[3].bias.fill_(float("nan"))
    with pytest.raises(ValueError, match="nonfinite"):
        backend.run_epoch(model, optimizer, dev, 1)
    body = json.loads(dev.read_text(encoding="utf-8"))
    body["scenes"][0]["camera_timestamps"][0] += 1
    body["manifest_sha256"] = manifest_hash(body)
    dev.write_text(json.dumps(body), encoding="utf-8")
    with pytest.raises(ValueError, match="manifest drift"):
        backend.run_epoch(model, optimizer, dev, 1)


def test_partial_batch_and_actual_parameter_update(training_workspace, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    from bevcalib.training.engine import train_learned_corrector
    from bevcalib.training.torch_backend import TorchBackend

    root, config, dev, cal = training_workspace
    document = yaml.safe_load(config.read_text(encoding="utf-8"))
    document.update(batch_size=2, warmup_epochs=0)
    config.write_text(yaml.safe_dump(document), encoding="utf-8")
    initial = []

    def factory():  # type: ignore[no-untyped-def]
        model = tiny_model()
        initial.append(model[3].bias.detach().clone())
        return model

    backend = TorchBackend(root, synthetic_fixture=True, model_factory=factory)
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        result = train_learned_corrector(
            config, dev, cal, tmp_path / "partial", 17, backend=backend, synthetic_fixture=True
        )
        state = torch.load(result.checkpoint_path, weights_only=True)["state_dict"]
        assert not torch.equal(initial[0], state["3.bias"])
        assert backend.optimizer_steps == 2
    finally:
        torch.set_num_threads(previous)


def test_local_timm_weights_are_loaded_and_widened_without_download(
    tmp_path: Path, trained_checkpoint
) -> None:  # type: ignore[no-untyped-def]
    import hashlib

    import timm

    from bevcalib.training.engine import load_corrector_config
    from bevcalib.training.torch_backend import (
        PretrainedWeights,
        TorchBackend,
        create_configured_model,
    )

    config = load_corrector_config(Path("configs/correctors/convnextv2_tiny_v1.yaml"))
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        # Locally initialized weights are synthetic adapter evidence, never pretrained accuracy.
        source = timm.create_model("convnextv2_tiny", pretrained=False)
        source_stem = source.get_submodule("stem.0")
        assert isinstance(source_stem, torch.nn.Conv2d)
        assert source_stem.bias is not None
        with torch.no_grad():
            source_stem.weight.fill_(0.125)
            source_stem.bias.fill_(0.25)
        path = tmp_path / "synthetic-local-weights.pt"
        torch.save(source.state_dict(), path)
        weights = PretrainedWeights(
            path,
            "convnextv2_tiny.fcmae_ft_in22k_in1k",
            hashlib.sha256(path.read_bytes()).hexdigest(),
        )
        _, training_backend = trained_checkpoint
        adapter = TorchBackend(training_backend.dataroot, weights=weights, synthetic_fixture=True)
        provenance = adapter.prepare(training_backend.config.model_dump(), training_backend.context)
        assert provenance["pretrained"]["sha256"] == weights.sha256
        model = create_configured_model(config.model_dump(), weights=weights)
        torch.testing.assert_close(
            model.stem[0].weight, torch.full_like(model.stem[0].weight, 0.125 * (3 / 5) ** 0.5)
        )
        torch.testing.assert_close(model.stem[0].bias, torch.full_like(model.stem[0].bias, 0.25))
        assert model.head.fc.out_features == 6
    finally:
        torch.set_num_threads(previous)


def test_factory_injection_is_synthetic_only() -> None:
    from bevcalib.training.torch_backend import create_configured_model

    with pytest.raises(ValueError, match="injected"):
        create_configured_model({}, model_factory=tiny_model)


def test_learned_inference_refuses_bad_input_prediction_and_evaluation_leakage(
    trained_checkpoint,
) -> None:  # type: ignore[no-untyped-def]
    from bevcalib.cohort.manifest import CohortManifestV2, manifest_hash
    from bevcalib.training.torch_backend import load_learned_checkpoint

    result, backend = trained_checkpoint
    predictor = load_learned_checkpoint(
        result.checkpoint_path,
        expected_protocol_hash=backend.context["protocol_sha256"],
        expected_seed=17,
        allow_synthetic=True,
        model_factory=tiny_model,
    )
    with pytest.raises(ValueError, match="input"):
        predictor.predict(np.zeros((5, 32, 32), dtype=np.float32))
    with torch.no_grad():
        predictor.model[3].bias.fill_(float("nan"))
    with pytest.raises(ValueError, match="prediction"):
        predictor.predict(np.zeros((5, 448, 800), dtype=np.float32))
    evaluation = cohort_document("evaluation", "eval", 1)
    predictor.validate_evaluation(CohortManifestV2.model_validate(evaluation))
    with pytest.raises(ValueError, match="role/protocol/dataset"):
        predictor.validate_evaluation(
            CohortManifestV2.model_validate(backend.context["development"])
        )
    evaluation["scenes"][0]["log_token"] = backend.context["development"]["scenes"][0]["log_token"]
    evaluation["manifest_sha256"] = manifest_hash(evaluation)
    with pytest.raises(ValueError, match="leaked"):
        predictor.validate_evaluation(CohortManifestV2.model_validate(evaluation))


def test_explicit_synthetic_service_still_refuses_role_leakage(
    training_workspace, tmp_path: Path
) -> None:  # type: ignore[no-untyped-def]
    from bevcalib.training.engine import train_learned_corrector
    from bevcalib.training.torch_backend import TorchBackend

    root, config, dev, cal = training_workspace
    with pytest.raises(ValueError, match="synthetic training"):
        train_learned_corrector(
            config,
            cal,
            dev,
            tmp_path / "bad-role",
            17,
            backend=TorchBackend(root, synthetic_fixture=True, model_factory=tiny_model),
            synthetic_fixture=True,
        )
    assert not (tmp_path / "bad-role").exists()


def formal_context(backend):  # type: ignore[no-untyped-def]
    from copy import deepcopy

    from bevcalib.artifacts.result_documents import digest

    context = deepcopy(backend.context)
    context["synthetic_fixture"] = False
    context["development"] = cohort_document("development", "dev", 100)
    context["calibration"] = cohort_document("calibration", "cal", 20)
    context["cohort_manifest_sha256"] = digest(
        {
            "development": context["development"]["manifest_sha256"],
            "calibration": context["calibration"]["manifest_sha256"],
        }
    )
    return json.loads(json.dumps(context))


def test_production_prepare_requires_matching_mode_and_pretraining(trained_checkpoint) -> None:  # type: ignore[no-untyped-def]
    from bevcalib.training.torch_backend import TorchBackend

    _, backend = trained_checkpoint
    context = formal_context(backend)
    with pytest.raises(ValueError, match="modes differ"):
        backend.prepare(backend.config.model_dump(), context)
    formal = TorchBackend(backend.dataroot)
    with pytest.raises(ValueError, match="pretrained"):
        formal.prepare(backend.config.model_dump(), context)


def write_changed_checkpoint(
    result, payload, *, bind_metadata: bool = True, bind_context: bool = False
) -> None:  # type: ignore[no-untyped-def]
    import hashlib

    from bevcalib.artifacts.result_documents import digest

    if bind_metadata:
        payload["metadata_sha256"] = digest(payload["metadata"])
    torch.save(payload, result.checkpoint_path)
    record = json.loads(result.run_record_path.read_text(encoding="utf-8"))
    record["artifacts"]["selected_checkpoint"] = hashlib.sha256(
        result.checkpoint_path.read_bytes()
    ).hexdigest()
    if bind_context:
        path = result.run_record_path.parent / "training_provenance.json"
        body = json.loads(path.read_text(encoding="utf-8"))
        for key in body:
            body[key] = payload["metadata"][key]
        path.write_text(json.dumps(body), encoding="utf-8")
        record["artifacts"]["training_provenance"] = hashlib.sha256(path.read_bytes()).hexdigest()
        for key in ("run_id", "config_sha256", "protocol_sha256", "cohort_manifest_sha256", "seed"):
            record[key] = payload["metadata"][key]
    result.run_record_path.write_text(json.dumps(record), encoding="utf-8")


@pytest.mark.parametrize(
    "case,message",
    [
        ("hash", "metadata hash"),
        ("identity", "selected run identity"),
        ("artifact", "artifact hash"),
        ("context", "bound training provenance"),
        ("policy", "policy mismatch"),
        ("formal_missing", "pretrained identity"),
        ("formal_factory", "injected model factory"),
    ],
)
def test_checkpoint_consumer_validates_every_provenance_binding(
    trained_checkpoint, case: str, message: str
) -> None:  # type: ignore[no-untyped-def]
    import hashlib

    from bevcalib.training.torch_backend import load_learned_checkpoint

    result, backend = trained_checkpoint
    payload = torch.load(result.checkpoint_path, weights_only=True)
    if case in ("hash", "identity"):
        payload["metadata"]["run_id"] = "different"
        write_changed_checkpoint(result, payload, bind_metadata=case != "hash")
    elif case in ("artifact", "context"):
        path = result.run_record_path.parent / "training_provenance.json"
        body = json.loads(path.read_text(encoding="utf-8"))
        body["run_id"] = "different"
        path.write_text(json.dumps(body), encoding="utf-8")
        if case == "context":
            record = json.loads(result.run_record_path.read_text(encoding="utf-8"))
            record["artifacts"]["training_provenance"] = hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
            result.run_record_path.write_text(json.dumps(record), encoding="utf-8")
    elif case == "policy":
        payload["metadata"]["calibration_policy"] = "unknown"
        write_changed_checkpoint(result, payload)
    else:
        payload["metadata"].update(formal_context(backend))
        payload["metadata"]["pretrained"] = (
            None
            if case == "formal_missing"
            else {"source_id": "convnextv2_tiny.fcmae_ft_in22k_in1k", "sha256": "a" * 64}
        )
        write_changed_checkpoint(result, payload, bind_context=True)
    with pytest.raises(ValueError, match=message):
        load_learned_checkpoint(
            result.checkpoint_path,
            expected_protocol_hash=backend.context["protocol_sha256"],
            expected_seed=17,
            allow_synthetic=True,
            model_factory=tiny_model,
        )


def test_changed_run_hardware_cannot_claim_checkpoint_provenance(trained_checkpoint) -> None:  # type: ignore[no-untyped-def]
    from bevcalib.training.torch_backend import load_learned_checkpoint

    result, backend = trained_checkpoint
    record = json.loads(result.run_record_path.read_text(encoding="utf-8"))
    record["hardware"] = {"runtime": "different"}
    result.run_record_path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError, match="producer"):
        load_learned_checkpoint(
            result.checkpoint_path,
            expected_protocol_hash=backend.context["protocol_sha256"],
            expected_seed=17,
            allow_synthetic=True,
            model_factory=tiny_model,
        )


def test_formal_checkpoint_requires_valid_pretrained_identity(trained_checkpoint) -> None:  # type: ignore[no-untyped-def]
    from bevcalib.training.torch_backend import load_learned_checkpoint

    result, backend = trained_checkpoint
    payload = torch.load(result.checkpoint_path, weights_only=True)
    payload["metadata"].update(formal_context(backend))
    payload["metadata"]["pretrained"] = {"source_id": "resnet18.a1_in1k", "sha256": "a" * 64}
    write_changed_checkpoint(result, payload, bind_context=True)
    with pytest.raises(ValueError, match="pretrained profile"):
        load_learned_checkpoint(
            result.checkpoint_path,
            expected_protocol_hash=backend.context["protocol_sha256"],
            expected_seed=17,
            allow_synthetic=True,
            model_factory=tiny_model,
        )


def test_checkpoint_evaluation_boundary_rechecks_manifest_hash(trained_checkpoint) -> None:  # type: ignore[no-untyped-def]
    from bevcalib.cohort.manifest import CohortManifestV2
    from bevcalib.training.torch_backend import load_learned_checkpoint

    result, backend = trained_checkpoint
    predictor = load_learned_checkpoint(
        result.checkpoint_path,
        expected_protocol_hash=backend.context["protocol_sha256"],
        expected_seed=17,
        allow_synthetic=True,
        model_factory=tiny_model,
    )
    evaluation = CohortManifestV2.model_validate(cohort_document("evaluation", "eval", 1))
    changed = evaluation.model_copy(update={"manifest_sha256": "c" * 64})
    with pytest.raises(ValueError, match="manifest hash"):
        predictor.validate_evaluation(changed)


def test_formal_predictor_refuses_underfilled_evaluation(trained_checkpoint) -> None:  # type: ignore[no-untyped-def]
    from bevcalib.cohort.manifest import CohortManifestV2
    from bevcalib.training.checkpoints import LearnedPredictor

    _, backend = trained_checkpoint
    predictor = LearnedPredictor(None, formal_context(backend), "a" * 64)
    predictor.validate_evaluation(
        CohortManifestV2.model_validate(cohort_document("evaluation", "eval", 30))
    )
    with pytest.raises(ValueError, match="complete formal evaluation"):
        predictor.validate_evaluation(
            CohortManifestV2.model_validate(cohort_document("evaluation", "eval", 1))
        )


def test_calibration_corruptions_ignore_training_seed_and_epoch(trained_checkpoint) -> None:  # type: ignore[no-untyped-def]
    _, backend = trained_checkpoint
    cal = backend.dataroot.parent / "calibration.json"
    dev = backend.dataroot.parent / "development.json"
    baseline = next(backend._batches(cal, "calibration", 0))
    development = next(backend._batches(dev, "development", 1))[1]
    backend.context["seed"] = 42
    other = next(backend._batches(cal, "calibration", 29))
    assert torch.equal(baseline[0], other[0])
    assert torch.equal(baseline[1], other[1])
    assert not torch.equal(development, next(backend._batches(dev, "development", 1))[1])
    assert not torch.equal(
        next(backend._batches(dev, "development", 1))[1],
        next(backend._batches(dev, "development", 2))[1],
    )
