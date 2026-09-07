"""Actual Torch training, local pretrained initialization and validated inference."""

from __future__ import annotations

import hashlib
import math
import random
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from bevcalib.artifacts.result_documents import digest
from bevcalib.cohort.manifest import CohortManifestV2, load_manifest
from bevcalib.correctors.learned import initialize_convnextv2_five_channel
from bevcalib.nuscenes_adapter.installation import resolve_installation
from bevcalib.perturbations.schedule import sample_training_fault
from bevcalib.preprocessing import load_observation, prepare_input
from bevcalib.training.checkpoints import (
    CALIBRATION_POLICY as CALIBRATION_POLICY,
)
from bevcalib.training.checkpoints import (
    PREPROCESSING,
    PretrainedIdentity,
    validate_training_context,
)
from bevcalib.training.checkpoints import (
    calibration_fault_seed as calibration_fault_seed,
)
from bevcalib.training.checkpoints import (
    load_learned_checkpoint as load_learned_checkpoint,
)
from bevcalib.training.dataset import target_for_fault
from bevcalib.training.engine import CorrectorConfigV1
from bevcalib.training.loss import normalized_huber_loss


@dataclass(frozen=True)
class PretrainedWeights:
    path: Path
    source_id: str
    sha256: str

    def validate(self) -> None:
        PretrainedIdentity(source_id=self.source_id, sha256=self.sha256).validate_profile()
        if hashlib.sha256(self.path.read_bytes()).hexdigest() != self.sha256:
            raise ValueError("local pretrained weight hash mismatch")


def create_configured_model(
    config: Mapping[str, Any],
    *,
    weights: PretrainedWeights | None = None,
    synthetic_fixture: bool = False,
    model_factory: Callable[[], Any] | None = None,
) -> Any:
    import timm

    if model_factory is not None and not synthetic_fixture:
        raise ValueError("injected architectures are restricted to explicit synthetic fixtures")
    if weights is None and not synthetic_fixture:
        raise ValueError("formal training requires explicit local pretrained weights")
    if weights is not None:
        weights.validate()
        model: Any = timm.create_model(weights.source_id, pretrained=False)
        timm.models.load_checkpoint(model, str(weights.path))
        model.reset_classifier(6)
    elif model_factory is not None:
        model = model_factory()
    else:
        model = timm.create_model(str(config["architecture"]), pretrained=False, num_classes=6)
    return initialize_convnextv2_five_channel(model)


class TorchBackend:
    """CPU by default; input checks precede model/optimizer/output creation."""

    def __init__(
        self,
        dataroot: Path,
        *,
        weights: PretrainedWeights | None = None,
        synthetic_fixture: bool = False,
        model_factory: Callable[[], Any] | None = None,
        device: str = "cpu",
    ) -> None:
        self.dataroot, self.weights, self.synthetic_fixture = (
            Path(dataroot),
            weights,
            synthetic_fixture,
        )
        self.model_factory, self.device = model_factory, device
        self.optimizer_steps = 0

    def prepare(
        self, config: Mapping[str, object], context: Mapping[str, object]
    ) -> Mapping[str, object]:
        self.config = CorrectorConfigV1.model_validate(config)
        self.context: dict[str, Any] = dict(context)
        validate_training_context(self.context, allow_synthetic=self.synthetic_fixture)
        if bool(context["synthetic_fixture"]) != self.synthetic_fixture:
            raise ValueError("backend and training evidence modes differ")
        if self.weights is None and not self.synthetic_fixture:
            raise ValueError("formal training requires local pretrained weights")
        if self.weights is not None:
            self.weights.validate()
        self.installation = resolve_installation(self.dataroot, "v1.0-trainval")
        self.provenance = {
            "preprocessing": PREPROCESSING,
            "calibration_policy": CALIBRATION_POLICY,
            "calibration_fault_seed": calibration_fault_seed(self.context["protocol_sha256"]),
            "pretrained": None
            if self.weights is None
            else {"source_id": self.weights.source_id, "sha256": self.weights.sha256},
        }
        return self.provenance

    def seed_all(self, seed: int) -> None:
        import torch

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(True)

    def create_model_and_optimizer(self, config: Mapping[str, object]) -> tuple[Any, Any]:
        import torch

        model = create_configured_model(
            config,
            weights=self.weights,
            synthetic_fixture=self.synthetic_fixture,
            model_factory=self.model_factory,
        ).to(self.device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=self.config.learning_rate, weight_decay=self.config.weight_decay
        )
        return model, optimizer

    def _batches(self, path: Path, role: str, epoch: int):  # type: ignore[no-untyped-def]
        import torch

        manifest = load_manifest(path)
        if (
            not isinstance(manifest, CohortManifestV2)
            or manifest.model_dump(mode="json") != self.context[role]
        ):
            raise ValueError("training manifest drift before observation access")
        seed = (
            self.context["seed"]
            if role == "development"
            else calibration_fault_seed(self.context["protocol_sha256"])
        )
        inputs, targets = [], []
        for scene in manifest.scenes:
            for index, sample in enumerate(scene.sample_tokens):
                fault = sample_training_fault(sample, epoch if role == "development" else 0, seed)
                observation = load_observation(self.installation, scene, index)
                inputs.append(
                    prepare_input(
                        observation, fault, self.config.input_height, self.config.input_width
                    )
                )
                targets.append(target_for_fault(fault))
                if len(inputs) == self.config.batch_size:
                    yield (
                        torch.as_tensor(np.stack(inputs), device=self.device),
                        torch.tensor(targets, dtype=torch.float32, device=self.device),
                    )
                    inputs, targets = [], []
        if inputs:
            yield (
                torch.as_tensor(np.stack(inputs), device=self.device),
                torch.tensor(targets, dtype=torch.float32, device=self.device),
            )

    def _loss(self, prediction: Any, target: Any) -> Any:
        return normalized_huber_loss(
            prediction,
            target,
            self.config.rotation_scale_deg,
            self.config.translation_scale_m,
            self.config.huber_delta,
        )

    def run_epoch(
        self, model: Any, optimizer: Any, development_manifest: Path, epoch: int
    ) -> float:
        warmup = self.config.warmup_epochs
        multiplier = (
            epoch / warmup
            if epoch <= warmup
            else 0.5 * (1 + math.cos(math.pi * (epoch - warmup) / (self.config.epochs - warmup)))
        )
        for group in optimizer.param_groups:
            group["lr"] = self.config.learning_rate * multiplier
        model.train()
        total, count = 0.0, 0
        for inputs, targets in self._batches(development_manifest, "development", epoch):
            optimizer.zero_grad(set_to_none=True)
            loss = self._loss(model(inputs), targets)
            if not bool(loss.isfinite()):
                raise ValueError("training loss is nonfinite")
            loss.backward()
            optimizer.step()
            self.optimizer_steps += 1
            total += float(loss.detach()) * len(inputs)
            count += len(inputs)
        return total / count

    def evaluate_loss(self, model: Any, calibration_manifest: Path) -> float:
        import torch

        model.eval()
        total, count = 0.0, 0
        with torch.no_grad():
            for inputs, targets in self._batches(calibration_manifest, "calibration", 0):
                total += float(self._loss(model(inputs), targets)) * len(inputs)
                count += len(inputs)
        return total / count

    def save_checkpoint(self, model: Any, path: Path, metadata: Mapping[str, object]) -> str:
        import torch

        body = dict(metadata) | self.provenance | {"schema_version": "bev-learned-checkpoint/v2"}
        temporary = path.with_suffix(".tmp")
        torch.save(
            {"metadata": body, "metadata_sha256": digest(body), "state_dict": model.state_dict()},
            temporary,
        )
        temporary.replace(path)
        return hashlib.sha256(path.read_bytes()).hexdigest()
