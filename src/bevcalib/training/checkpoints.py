"""Checkpoint provenance validation and learned inference, with lazy Torch loading."""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from pydantic import BaseModel, ConfigDict, Field

from bevcalib.artifacts.result_documents import digest
from bevcalib.artifacts.run_record import RunProvenance, RunRecordV1
from bevcalib.cohort.manifest import CohortManifestV2
from bevcalib.cohort.records import validate_records
from bevcalib.preprocessing import PREPROCESSING_ID
from bevcalib.training.engine import CorrectorConfigV1, _require_disjoint

PREPROCESSING = {
    "id": PREPROCESSING_ID,
    "rgb_mean": [0.485, 0.456, 0.406],
    "rgb_std": [0.229, 0.224, 0.225],
    "resize": "bilinear-half-pixel",
    "depth": "log1p(d)/log1p(80);clip[0,1]",
    "channels": ["red", "green", "blue", "depth", "valid"],
}
CALIBRATION_POLICY = "bev-calibration-objective/v1:sha256(calibration|protocol):epoch0"


class PretrainedIdentity(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    source_id: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    def validate_profile(self) -> None:
        import timm

        profile = timm.get_pretrained_cfg(self.source_id)
        if (
            profile is None
            or profile.architecture != "convnextv2_tiny"
            or tuple(profile.mean) != tuple(PREPROCESSING["rgb_mean"])
            or tuple(profile.std) != tuple(PREPROCESSING["rgb_std"])
            or profile.num_classes != 1000
        ):
            raise ValueError("pretrained profile is incompatible with ImageNet preprocessing")


def calibration_fault_seed(protocol_hash: str) -> int:
    return int.from_bytes(
        hashlib.sha256(f"calibration|{protocol_hash}".encode()).digest()[:4], "big"
    )


def validate_training_context(context: dict[str, Any], *, allow_synthetic: bool) -> None:
    RunProvenance.model_validate(context["producer"])
    if context["synthetic_fixture"] and not allow_synthetic:
        raise ValueError("synthetic checkpoint is not formal pretrained evidence")
    config = CorrectorConfigV1.model_validate(yaml.safe_load(context["config_raw"]))
    if hashlib.sha256(context["config_raw"].encode()).hexdigest() != context["config_sha256"]:
        raise ValueError("checkpoint config hash mismatch")
    if context["seed"] not in config.seeds:
        raise ValueError("checkpoint seed is outside config")
    development = CohortManifestV2.model_validate(context["development"])
    calibration = CohortManifestV2.model_validate(context["calibration"])
    for manifest, role, count in (
        (development, "development", 100),
        (calibration, "calibration", 20),
    ):
        if (
            manifest.role != role
            or manifest.protocol_hash != context["protocol_sha256"]
            or manifest.dataset_version != "v1.0-trainval"
            or not manifest.scenes
            or any(scene.official_split != "train" for scene in manifest.scenes)
            or (not context["synthetic_fixture"] and len(manifest.scenes) != count)
        ):
            raise ValueError(
                "checkpoint training cohort role/protocol/dataset/completeness mismatch"
            )
    _require_disjoint(development, calibration)
    if (
        not context["synthetic_fixture"]
        and len({scene.log_token for scene in calibration.scenes}) != 20
    ):
        raise ValueError("checkpoint calibration needs twenty distinct logs")
    if (
        digest(
            {"development": development.manifest_sha256, "calibration": calibration.manifest_sha256}
        )
        != context["cohort_manifest_sha256"]
    ):
        raise ValueError("checkpoint joint cohort hash mismatch")


@dataclass
class LearnedPredictor:
    model: Any
    metadata: dict[str, Any]
    checkpoint_sha256: str
    device: str = "cpu"

    def predict(self, tensor: np.ndarray) -> np.ndarray:
        import torch

        config = yaml.safe_load(self.metadata["config_raw"])
        if (
            tensor.shape != (5, config["input_height"], config["input_width"])
            or not np.isfinite(tensor).all()
        ):
            raise ValueError("learned input shape or values differ from checkpoint preprocessing")
        with torch.no_grad():
            result = (
                self.model(
                    torch.as_tensor(tensor, dtype=torch.float32, device=self.device).unsqueeze(0)
                )
                .cpu()
                .numpy()[0]
            )
        if result.shape != (6,) or not np.isfinite(result).all():
            raise ValueError("learned prediction must contain six finite degree/metre values")
        return result

    def validate_evaluation(self, evaluation: CohortManifestV2) -> None:
        evaluation = CohortManifestV2.model_validate(evaluation.model_dump(mode="json"))
        development = CohortManifestV2.model_validate(self.metadata["development"])
        calibration = CohortManifestV2.model_validate(self.metadata["calibration"])
        if (
            evaluation.role != "evaluation"
            or evaluation.protocol_hash != self.metadata["protocol_sha256"]
            or evaluation.dataset_version != development.dataset_version
        ):
            raise ValueError("evaluation role/protocol/dataset differs from checkpoint")
        if not self.metadata["synthetic_fixture"] and (
            len(evaluation.scenes) != 30
            or any(scene.official_split != "val" for scene in evaluation.scenes)
        ):
            raise ValueError("checkpoint requires a complete formal evaluation cohort")
        training = development.scenes + calibration.scenes
        if {scene.log_token for scene in training} & {
            scene.log_token for scene in evaluation.scenes
        }:
            raise ValueError("evaluation log leaked into checkpoint training")
        validate_records(training + evaluation.scenes)


def load_learned_checkpoint(
    path: Path,
    *,
    expected_protocol_hash: str,
    expected_seed: int,
    allow_synthetic: bool = False,
    model_factory: Callable[[], Any] | None = None,
    device: str = "cpu",
) -> LearnedPredictor:
    import torch

    from bevcalib.training.torch_backend import create_configured_model

    if device not in ("cpu", "cuda"):
        raise ValueError(f"unsupported inference device: {device}")
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA inference was requested but CUDA is unavailable")

    checkpoint_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    record = RunRecordV1.model_validate_json((path.parent / "run_record.json").read_bytes())
    if (
        record.status != "succeeded"
        or record.artifacts.get("selected_checkpoint") != checkpoint_hash
        or record.seed != expected_seed
        or record.protocol_sha256 != expected_protocol_hash
    ):
        raise ValueError(
            "selected checkpoint hash/status/seed/protocol differs from the run record"
        )
    payload = torch.load(path, map_location="cpu", weights_only=True)
    metadata = payload["metadata"]
    if payload["metadata_sha256"] != digest(metadata):
        raise ValueError("checkpoint metadata hash mismatch")
    validate_training_context(metadata, allow_synthetic=allow_synthetic)
    for key in ("run_id", "config_sha256", "protocol_sha256", "cohort_manifest_sha256", "seed"):
        if metadata[key] != getattr(record, key):
            raise ValueError("checkpoint metadata differs from selected run identity")
    if metadata["producer"] != {
        "commit": record.commit,
        "lock_sha256": record.lock_sha256,
        "hardware": record.hardware,
    }:
        raise ValueError("checkpoint producer differs from selected run record")
    provenance_path = path.parent / "training_provenance.json"
    if (
        record.artifacts.get("training_provenance")
        != hashlib.sha256(provenance_path.read_bytes()).hexdigest()
    ):
        raise ValueError("training provenance artifact hash mismatch")
    import json

    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    if any(metadata[key] != value for key, value in provenance.items()):
        raise ValueError("checkpoint differs from bound training provenance")
    if (
        metadata["schema_version"] != "bev-learned-checkpoint/v2"
        or metadata["preprocessing"] != PREPROCESSING
        or metadata["calibration_policy"] != CALIBRATION_POLICY
        or metadata["calibration_fault_seed"] != calibration_fault_seed(expected_protocol_hash)
    ):
        raise ValueError("checkpoint preprocessing or calibration policy mismatch")
    if not metadata["synthetic_fixture"] and metadata["pretrained"] is None:
        raise ValueError("formal checkpoint lacks pretrained identity")
    if not metadata["synthetic_fixture"]:
        PretrainedIdentity.model_validate(metadata["pretrained"]).validate_profile()
    if model_factory is not None and not metadata["synthetic_fixture"]:
        raise ValueError("formal checkpoint cannot use an injected model factory")
    config = CorrectorConfigV1.model_validate(yaml.safe_load(metadata["config_raw"]))
    model = create_configured_model(
        config.model_dump(), synthetic_fixture=True, model_factory=model_factory
    )
    model.load_state_dict(payload["state_dict"], strict=True)
    model.to(device)
    model.eval()
    return LearnedPredictor(model, metadata, checkpoint_hash, device)
