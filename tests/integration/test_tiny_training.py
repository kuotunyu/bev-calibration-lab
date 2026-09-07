"""A real, tiny training run on synthetic tensors, end to end on the CPU.

The unit tests inject a fake framework so the orchestration can be checked without
torch. This one uses the actual model, the actual loss and the actual optimiser on
four synthetic five-channel examples, and asks the only question that matters
before spending a GPU on it: can this thing learn at all, and does it learn the
same way twice?
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="a real training step needs the train extra")

from bevcalib.correctors.learned import (  # noqa: E402
    build_five_channel_input,
    initialize_convnextv2_five_channel,
)
from bevcalib.training.loss import normalized_huber_loss  # noqa: E402

EXAMPLES = 4
IMAGE = 32


def tiny_model() -> Any:
    """A ConvNeXt-shaped stem and head, small enough to overfit four examples on a CPU."""

    from torch import nn

    class Tiny(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.stem = nn.Conv2d(3, 16, kernel_size=4, stride=4)
            self.norm = nn.GroupNorm(1, 16)
            self.head = nn.Linear(16, 6)

        def forward(self, tensor: Any) -> Any:
            hidden = self.norm(self.stem(tensor)).mean(dim=(2, 3))
            return self.head(hidden)

    torch.manual_seed(20260902)
    return initialize_convnextv2_five_channel(Tiny())


def synthetic_batch() -> tuple[Any, Any]:
    """Four distinguishable inputs and four fixed 6DoF targets, in physical units."""

    generator = np.random.default_rng(20260902)
    tensors = []
    for _ in range(EXAMPLES):
        # Spatially varied on purpose. A uniform image collapses to almost nothing
        # once the stem is followed by a group norm and a spatial mean, so it would
        # test the optimiser's patience rather than whether the pipeline can learn.
        rgb = generator.uniform(0.0, 1.0, size=(3, IMAGE, IMAGE)).astype(np.float32)
        depth = generator.uniform(1.0, 60.0, size=(IMAGE, IMAGE)).astype(np.float32)
        valid = generator.random((IMAGE, IMAGE)) < 0.3
        tensors.append(build_five_channel_input(rgb, depth, valid))
    inputs = torch.from_numpy(np.stack(tensors))
    targets = torch.tensor(
        generator.uniform(-1.0, 1.0, size=(EXAMPLES, 6)) * [2.0, 2.0, 2.0, 0.2, 0.2, 0.2],
        dtype=torch.float32,
    )
    return inputs, targets


def overfit(steps: int = 300) -> tuple[float, float, list[float]]:
    """Train the tiny model to memorise four examples; return first, last and the curve."""

    model = tiny_model()
    inputs, targets = synthetic_batch()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=0.05)

    curve: list[float] = []
    for _ in range(steps):
        optimizer.zero_grad()
        loss = normalized_huber_loss(model(inputs), targets)
        loss.backward()
        optimizer.step()
        curve.append(float(loss.detach()))
    return curve[0], curve[-1], curve


def test_the_model_can_memorise_four_examples() -> None:
    """If it cannot overfit four examples, no amount of real data will help it."""

    first, last, _ = overfit()

    assert last < 0.02 * first, f"loss only fell from {first} to {last}"


def test_the_loss_falls_rather_than_wandering() -> None:
    """A curve that bounces means the learning rate or the loss scale is wrong."""

    _, _, curve = overfit()

    early = sum(curve[:20]) / 20.0
    late = sum(curve[-20:]) / 20.0
    assert late < early
    assert all(value == value for value in curve), "the loss went NaN"


def test_the_same_seed_trains_to_the_same_place_twice() -> None:
    """Two runs that disagree cannot be told apart from two configurations that disagree."""

    _, first_last, _ = overfit(steps=60)
    _, second_last, _ = overfit(steps=60)

    assert first_last == pytest.approx(second_last, rel=1e-9)


def test_the_five_channel_input_is_what_the_model_actually_consumes() -> None:
    """The tensor the builder produces goes straight in, with no reshaping in between."""

    model = tiny_model()
    inputs, _ = synthetic_batch()

    assert tuple(inputs.shape) == (EXAMPLES, 5, IMAGE, IMAGE)
    assert tuple(model(inputs).shape) == (EXAMPLES, 6)


def test_a_trained_model_saves_and_reloads_to_the_same_predictions(tmp_path: Path) -> None:
    """A checkpoint that does not restore its model makes every recorded digest pointless."""

    model = tiny_model()
    inputs, _ = synthetic_batch()
    with torch.no_grad():
        before = model(inputs).clone()

    path = tmp_path / "corrector.pt"
    torch.save(model.state_dict(), path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()

    restored = tiny_model()
    restored.load_state_dict(torch.load(path, weights_only=True))
    with torch.no_grad():
        after = restored(inputs)

    torch.testing.assert_close(before, after)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest


def test_the_engine_drives_a_real_backend_through_a_whole_run(tmp_path: Path) -> None:
    """The orchestration and the framework meet here for the first time.

    The unit tests prove the engine's rules on a fake; this proves the rules hold
    when the thing being driven really trains, saves and reloads.
    """

    import shutil

    import yaml

    from bevcalib.artifacts.run_record import PROVENANCE_ENV_VAR
    from bevcalib.training.engine import train_learned_corrector

    repo_root = Path(__file__).resolve().parents[2]
    configs = tmp_path / "configs"
    (configs / "correctors").mkdir(parents=True)
    (configs / "perturbations").mkdir()
    shutil.copytree(repo_root / "configs/protocols", configs / "protocols")
    shutil.copyfile(
        repo_root / "configs" / "perturbations" / "formal_v1.yaml",
        configs / "perturbations" / "formal_v1.yaml",
    )
    document = yaml.safe_load(
        (repo_root / "configs" / "correctors" / "convnextv2_tiny_v1.yaml").read_text("utf-8")
    )
    config = configs / "correctors" / "tiny.yaml"
    config.write_text(yaml.safe_dump(document | {"epochs": 3}), encoding="utf-8")

    def cohort(role: str, prefix: str, count: int) -> Path:
        from tests.unit.training.test_engine import cohort_document

        body = cohort_document(role, prefix, count)
        path = tmp_path / f"{role}.json"
        path.write_text(json.dumps(body), encoding="utf-8")
        return path

    class TorchBackend:
        """A real backend: a real model, a real optimiser, a real checkpoint file."""

        def __init__(self) -> None:
            self.inputs, self.targets = synthetic_batch()

        def seed_all(self, seed: int) -> None:
            torch.manual_seed(seed)

        def create_model_and_optimizer(self, config: dict[str, Any]) -> tuple[Any, Any]:
            model = tiny_model()
            return model, torch.optim.AdamW(
                model.parameters(),
                lr=float(config["learning_rate"]) * 100.0,
                weight_decay=float(config["weight_decay"]),
            )

        def run_epoch(self, model: Any, optimizer: Any, manifest: Path, epoch: int) -> float:
            total = 0.0
            for _ in range(20):
                optimizer.zero_grad()
                loss = normalized_huber_loss(model(self.inputs), self.targets)
                loss.backward()
                optimizer.step()
                total += float(loss.detach())
            return total / 20.0

        def evaluate_loss(self, model: Any, manifest: Path) -> float:
            with torch.no_grad():
                return float(normalized_huber_loss(model(self.inputs), self.targets))

        def save_checkpoint(self, model: Any, path: Path, metadata: dict[str, Any]) -> str:
            torch.save(model.state_dict(), path)
            return hashlib.sha256(path.read_bytes()).hexdigest()

    import os

    os.environ[PROVENANCE_ENV_VAR] = json.dumps(
        {
            "commit": "a" * 40,
            "lock_sha256": "b" * 64,
            "hardware": {"gpu": "none", "runtime": "integration"},
        }
    )
    try:
        result = train_learned_corrector(
            config=config,
            development_manifest=cohort("development", "dev", 100),
            calibration_manifest=cohort("calibration", "cal", 20),
            output_dir=tmp_path / "run",
            seed=17,
            backend=TorchBackend(),
        )
    finally:
        os.environ.pop(PROVENANCE_ENV_VAR, None)

    assert result.selected_epoch in (1, 2, 3)
    assert result.checkpoint_path.exists()
    record = json.loads(result.run_record_path.read_text(encoding="utf-8"))
    assert record["status"] == "succeeded"
    assert (
        record["artifacts"]["selected_checkpoint"]
        == hashlib.sha256(result.checkpoint_path.read_bytes()).hexdigest()
    )
