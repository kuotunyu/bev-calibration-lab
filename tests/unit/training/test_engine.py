"""Contracts for orchestrating a learned-corrector training run.

The engine owns the parts that decide whether the result means anything: which
cohort each manifest is, that the two do not share a log, that the seed is set
before anything random exists, and that the checkpoint is chosen by the
calibration cohort alone. The framework is injected, so all of that is testable
without a GPU and without a dataset.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import pytest
import yaml

from bevcalib.artifacts.run_record import PROVENANCE_ENV_VAR

REPO_ROOT = Path(__file__).resolve().parents[3]
COMMITTED_CONFIG = REPO_ROOT / "configs" / "correctors" / "convnextv2_tiny_v1.yaml"
PROVENANCE = {
    "commit": "a" * 40,
    "lock_sha256": "b" * 64,
    "hardware": {"gpu": "none", "runtime": "pytest"},
}


def cohort_document(role: str, prefix: str, scene_count: int) -> dict[str, Any]:
    scenes = [
        {
            "scene_token": f"{prefix}-scene-{index}",
            "log_token": f"{prefix}-log-{index // 3}",
            "sample_tokens": (f"{prefix}-sample-{index}-0", f"{prefix}-sample-{index}-1"),
        }
        for index in range(scene_count)
    ]
    document = {"schema_version": "bev-calibration-cohort/v1", "role": role, "scenes": scenes}
    payload = json.dumps(document, sort_keys=True, separators=(",", ":")).encode()
    return document | {"manifest_sha256": hashlib.sha256(payload).hexdigest()}


def write_cohort(path: Path, role: str, prefix: str, scene_count: int) -> Path:
    path.write_text(json.dumps(cohort_document(role, prefix, scene_count)), encoding="utf-8")
    return path


class FakeBackend:
    """A stand-in framework whose per-epoch losses are decided by the test."""

    def __init__(
        self,
        training_losses: tuple[float, ...] = (4.0, 3.0, 2.0, 1.0),
        calibration_losses: tuple[float, ...] = (0.9, 0.4, 0.5, 0.6),
    ) -> None:
        self.training_losses = training_losses
        self.calibration_losses = calibration_losses
        self.calls: list[str] = []
        self.seed: int | None = None
        self.epochs_run: list[int] = []
        self.evaluated = 0

    def seed_all(self, seed: int) -> None:
        self.calls.append("seed_all")
        self.seed = seed

    def create_model_and_optimizer(self, config: dict[str, Any]) -> tuple[object, object]:
        self.calls.append("create_model_and_optimizer")
        return {"weights": 0.0, "architecture": config["architecture"]}, {"step": 0}

    def run_epoch(
        self, model: Any, optimizer: Any, development_manifest: Path, epoch: int
    ) -> float:
        self.calls.append(f"run_epoch:{epoch}")
        self.epochs_run.append(epoch)
        return self.training_losses[epoch - 1]

    def evaluate_loss(self, model: Any, calibration_manifest: Path) -> float:
        self.calls.append("evaluate_loss")
        value = self.calibration_losses[self.evaluated]
        self.evaluated += 1
        return value

    def save_checkpoint(self, model: Any, path: Path, metadata: dict[str, Any]) -> str:
        payload = json.dumps({"model": model, "metadata": dict(metadata)}, sort_keys=True)
        path.write_bytes(payload.encode())
        return hashlib.sha256(payload.encode()).hexdigest()


@pytest.fixture
def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    monkeypatch.setenv(PROVENANCE_ENV_VAR, json.dumps(PROVENANCE))

    configs = tmp_path / "configs"
    (configs / "correctors").mkdir(parents=True)
    (configs / "perturbations").mkdir()
    shutil.copyfile(
        REPO_ROOT / "configs" / "perturbations" / "formal_v1.yaml",
        configs / "perturbations" / "formal_v1.yaml",
    )
    document = yaml.safe_load(COMMITTED_CONFIG.read_text(encoding="utf-8"))
    config = configs / "correctors" / "tiny.yaml"
    config.write_text(yaml.safe_dump(document | {"epochs": 4}), encoding="utf-8")

    return {
        "config": config,
        "development": write_cohort(tmp_path / "dev.json", "development", "dev", 100),
        "calibration": write_cohort(tmp_path / "cal.json", "calibration", "cal", 20),
        "output": tmp_path / "run",
    }


def train(workspace: dict[str, Path], backend: FakeBackend, **overrides: Any) -> Any:
    from bevcalib.training.engine import train_learned_corrector

    arguments: dict[str, Any] = {
        "config": workspace["config"],
        "development_manifest": workspace["development"],
        "calibration_manifest": workspace["calibration"],
        "output_dir": workspace["output"],
        "seed": 17,
        "backend": backend,
    }
    return train_learned_corrector(**(arguments | overrides))


def test_the_seed_is_set_before_anything_random_can_exist(workspace: dict[str, Path]) -> None:
    """Seeding after the model is built leaves its initialisation unreproducible."""

    backend = FakeBackend()

    train(workspace, backend)

    assert backend.seed == 17
    assert backend.calls[:2] == ["seed_all", "create_model_and_optimizer"]


def test_every_epoch_in_the_config_is_run_once_in_order(workspace: dict[str, Path]) -> None:
    """The epoch count is pinned in the config; the engine does not get to reinterpret it."""

    backend = FakeBackend()

    train(workspace, backend)

    assert backend.epochs_run == [1, 2, 3, 4]
    assert backend.evaluated == 4


def test_the_checkpoint_is_chosen_by_the_calibration_loss_alone(
    workspace: dict[str, Path],
) -> None:
    """Training loss falls every epoch here; selection must ignore it completely."""

    backend = FakeBackend(
        training_losses=(4.0, 3.0, 2.0, 1.0), calibration_losses=(0.9, 0.4, 0.5, 0.6)
    )

    result = train(workspace, backend)

    assert result.selected_epoch == 2
    assert result.calibration_loss == pytest.approx(0.4)


def test_a_tie_on_the_calibration_loss_keeps_the_earliest_epoch(
    workspace: dict[str, Path],
) -> None:
    """A later epoch that is no better is more training for no measured gain."""

    backend = FakeBackend(calibration_losses=(0.9, 0.4, 0.4, 0.4))

    assert train(workspace, backend).selected_epoch == 2


def test_the_selected_checkpoint_exists_and_still_hashes_to_what_was_recorded(
    workspace: dict[str, Path],
) -> None:
    """The digest is re-computed from the file, not taken on the backend's word."""

    result = train(workspace, FakeBackend())

    record = json.loads(result.run_record_path.read_text(encoding="utf-8"))
    digest = hashlib.sha256(result.checkpoint_path.read_bytes()).hexdigest()
    assert record["artifacts"]["selected_checkpoint"] == digest


def test_a_checkpoint_that_does_not_match_its_reported_digest_fails_closed(
    workspace: dict[str, Path],
) -> None:
    """A backend that saved one thing and reported another would poison every later check."""

    class LyingBackend(FakeBackend):
        def save_checkpoint(self, model: Any, path: Path, metadata: dict[str, Any]) -> str:
            path.write_bytes(b"the real bytes")
            return "c" * 64

    with pytest.raises(
        ValueError,
        match=r"^the selected checkpoint on disk does not match the digest the backend reported: ",
    ):
        train(workspace, LyingBackend())


def test_a_failed_run_still_leaves_a_record_saying_so(workspace: dict[str, Path]) -> None:
    """A run that vanishes without a trace is indistinguishable from one never started."""

    class BrokenBackend(FakeBackend):
        def run_epoch(self, model: Any, optimizer: Any, manifest: Path, epoch: int) -> float:
            raise RuntimeError("the framework fell over")

    with pytest.raises(RuntimeError, match=r"^the framework fell over$"):
        train(workspace, BrokenBackend())

    record = json.loads((workspace["output"] / "run_record.json").read_text(encoding="utf-8"))
    assert record["status"] == "failed"
    assert record["artifacts"] == {}
    assert record["finished_at_utc"] is not None


def test_the_run_record_ties_the_result_to_its_config_cohort_and_machine(
    workspace: dict[str, Path],
) -> None:
    """Without these hashes a reported number cannot be attributed to anything."""

    result = train(workspace, FakeBackend())

    record = json.loads(result.run_record_path.read_text(encoding="utf-8"))
    assert record["schema_version"] == "bev-calibration-run/v1"
    assert record["run_id"] == "convnextv2_tiny-seed-17"
    assert record["seed"] == 17
    assert record["status"] == "succeeded"
    assert record["commit"] == PROVENANCE["commit"]
    assert record["hardware"] == PROVENANCE["hardware"]
    for key in ("config_sha256", "protocol_sha256", "cohort_manifest_sha256", "lock_sha256"):
        assert len(record[key]) == 64


def test_the_protocol_hash_is_the_perturbation_matrix_the_corrector_is_trained_for(
    workspace: dict[str, Path],
) -> None:
    """A corrector is only meaningful against the fault distribution it was trained on."""

    result = train(workspace, FakeBackend())

    matrix = (workspace["config"].parent.parent / "perturbations" / "formal_v1.yaml").read_bytes()
    record = json.loads(result.run_record_path.read_text(encoding="utf-8"))
    assert record["protocol_sha256"] == hashlib.sha256(matrix).hexdigest()


def test_two_runs_of_the_same_thing_produce_the_same_record_apart_from_its_clock(
    workspace: dict[str, Path], tmp_path: Path
) -> None:
    """Determinism is the property that makes a seed spread mean anything."""

    first = json.loads(train(workspace, FakeBackend()).run_record_path.read_text("utf-8"))
    second = json.loads(
        train(workspace, FakeBackend(), output_dir=tmp_path / "again").run_record_path.read_text(
            "utf-8"
        )
    )

    for key in ("run_id", "config_sha256", "protocol_sha256", "cohort_manifest_sha256", "seed"):
        assert first[key] == second[key]
    assert first["artifacts"] == second["artifacts"]


@pytest.mark.parametrize(
    ("role_key", "role"),
    [
        ("development", "calibration"),
        ("development", "evaluation"),
        ("calibration", "development"),
        ("calibration", "evaluation"),
    ],
)
def test_a_manifest_playing_the_wrong_part_is_refused(
    workspace: dict[str, Path], tmp_path: Path, role_key: str, role: str
) -> None:
    """Handing the evaluation cohort to a trainer is the mistake that voids the study."""

    wrong = write_cohort(tmp_path / "wrong.json", role, "wrong", 20)

    expected = rf"^the {role_key} manifest declares role '{role}'; "
    with pytest.raises(ValueError, match=expected):
        train(workspace, FakeBackend(), **{f"{role_key}_manifest": wrong})


def test_the_two_cohorts_may_not_share_a_log(workspace: dict[str, Path], tmp_path: Path) -> None:
    """Two scenes from one log are the same road minutes apart, not independent samples.

    Sharing a log between training and selection means the checkpoint is chosen on
    something the model has effectively already seen, and the selection stops being
    a check on generalisation at all.
    """

    document = cohort_document("calibration", "cal", 20)
    document["scenes"][0]["log_token"] = "dev-log-0"
    overlapping = tmp_path / "overlap.json"
    overlapping.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ValueError, match=r"^the development and calibration cohorts share a log: "):
        train(workspace, FakeBackend(), calibration_manifest=overlapping)


def test_the_two_cohorts_may_not_share_a_scene(workspace: dict[str, Path], tmp_path: Path) -> None:
    """The blunter version of the same leak, checked separately so the message is clear."""

    document = cohort_document("calibration", "cal", 20)
    document["scenes"][0]["scene_token"] = "dev-scene-0"
    document["scenes"][0]["log_token"] = "cal-log-0"
    overlapping = tmp_path / "overlap.json"
    overlapping.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(
        ValueError, match=r"^the development and calibration cohorts share a scene: "
    ):
        train(workspace, FakeBackend(), calibration_manifest=overlapping)


def test_the_calibration_cohort_must_be_the_twenty_scenes_the_protocol_names(
    workspace: dict[str, Path], tmp_path: Path
) -> None:
    """A truncated manifest would select on less evidence while looking like it did not."""

    short = write_cohort(tmp_path / "short.json", "calibration", "cal", 19)

    with pytest.raises(
        ValueError, match=r"^the calibration cohort must hold exactly 20 scenes, got "
    ):
        train(workspace, FakeBackend(), calibration_manifest=short)


def test_an_unapproved_seed_is_refused(workspace: dict[str, Path]) -> None:
    """The three seeds are pinned in the config; a fourth would not be comparable."""

    with pytest.raises(
        ValueError, match=r"^seed .* is not one of the approved seeds \[17, 42, 73\]$"
    ):
        train(workspace, FakeBackend(), seed=1)


def test_writing_over_a_finished_run_is_refused(workspace: dict[str, Path]) -> None:
    """A run record is evidence, and silently replacing one detaches every claim citing it."""

    train(workspace, FakeBackend())

    with pytest.raises(FileExistsError):
        train(workspace, FakeBackend())


def test_the_committed_config_pins_everything_that_could_change_a_result() -> None:
    """These are the knobs; leaving one unpinned makes two runs incomparable."""

    from bevcalib.training.engine import load_corrector_config

    config = load_corrector_config(COMMITTED_CONFIG)

    assert config.architecture == "convnextv2_tiny"
    assert config.input_channels == 5
    assert config.epochs == 30
    assert config.batch_size == 16
    assert config.optimizer == "adamw"
    assert config.learning_rate == pytest.approx(1e-4)
    assert config.weight_decay == pytest.approx(0.05)
    assert config.schedule == "cosine"
    assert config.seeds == (17, 42, 73)
    assert config.rotation_scale_deg == pytest.approx(2.0)
    assert config.translation_scale_m == pytest.approx(0.2)


def test_a_config_carrying_an_unknown_key_is_refused(tmp_path: Path) -> None:
    """An unrecognised knob is either a typo or a setting nothing reads; both are bugs."""

    from pydantic import ValidationError

    from bevcalib.training.engine import load_corrector_config

    document = yaml.safe_load(COMMITTED_CONFIG.read_text(encoding="utf-8"))
    path = tmp_path / "extra.yaml"
    path.write_text(yaml.safe_dump(document | {"momentum": 0.9}), encoding="utf-8")

    with pytest.raises(ValidationError):
        load_corrector_config(path)


def test_a_calibration_loss_that_is_not_a_number_stops_the_run(
    workspace: dict[str, Path],
) -> None:
    """NaN compares false against everything, so it would silently select nothing."""

    backend = FakeBackend(calibration_losses=(0.9, float("nan"), 0.5, 0.6))

    with pytest.raises(
        ValueError, match=r"^the calibration loss at epoch .* is nan, which cannot select anything$"
    ):
        train(workspace, backend)

    record = json.loads((workspace["output"] / "run_record.json").read_text(encoding="utf-8"))
    assert record["status"] == "failed"
