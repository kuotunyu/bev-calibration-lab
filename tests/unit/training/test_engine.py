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
    from bevcalib.cohort.protocol import resolve_protocol

    scenes = [
        {
            "scene_token": f"{prefix}-scene-{index}",
            "log_token": f"{prefix}-log-{index}",
            "sample_tokens": (f"{prefix}-sample-{index}-0", f"{prefix}-sample-{index}-1"),
            "location": "boston-seaport",
            "official_split": "val" if role == "evaluation" else "train",
            "camera_sample_data_tokens": (
                f"{prefix}-camera-{index}-0",
                f"{prefix}-camera-{index}-1",
            ),
            "lidar_sample_data_tokens": (f"{prefix}-lidar-{index}-0", f"{prefix}-lidar-{index}-1"),
            "sample_timestamps": (100, 200),
            "camera_timestamps": (101, 201),
            "lidar_timestamps": (98, 198),
        }
        for index in range(scene_count)
    ]
    count = {"development": 100, "calibration": 20, "evaluation": 30}[role]
    document = {
        "schema_version": "bev-calibration-cohort/v2",
        "role": role,
        "scenes": scenes,
        "dataset_version": "v1.0-trainval",
        "protocol_hash": resolve_protocol(
            REPO_ROOT / "configs/protocols/nuscenes_calibration_v1.yaml"
        ).protocol_hash,
        "allocation": [
            {
                "location": loc,
                "requested": count if i == 0 else 0,
                "available": scene_count if i == 0 else 0,
                "available_logs": scene_count if i == 0 else 0,
                "selected": scene_count if i == 0 else 0,
                "shortage_reason": "insufficient_scenes"
                if i == 0 and scene_count < count
                else None,
            }
            for i, loc in enumerate(
                (
                    "boston-seaport",
                    "singapore-hollandvillage",
                    "singapore-onenorth",
                    "singapore-queenstown",
                )
            )
        ],
    }
    payload = json.dumps(document, sort_keys=True, separators=(",", ":")).encode()
    return document | {"manifest_sha256": hashlib.sha256(payload).hexdigest()}


def rehash(document: dict[str, Any]) -> None:
    payload = json.dumps(
        {k: v for k, v in document.items() if k != "manifest_sha256"},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    document["manifest_sha256"] = hashlib.sha256(payload).hexdigest()


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
    shutil.copytree(REPO_ROOT / "configs/protocols", configs / "protocols")
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


def test_legacy_manifest_is_refused_before_backend(workspace: dict[str, Path]) -> None:
    document = {
        "schema_version": "bev-calibration-cohort/v1",
        "role": "development",
        "scenes": [],
        "manifest_sha256": "a" * 64,
    }
    rehash(document)
    workspace["development"].write_text(json.dumps(document), encoding="utf-8")
    backend = FakeBackend()
    with pytest.raises(ValueError, match="legacy"):
        train(workspace, backend)
    assert backend.calls == []


def test_changed_scene_with_stale_digest_is_refused_before_backend(
    workspace: dict[str, Path],
) -> None:
    path = workspace["development"]
    document = json.loads(path.read_text(encoding="utf-8"))
    document["scenes"][0]["scene_token"] = "synthetic-tampered-scene"
    path.write_text(json.dumps(document), encoding="utf-8")
    backend = FakeBackend()
    with pytest.raises(ValueError, match="hash"):
        train(workspace, backend)
    assert backend.calls == []


@pytest.mark.parametrize(
    "mutation,match",
    [
        ("protocol", "protocol hash"),
        ("dataset", "dataset version"),
        ("logs", "20 distinct logs"),
        ("split", "official split"),
    ],
)
def test_validly_rehashed_wrong_provenance_is_refused_before_backend(
    workspace: dict[str, Path], mutation: str, match: str
) -> None:
    path = workspace["calibration"]
    document = json.loads(path.read_text(encoding="utf-8"))
    if mutation == "protocol":
        document["protocol_hash"] = "f" * 64
    elif mutation == "dataset":
        document["dataset_version"] = "v1.0-mini"
    elif mutation == "logs":
        document["scenes"][1]["log_token"] = document["scenes"][0]["log_token"]
    else:
        for scene in document["scenes"]:
            scene["official_split"] = "val"
    rehash(document)
    path.write_text(json.dumps(document), encoding="utf-8")
    backend = FakeBackend()
    with pytest.raises(ValueError, match=match):
        train(workspace, backend)
    assert backend.calls == []


def test_sample_leakage_is_refused_before_backend(workspace: dict[str, Path]) -> None:
    path = workspace["calibration"]
    document = json.loads(path.read_text(encoding="utf-8"))
    document["scenes"][0]["sample_tokens"][0] = "dev-sample-0-0"
    rehash(document)
    path.write_text(json.dumps(document), encoding="utf-8")
    backend = FakeBackend()
    with pytest.raises(ValueError, match="share a sample"):
        train(workspace, backend)
    assert backend.calls == []


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


def test_the_protocol_hash_binds_the_resolved_dataset_and_cohort_contract(
    workspace: dict[str, Path],
) -> None:
    """A corrector is only meaningful against the fault distribution it was trained on."""

    result = train(workspace, FakeBackend())

    from bevcalib.cohort.protocol import resolve_protocol

    resolved = resolve_protocol(
        workspace["config"].parent.parent / "protocols" / "nuscenes_calibration_v1.yaml"
    )
    record = json.loads(result.run_record_path.read_text(encoding="utf-8"))
    assert record["protocol_sha256"] == resolved.protocol_hash


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

    wrong = write_cohort(
        tmp_path / "wrong.json",
        role,
        "wrong",
        {"development": 100, "calibration": 20, "evaluation": 30}[role],
    )

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
    rehash(document)
    overlapping.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ValueError, match=r"^the development and calibration cohorts share a log: "):
        train(workspace, FakeBackend(), calibration_manifest=overlapping)


def test_the_two_cohorts_may_not_share_a_scene(workspace: dict[str, Path], tmp_path: Path) -> None:
    """The blunter version of the same leak, checked separately so the message is clear."""

    document = cohort_document("calibration", "cal", 20)
    document["scenes"][0]["scene_token"] = "dev-scene-0"
    document["scenes"][0]["log_token"] = "cal-log-0"
    overlapping = tmp_path / "overlap.json"
    rehash(document)
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
    assert config.input_height == 448
    assert config.input_width == 800
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


@pytest.mark.parametrize("field", ["input_height", "input_width"])
@pytest.mark.parametrize("value", [0, -32, 450])
def test_input_size_rejects_nonpositive_or_non_stride_values(
    tmp_path: Path, field: str, value: int
) -> None:
    from bevcalib.training.engine import load_corrector_config

    document = yaml.safe_load(COMMITTED_CONFIG.read_text(encoding="utf-8"))
    assert document["input_height"] == 448 and document["input_width"] == 800
    document[field] = value
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    with pytest.raises(ValueError):
        load_corrector_config(path)


@pytest.mark.parametrize("change", ["modified", "missing", "dataset"])
def test_resolved_protocol_drift_fails_before_backend(
    workspace: dict[str, Path], change: str
) -> None:
    matrix = workspace["config"].parent.parent / "perturbations/formal_v1.yaml"
    if change == "modified":
        matrix.write_bytes(matrix.read_bytes() + b"\n# synthetic drift\n")
    elif change == "missing":
        matrix.unlink()
    else:
        path = workspace["config"].parent.parent / "protocols/nuscenes_calibration_v1.yaml"
        doc = yaml.safe_load(path.read_text())
        doc["dataset"]["version"] = "v1.0-mini"
        path.write_text(yaml.safe_dump(doc))
    backend = FakeBackend()
    with pytest.raises((ValueError, FileNotFoundError)):
        train(workspace, backend)
    assert backend.calls == []


@pytest.mark.parametrize(
    "field,match",
    [
        ("camera_sample_data_tokens", "camera sample-data"),
        ("lidar_sample_data_tokens", "LiDAR sample-data"),
    ],
)
def test_sensor_identifier_leakage_is_refused(
    workspace: dict[str, Path], field: str, match: str
) -> None:
    path = workspace["calibration"]
    doc = json.loads(path.read_text())
    dev = json.loads(workspace["development"].read_text())
    doc["scenes"][0][field][0] = dev["scenes"][0][field][0]
    rehash(doc)
    path.write_text(json.dumps(doc))
    backend = FakeBackend()
    with pytest.raises(ValueError, match=match):
        train(workspace, backend)
    assert backend.calls == []


@pytest.mark.parametrize("version", ["v1.0-mini", "synthetic-unknown-version"])
def test_matching_rehashed_inputs_cannot_authorize_an_unsupported_dataset(
    workspace: dict[str, Path], version: str
) -> None:
    """Integrity cannot turn an unsupported formal dataset into an approved one."""
    from bevcalib.cohort.manifest import CohortManifestV2

    root = workspace["config"].parent.parent
    protocol_path = root / "protocols/nuscenes_calibration_v1.yaml"
    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    protocol["dataset"]["version"] = version
    protocol_path.write_text(yaml.safe_dump(protocol), encoding="utf-8")
    identity = {
        "schema_version": "bev-calibration-protocol-identity/v1",
        "protocol": protocol,
        "perturbations_sha256": hashlib.sha256(
            (root / "perturbations/formal_v1.yaml").read_bytes()
        ).hexdigest(),
    }
    protocol_hash = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    for role in ("development", "calibration"):
        path = workspace[role]
        doc = json.loads(path.read_text(encoding="utf-8"))
        doc["dataset_version"] = version
        doc["protocol_hash"] = protocol_hash
        rehash(doc)
        CohortManifestV2.model_validate(doc)  # Each input is internally valid and hash-bound.
        path.write_text(json.dumps(doc), encoding="utf-8")
    backend = FakeBackend()
    with pytest.raises(ValueError, match=r"v1\.0-trainval"):
        train(workspace, backend)
    assert backend.calls == []
    assert not workspace["output"].exists()


@pytest.mark.parametrize(
    "development_field,calibration_field",
    [
        ("camera_sample_data_tokens", "lidar_sample_data_tokens"),
        ("lidar_sample_data_tokens", "camera_sample_data_tokens"),
        ("sample_tokens", "camera_sample_data_tokens"),
        ("sample_tokens", "lidar_sample_data_tokens"),
        ("camera_sample_data_tokens", "sample_tokens"),
        ("lidar_sample_data_tokens", "sample_tokens"),
    ],
)
def test_rehashed_cross_field_identifier_collisions_fail_before_backend(
    workspace: dict[str, Path], development_field: str, calibration_field: str
) -> None:
    from bevcalib.cohort.manifest import CohortManifestV2

    dev = json.loads(workspace["development"].read_text(encoding="utf-8"))
    path = workspace["calibration"]
    cal = json.loads(path.read_text(encoding="utf-8"))
    cal["scenes"][0][calibration_field][0] = dev["scenes"][0][development_field][0]
    rehash(cal)
    CohortManifestV2.model_validate(dev)
    CohortManifestV2.model_validate(cal)
    path.write_text(json.dumps(cal), encoding="utf-8")
    backend = FakeBackend()
    with pytest.raises(ValueError, match="identifier belongs to multiple scenes"):
        train(workspace, backend)
    assert backend.calls == []
    assert not workspace["output"].exists()
