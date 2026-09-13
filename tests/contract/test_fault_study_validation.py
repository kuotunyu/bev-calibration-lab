"""Join runtime, external expectations, formal claims and raw-scene validation.

The small tests reduce the publication mapping and stub the separately tested
runtime probe; other file validators remain real. The full-mapping test is
retained separately for resource-admitted execution and also stubs that probe.
"""

import hashlib
import importlib
import json
import shutil

import pytest
from tests.unit.artifacts.test_result_documents import inputs
from tests.unit.metrics.test_summary import run_fixture

from bevcalib.analysis.aggregate import aggregate_formal_results
from bevcalib.analysis.formal_claims import PublicationRow, generate_formal_claims


@pytest.fixture(scope="module")
def frozen_fixture(tmp_path_factory):
    root = tmp_path_factory.mktemp("fault-study-joined")
    identity, manifest = inputs()
    checkpoints = {}
    for seed in (17, 42, 73):
        data = f"independent-checkpoint-{seed}".encode()
        (root / f"checkpoint-{seed}.bin").write_bytes(data)
        checkpoints[f"learned-{seed}"] = hashlib.sha256(data).hexdigest()
    expected = {
        "schema_version": "bev-fault-study-validation-expectations/v1",
        "study": {
            "schema_version": "bev-fault-study-expectations/v1",
            "evidence_type": "synthetic",
            "protocol_hash": manifest.protocol_hash,
            "dataset_manifest_hash": manifest.manifest_sha256,
            "raw_producer_commit": identity.producer.commit,
            "raw_producer_lock_sha256": identity.producer.lock_sha256,
            "checkpoints": checkpoints,
        },
        "validator": {
            "schema_version": "bev-validator-runtime-expectations/v1",
            "python_version": "3.12.13",
            "commit": "c" * 40,
            "lock_sha256": "d" * 64,
            "source_sha256": "e" * 64,
        },
    }
    (root / "expected.json").write_text(json.dumps(expected), encoding="utf-8")
    specs = (
        ("identity", None, None),
        ("classical", None, None),
        *(("learned", seed, checkpoints[f"learned-{seed}"]) for seed in (17, 42, 73)),
    )
    raw, _ = run_fixture(root, specs)
    aggregate_formal_results(raw, root / "evidence", synthetic_fixture=True)
    return root


@pytest.fixture
def study(frozen_fixture, tmp_path, monkeypatch):
    root = tmp_path / "study"
    shutil.copytree(frozen_fixture, root)
    api = importlib.import_module("bevcalib.artifacts.fault_study")
    calls = []

    def runtime(expected):
        calls.append(expected)
        return expected.model_dump(exclude={"schema_version"}) | {"executable": "private-runtime"}

    monkeypatch.setattr(api, "validate_validator_runtime", runtime)
    return api, root, calls


def small_rows(artifacts):
    value = artifacts.metrics.runs["identity"]["yaw:0"]["recovery_rate_pct"].value
    yield PublicationRow(
        "metrics",
        "fixture recovery",
        "percent",
        (("value", "/runs/identity/yaw:0/recovery_rate_pct/value", value),),
    )


def claims_for(root, monkeypatch, *, full=False):
    if not full:
        from bevcalib.analysis import formal_claims

        monkeypatch.setattr(formal_claims, "publication_rows", small_rows)
        monkeypatch.setattr(
            importlib.import_module("bevcalib.artifacts.fault_study"),
            "publication_rows",
            small_rows,
        )
    return generate_formal_claims(root / "evidence", root / "claims.yaml", repository_root=root)


def validate(api, root):
    return api.validate_fault_study(
        root / "expected.json",
        root / "evidence",
        root / "claims.yaml",
        root / "private",
        {f"learned-{seed}": root / f"checkpoint-{seed}.bin" for seed in (17, 42, 73)},
        repository_root=root,
    )


def test_joined_file_validators_and_private_free_receipt(study, monkeypatch):
    api, root, calls = study
    claims_for(root, monkeypatch)
    result = validate(api, root)
    assert result["status"] == "validated-inputs"
    assert result["evidence_type"] == "synthetic"
    assert result["validated_scene_files"] == 5
    assert len(result["documents"]) == 5
    assert len(result["checkpoints"]) == 3
    assert len(calls) == 2
    assert (
        result["expectations_sha256"]
        == hashlib.sha256((root / "expected.json").read_bytes()).hexdigest()
    )
    serialized = json.dumps(result)
    assert "private-runtime" not in serialized
    assert "eval-scene" not in serialized
    assert result["validator"]["commit"] == "c" * 40


def test_full_publication_mapping_and_all_source_checks(study, monkeypatch):
    api, root, _ = study
    claims_for(root, monkeypatch, full=True)
    result = validate(api, root)
    assert result["validated_scene_files"] == 5
    assert result["validated_scalar_bindings"] > 1000


@pytest.mark.parametrize(
    "case,match",
    [
        ("expectation", "frozen study expectation"),
        ("checkpoint", "checkpoint bytes"),
        ("missing_document", "file inventory"),
        ("raw_scene", "hash mismatch"),
        ("missing_seed", "run inventory"),
        ("missing_claim", "nonempty"),
    ],
)
def test_joined_workflow_refuses_bad_input(study, monkeypatch, case, match):
    api, root, _ = study
    claims_for(root, monkeypatch)
    if case == "expectation":
        path = root / "expected.json"
        payload = json.loads(path.read_bytes())
        payload["study"]["protocol_hash"] = "f" * 64
        path.write_text(json.dumps(payload), encoding="utf-8")
    elif case == "checkpoint":
        (root / "checkpoint-42.bin").write_bytes(b"wrong bytes")
    elif case == "missing_document":
        (root / "evidence/timing.json").unlink()
    elif case == "raw_scene":
        path = next((root / "private").glob("classical-*/scene-*.json"))
        payload = json.loads(path.read_bytes())
        payload["document_sha256"] = "f" * 64
        path.write_text(json.dumps(payload), encoding="utf-8")
    elif case == "missing_seed":
        shutil.rmtree(next((root / "private").glob("learned-*")))
    else:
        import yaml

        path = root / "claims.yaml"
        registry = yaml.safe_load(path.read_text(encoding="utf-8"))
        registry["claims"] = []
        path.write_text(yaml.safe_dump(registry), encoding="utf-8")
    with pytest.raises((ValueError, FileNotFoundError), match=match):
        validate(api, root)


def test_runtime_failure_stops_before_reading_study(study, monkeypatch):
    api, root, _ = study

    def fail(expected):
        raise ValueError("runtime mismatch")

    monkeypatch.setattr(api, "validate_validator_runtime", fail)
    with pytest.raises(ValueError, match="runtime mismatch"):
        validate(api, root)  # No claims file exists; it must not be read first.


def test_expectations_changed_during_validation_refused(study, monkeypatch):
    api, root, _ = study
    claims_for(root, monkeypatch)
    original = api.validate_checkpoint_files

    def changed(expected, paths):
        result = original(expected, paths)
        with (root / "expected.json").open("ab") as stream:
            stream.write(b"\n")
        return result

    monkeypatch.setattr(api, "validate_checkpoint_files", changed)
    with pytest.raises(ValueError, match="expectations changed"):
        validate(api, root)


@pytest.mark.parametrize("change", ["manifest", "measurement", "source_marker"])
def test_raw_sources_reject_valid_but_different_formal_identity(study, change):
    from bevcalib.artifacts.documents import FormalIdentity, load_formal_artifact_set

    api, root, _ = study
    body = load_formal_artifact_set(root / "evidence").metrics.identity.model_dump()
    if change == "manifest":
        body["dataset_manifest_hash"] = "f" * 64
    elif change == "measurement":
        body["measurement_identity_sha256"] = "f" * 64
    else:
        body["source_runs"]["learned-17"]["source_complete_sha256"] = "f" * 64
    identity = FormalIdentity.model_validate(body)
    with pytest.raises(ValueError, match=r"raw source|raw run identity"):
        api.validate_raw_sources(identity, root / "private")


def test_raw_marker_changed_after_scene_read_refused(study, monkeypatch):
    from bevcalib.artifacts.documents import load_formal_artifact_set
    from bevcalib.artifacts.result_sets import VerifiedRun

    api, root, _ = study
    original = VerifiedRun.scene_rows

    def changed(run, manifest, scene):
        rows = original(run, manifest, scene)
        with (run.directory / "run_complete.json").open("ab") as stream:
            stream.write(b"\n")
        return rows

    monkeypatch.setattr(VerifiedRun, "scene_rows", changed)
    identity = load_formal_artifact_set(root / "evidence").metrics.identity
    with pytest.raises(ValueError, match="raw run marker changed"):
        api.validate_raw_sources(identity, root / "private")


def test_raw_manifest_changed_after_loading_refused(study, monkeypatch):
    from bevcalib.artifacts.documents import load_formal_artifact_set
    from bevcalib.artifacts.result_sets import VerifiedRun

    api, root, _ = study
    original = VerifiedRun.scene_rows

    def changed(run, manifest, scene):
        rows = original(run, manifest, scene)
        with (root / "private/evaluation_manifest.json").open("ab") as stream:
            stream.write(b"\n")
        return rows

    monkeypatch.setattr(VerifiedRun, "scene_rows", changed)
    identity = load_formal_artifact_set(root / "evidence").metrics.identity
    with pytest.raises(ValueError, match="raw manifest changed"):
        api.validate_raw_sources(identity, root / "private")


def test_earlier_run_marker_changed_during_later_run_refused(study, monkeypatch):
    from bevcalib.artifacts.documents import load_formal_artifact_set
    from bevcalib.artifacts.result_sets import VerifiedRun

    api, root, _ = study
    first = sorted(path for path in (root / "private").iterdir() if path.is_dir())[0]
    original = VerifiedRun.scene_rows

    def changed(run, manifest, scene):
        rows = original(run, manifest, scene)
        if run.directory != first:
            with (first / "run_complete.json").open("ab") as stream:
                stream.write(b"\n")
        return rows

    monkeypatch.setattr(VerifiedRun, "scene_rows", changed)
    identity = load_formal_artifact_set(root / "evidence").metrics.identity
    with pytest.raises(ValueError, match="raw run marker changed"):
        api.validate_raw_sources(identity, root / "private")


def test_unavailable_cell_is_not_counted_as_verified_scalar(study, monkeypatch):
    api, root, _ = study
    claims_for(root, monkeypatch)

    def rows(artifacts):
        yield from small_rows(artifacts)
        value = artifacts.metrics.runs["identity"]["yaw:0"]["bev_frame_mean_m/80+"].value
        assert value is None
        yield PublicationRow(
            "metrics",
            "unavailable fixture bin",
            "m",
            (("value", "/runs/identity/yaw:0/bev_frame_mean_m~180+/value", value),),
        )

    monkeypatch.setattr(api, "publication_rows", rows)
    assert validate(api, root)["validated_scalar_bindings"] == 1
