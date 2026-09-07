"""The experiment identity binds the resolved protocol and its referenced bytes."""

import hashlib
import json
import shutil

import pytest
import yaml
from tests.unit.training.test_engine import REPO_ROOT


def test_protocol_identity_uses_relative_reference_and_binds_transitive_content(tmp_path):
    from bevcalib.cohort.protocol import resolve_protocol

    shutil.copytree(REPO_ROOT / "configs", tmp_path / "configs")
    path = tmp_path / "configs/protocols/nuscenes_calibration_v1.yaml"
    resolved = resolve_protocol(path)
    matrix = tmp_path / "configs/perturbations/formal_v1.yaml"
    body = {
        "schema_version": "bev-calibration-protocol-identity/v1",
        "protocol": yaml.safe_load(path.read_text()),
        "perturbations_sha256": hashlib.sha256(matrix.read_bytes()).hexdigest(),
    }
    assert (
        resolved.protocol_hash
        == hashlib.sha256(
            json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
    )
    assert resolved.dataset_version == "v1.0-trainval"
    assert resolved.perturbations_path == matrix.resolve()
    matrix.write_bytes(matrix.read_bytes() + b"\n# synthetic drift\n")
    assert resolve_protocol(path).protocol_hash != resolved.protocol_hash
    matrix.unlink()
    with pytest.raises(FileNotFoundError):
        resolve_protocol(path)


@pytest.mark.parametrize(
    "section,key,value",
    [
        (None, "schema_version", "unknown"),
        ("dataset", "keyframes_only", False),
        ("dataset", "version", ""),
        ("cohort", "development", 99),
        ("cohort", "calibration", 19),
        ("cohort", "evaluation", 29),
        ("cohort", "location_stratified", False),
        ("cohort", "log_disjoint", False),
        ("bootstrap", "resamples", 1),
        ("bootstrap", "seed", 1),
    ],
)
def test_protocol_contract_rejects_unsupported_changes(tmp_path, section, key, value):
    from bevcalib.cohort.protocol import resolve_protocol

    doc = yaml.safe_load((REPO_ROOT / "configs/protocols/nuscenes_calibration_v1.yaml").read_text())
    (doc if section is None else doc[section])[key] = value
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(doc))
    with pytest.raises(ValueError):
        resolve_protocol(path)


@pytest.mark.parametrize("version", ["v1.0-mini", "synthetic-unknown-version"])
def test_formal_resolver_refuses_nonempty_unsupported_dataset(tmp_path, version):
    from bevcalib.cohort.protocol import resolve_protocol

    shutil.copytree(REPO_ROOT / "configs", tmp_path / "configs")
    path = tmp_path / "configs/protocols/nuscenes_calibration_v1.yaml"
    doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    doc["dataset"]["version"] = version
    path.write_text(yaml.safe_dump(doc), encoding="utf-8")
    with pytest.raises(ValueError, match=r"v1\.0-trainval"):
        resolve_protocol(path)


@pytest.mark.parametrize(
    "field,value",
    [
        ("rotation_single_axis_deg", [-3.0, 0.0, 3.0]),
        ("translation_single_axis_m", [-0.3, 0.0, 0.3]),
        ("timing_offset_ms", [-300, 0, 300]),
        ("timing_max_selection_error_ms", 26.0),
        ("learned_training_rotation_bound_deg", 3.0),
        ("learned_training_translation_bound_m", 0.3),
        ("recovery_rotation_threshold_deg", 0.5),
        ("recovery_translation_threshold_m", 0.1),
    ],
)
def test_runtime_refuses_schema_valid_matrix_drift(tmp_path, field, value):
    from bevcalib.cohort.protocol import resolve_protocol

    shutil.copytree(REPO_ROOT / "configs", tmp_path / "configs")
    matrix = tmp_path / "configs/perturbations/formal_v1.yaml"
    body = yaml.safe_load(matrix.read_text(encoding="utf-8"))
    body[field] = value
    matrix.write_text(yaml.safe_dump(body), encoding="utf-8")
    with pytest.raises(ValueError, match="supported compiled V1"):
        resolve_protocol(tmp_path / "configs/protocols/nuscenes_calibration_v1.yaml")


def test_matrix_key_order_changes_identity_but_not_supported_values(tmp_path):
    from bevcalib.cohort.protocol import resolve_protocol

    shutil.copytree(REPO_ROOT / "configs", tmp_path / "configs")
    protocol = tmp_path / "configs/protocols/nuscenes_calibration_v1.yaml"
    original = resolve_protocol(protocol)
    matrix = tmp_path / "configs/perturbations/formal_v1.yaml"
    body = yaml.safe_load(matrix.read_text(encoding="utf-8"))
    matrix.write_text(
        yaml.safe_dump(dict(reversed(list(body.items()))), sort_keys=False), encoding="utf-8"
    )
    assert resolve_protocol(protocol).protocol_hash != original.protocol_hash
