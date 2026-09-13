"""External frozen identities must be checked even when artifacts are coherent."""

import hashlib
from pathlib import Path

import pytest

from bevcalib.analysis.policy import ESTIMAND_DESCRIPTION, METRIC_UNITS
from bevcalib.artifacts.documents import FormalIdentity


def expected_payload() -> dict:
    return {
        "schema_version": "bev-fault-study-expectations/v1",
        "evidence_type": "synthetic",
        "protocol_hash": "a" * 64,
        "dataset_manifest_hash": "b" * 64,
        "raw_producer_commit": "c" * 40,
        "raw_producer_lock_sha256": "d" * 64,
        "checkpoints": {
            f"learned-{seed}": hashlib.sha256(f"checkpoint-{seed}".encode()).hexdigest()
            for seed in (17, 42, 73)
        },
    }


def source_identity() -> FormalIdentity:
    sources = {}
    for index, (method, seed) in enumerate(
        (("identity", None), ("classical", None), ("learned", 17), ("learned", 42), ("learned", 73))
    ):
        label = method if seed is None else f"learned-{seed}"
        sources[label] = {
            "method": method,
            "seed": seed,
            "checkpoint_sha256": None
            if seed is None
            else hashlib.sha256(f"checkpoint-{seed}".encode()).hexdigest(),
            "run_identity_sha256": str(index) * 64,
            "source_complete_sha256": str(index + 1) * 64,
            "producer_commit": "c" * 40,
            "producer_lock_sha256": "d" * 64,
        }
    return FormalIdentity.model_validate(
        {
            "protocol_hash": "a" * 64,
            "dataset_manifest_hash": "b" * 64,
            "dataset_version": "v1.0-trainval",
            "evidence_type": "synthetic",
            "measurement_identity_sha256": "e" * 64,
            "source_runs": sources,
            "scene_count": 2,
            "sample_count": 4,
            "estimands": ESTIMAND_DESCRIPTION,
            "units": METRIC_UNITS,
        }
    )


def test_matching_independently_supplied_identity_is_accepted() -> None:
    from bevcalib.artifacts.study_expectations import ExpectedStudy, validate_expected_identity

    assert (
        validate_expected_identity(
            source_identity(), ExpectedStudy.model_validate(expected_payload())
        )
        is None
    )


@pytest.mark.parametrize(
    "field",
    [
        "protocol_hash",
        "dataset_manifest_hash",
        "raw_producer_commit",
        "raw_producer_lock_sha256",
        "checkpoints",
        "evidence_type",
    ],
)
def test_coherent_source_still_refuses_wrong_frozen_expectation(field: str) -> None:
    from bevcalib.artifacts.study_expectations import ExpectedStudy, validate_expected_identity

    expected = expected_payload()
    if field == "checkpoints":
        expected[field]["learned-42"] = "f" * 64
    elif field == "evidence_type":
        expected[field] = "observed"
    else:
        expected[field] = "f" * (40 if field == "raw_producer_commit" else 64)
    identity = source_identity()  # This is an internally valid source, not a broken-schema mock.
    with pytest.raises(ValueError, match="frozen study expectation mismatch"):
        validate_expected_identity(identity, ExpectedStudy.model_validate(expected))


@pytest.mark.parametrize(
    "case", ["missing", "extra", "same-checkpoint", "unknown-field", "bad-hash"]
)
def test_expectation_contract_refuses_ambiguous_inventory(case: str) -> None:
    from bevcalib.artifacts.study_expectations import ExpectedStudy

    expected = expected_payload()
    if case == "missing":
        del expected["checkpoints"]["learned-73"]
    elif case == "extra":
        expected["checkpoints"]["learned-99"] = "f" * 64
    elif case == "same-checkpoint":
        expected["checkpoints"]["learned-73"] = expected["checkpoints"]["learned-17"]
    elif case == "unknown-field":
        expected["accept_any_producer"] = True
    else:
        expected["protocol_hash"] = "not-a-digest"
    with pytest.raises(ValueError):
        ExpectedStudy.model_validate(expected)


def checkpoint_files(tmp_path: Path) -> dict[str, Path]:
    paths = {}
    for seed in (17, 42, 73):
        path = tmp_path / f"seed-{seed}.pt"
        path.write_bytes(f"checkpoint-{seed}".encode())
        paths[f"learned-{seed}"] = path
    return paths


def test_checkpoint_bytes_match_frozen_hashes_without_loading_torch(tmp_path: Path) -> None:
    from bevcalib.artifacts.study_expectations import ExpectedStudy, validate_checkpoint_files

    paths = checkpoint_files(tmp_path)
    expected = ExpectedStudy.model_validate(expected_payload())
    assert validate_checkpoint_files(expected, paths) == expected.checkpoints


@pytest.mark.parametrize("case", ["missing-label", "extra-label", "missing-file", "wrong-bytes"])
def test_checkpoint_paths_do_not_substitute_for_actual_bytes(tmp_path: Path, case: str) -> None:
    from bevcalib.artifacts.study_expectations import ExpectedStudy, validate_checkpoint_files

    paths = checkpoint_files(tmp_path)
    if case == "missing-label":
        del paths["learned-42"]
    elif case == "extra-label":
        paths["learned-99"] = paths["learned-17"]
    elif case == "missing-file":
        paths["learned-73"].unlink()
    else:
        paths["learned-73"].write_bytes(b"changed checkpoint")
    with pytest.raises((ValueError, FileNotFoundError)):
        validate_checkpoint_files(ExpectedStudy.model_validate(expected_payload()), paths)
