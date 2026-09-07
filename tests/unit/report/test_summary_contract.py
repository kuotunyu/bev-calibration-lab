"""Portable contracts refuse internally inconsistent or mislabeled aggregates."""

from copy import deepcopy
from pathlib import Path

import pytest
from tests.unit.report.test_builder import summary_document as summary_document

from bevcalib.artifacts.result_documents import digest
from bevcalib.artifacts.summary import CalibrationSummaryV1, load_safe_summary, write_safe_summary


def complete_summary(document):
    result = deepcopy(document)
    for seed in (17, 42, 73):
        run = deepcopy(result["runs"]["classical"])
        run.update(method="learned", seed=seed, checkpoint_sha256=f"{seed:064x}")
        result["runs"][f"learned-{seed}"] = run
    result["document_sha256"] = digest({k: v for k, v in result.items() if k != "document_sha256"})
    return result


def test_complete_observed_contract_round_trips_with_signed_values(
    summary_document, tmp_path: Path
) -> None:
    document = complete_summary(summary_document)
    document["evidence_type"] = "observed"
    document["document_sha256"] = digest(
        {k: v for k, v in document.items() if k != "document_sha256"}
    )
    path = write_safe_summary(document, tmp_path / "safe/summary.json")
    actual = load_safe_summary(path)
    assert actual.evidence_type == "observed"
    assert actual.runs["identity"].conditions["yaw:0"].pose.rotation_rpy_error_deg[0].mean < 0
    assert path.read_bytes().endswith(b"\n") and b"\r\n" not in path.read_bytes()


@pytest.mark.parametrize(
    "case",
    [
        "empty_measured",
        "measured_null",
        "valid_sum",
        "valid_rate",
        "valid_reasons",
        "ground_sum",
        "ground_reasons",
        "pixel_count",
        "recovery_count",
        "recovery_rate",
        "missing_bin",
        "method_seed",
        "method_checkpoint",
        "learned_seed",
        "learned_checkpoint",
        "inventory",
        "condition_label",
        "timing_recovery",
        "policy",
        "missing_run",
        "duplicate_run",
        "run_label",
        "checkpoint_reuse",
        "hash",
    ],
)
def test_safe_contract_refuses_inconsistent_provenance_or_denominators(
    summary_document, case: str
) -> None:
    document = complete_summary(summary_document)
    run = document["runs"]["identity"]
    row = run["conditions"]["yaw:0"]
    if case == "empty_measured":
        row["edge_alignment_score"]["count"] = 0
    elif case == "measured_null":
        row["edge_alignment_score"]["mean"] = None
    elif case == "valid_sum":
        row["validity"]["total"] += 1
    elif case == "valid_rate":
        row["validity"]["invalid_rate"] = 0.5
    elif case == "valid_reasons":
        row["validity"]["reasons"] = {"wrong": 1}
    elif case == "ground_sum":
        row["ground_contact_by_range"]["80+"]["total"] += 1
    elif case == "ground_reasons":
        row["ground_contact_by_range"]["80+"]["reasons"] = {"wrong": 1}
    elif case == "pixel_count":
        row["pixel_error_px"]["projection_count"] = 0
    elif case == "recovery_count":
        row["pose_recovery"]["recovered"] = 999
    elif case == "recovery_rate":
        row["pose_recovery"]["rate"] = None
    elif case == "missing_bin":
        row["ground_contact_by_range"].pop("80+")
    elif case == "method_seed":
        run["seed"] = 17
    elif case == "method_checkpoint":
        run["checkpoint_sha256"] = "a" * 64
    elif case == "learned_seed":
        document["runs"]["learned-17"]["seed"] = None
    elif case == "learned_checkpoint":
        document["runs"]["learned-17"]["checkpoint_sha256"] = None
    elif case == "inventory":
        run["conditions"].pop("time:0")
    elif case == "condition_label":
        row["fault_level"] = 1
    elif case == "timing_recovery":
        run["conditions"]["time:0"]["pose_recovery"] = deepcopy(row["pose_recovery"])
    elif case == "policy":
        document["measurement_policy"] = {}
    elif case == "missing_run":
        document["runs"].pop("classical")
    elif case == "duplicate_run":
        document["runs"]["extra"] = deepcopy(run)
    elif case == "run_label":
        document["runs"]["wrong"] = document["runs"].pop("identity")
    elif case == "checkpoint_reuse":
        document["runs"]["learned-42"]["checkpoint_sha256"] = document["runs"]["learned-17"][
            "checkpoint_sha256"
        ]
    else:
        document["document_sha256"] = "c" * 64
    with pytest.raises(ValueError):
        CalibrationSummaryV1.model_validate(document)


def test_unmeasurable_pose_recovery_remains_null(summary_document) -> None:
    document = deepcopy(summary_document)
    document["runs"]["identity"]["conditions"]["yaw:0"]["pose_recovery"] = {
        "valid": 0,
        "recovered": 0,
        "rate": None,
    }
    document["document_sha256"] = digest(
        {k: v for k, v in document.items() if k != "document_sha256"}
    )
    assert (
        CalibrationSummaryV1.model_validate(document)
        .runs["identity"]
        .conditions["yaw:0"]
        .pose_recovery.rate
        is None
    )
