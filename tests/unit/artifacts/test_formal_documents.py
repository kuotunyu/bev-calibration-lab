"""Semantic refusal tests rehash tampered documents so checks reach the contract."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
from tests.contract.test_formal_artifact_set import RUNS
from tests.unit.metrics.test_summary import run_fixture


class _StopAfterBoundary(Exception):
    pass


def test_aggregate_defaults_to_observed(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import bevcalib.analysis.aggregate as aggregate_module

    calls = []

    def loader(path, *, synthetic_fixture, **_kwargs):
        calls.append((path, synthetic_fixture))
        raise _StopAfterBoundary

    monkeypatch.setattr(aggregate_module, "load_run_set", loader)

    with pytest.raises(_StopAfterBoundary):
        aggregate_module.aggregate_formal_results(tmp_path / "input", tmp_path / "output")

    assert calls == [(tmp_path / "input", False)]


def test_aggregate_constructs_real_observed_identity_from_complete_run_inventory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import bevcalib.analysis.aggregate as aggregate_module
    from bevcalib.artifacts.documents import FormalIdentity

    measurements = SimpleNamespace(model_dump=lambda **_kwargs: {"policy": "fixed"})
    producer = SimpleNamespace(commit="a" * 40, lock_sha256="b" * 64)
    specifications = (
        ("identity", None, None),
        ("classical", None, None),
        ("learned", 17, "1" * 64),
        ("learned", 42, "2" * 64),
        ("learned", 73, "3" * 64),
    )
    runs = tuple(
        SimpleNamespace(
            label=method if seed is None else f"{method}-{seed}",
            marker=SimpleNamespace(
                identity=SimpleNamespace(
                    method=method,
                    seed=seed,
                    checkpoint_sha256=checkpoint,
                    run_id=f"{index + 4:x}" * 64,
                    measurements=measurements,
                    producer=producer,
                )
            ),
            source_complete_sha256=f"{index + 9:x}" * 64,
        )
        for index, (method, seed, checkpoint) in enumerate(specifications)
    )
    manifest = SimpleNamespace(
        protocol_hash="c" * 64,
        manifest_sha256="d" * 64,
        dataset_version="v1.0-trainval",
        scenes=tuple(SimpleNamespace(sample_tokens=(str(index),)) for index in range(30)),
    )
    captured = []

    def identity_boundary(**kwargs):
        captured.append(FormalIdentity(**kwargs))
        raise _StopAfterBoundary

    monkeypatch.setattr(
        aggregate_module, "load_run_set", lambda *_args, **_kwargs: (manifest, runs)
    )
    monkeypatch.setattr(aggregate_module, "FormalIdentity", identity_boundary)

    with pytest.raises(_StopAfterBoundary):
        aggregate_module.aggregate_formal_results(tmp_path / "input", tmp_path / "output")

    assert captured[0].evidence_type == "observed"


@pytest.fixture(scope="module")
def payloads(tmp_path_factory):
    from bevcalib.analysis.aggregate import aggregate_formal_results

    directory = tmp_path_factory.mktemp("formal-documents")
    root, _ = run_fixture(directory, RUNS)
    return aggregate_formal_results(root, directory / "formal", synthetic_fixture=True).model_dump(
        mode="json"
    )


def test_aggregate_exclusion_counts_sum_legal_fixture_rows(payloads) -> None:
    exclusions = payloads["exclusions"]["runs"]["identity"]["x:0.1"]
    assert exclusions["projection_input_points"] == 6
    assert exclusions["within_row_pixel_error_count"] == 4


def test_aggregate_writes_five_documents_beneath_missing_parent(tmp_path: Path) -> None:
    from bevcalib.analysis.aggregate import aggregate_formal_results

    root, _ = run_fixture(tmp_path, RUNS)
    output_dir = tmp_path / "new-parent" / "formal"
    aggregate_formal_results(root, output_dir, synthetic_fixture=True)

    assert {path.name for path in output_dir.iterdir()} == {
        "metrics.json",
        "intervals.json",
        "recovery.json",
        "timing.json",
        "exclusions.json",
    }


def test_aggregate_preserves_higher_is_better_improvement_direction(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import tests.unit.metrics.test_summary as fixture_module

    from bevcalib.analysis.aggregate import aggregate_formal_results
    from bevcalib.artifacts.results import CalibrationResultV2

    original_rows_for = fixture_module.rows_for
    calls = 0

    def rows_with_better_classical_edges(manifest):
        nonlocal calls
        calls += 1
        rows = original_rows_for(manifest)
        if calls != 2:
            return rows
        return tuple(
            CalibrationResultV2.model_validate(row.model_dump() | {"edge_alignment_score": -1.0})
            if row.edge_alignment_score is not None
            else row
            for row in rows
        )

    monkeypatch.setattr(fixture_module, "rows_for", rows_with_better_classical_edges)
    root, _ = run_fixture(tmp_path, RUNS)
    documents = aggregate_formal_results(
        root, tmp_path / "formal", synthetic_fixture=True
    ).model_dump(mode="json")

    comparison = documents["intervals"]["comparisons"]["identity->classical"]["x:0.1"]
    assert comparison["edge_score_px"]["improvement"] > 0


def rehash(body):
    from bevcalib.artifacts.result_documents import digest

    return body | {
        "document_sha256": digest(
            {key: value for key, value in body.items() if key != "document_sha256"}
        )
    }


@pytest.mark.parametrize(
    "metric,value",
    [
        ("rotation_geodesic_deg", -1.0),
        ("pixel_frame_p50_px", -1.0),
        ("edge_score_px", 1.0),
        ("recovery_rate_pct", 101.0),
    ],
)
def test_metrics_enforce_physical_sign_and_rate_units(payloads, metric: str, value: float) -> None:
    from bevcalib.artifacts.documents import MetricsDocument

    body = deepcopy(payloads["metrics"])
    body["runs"]["identity"]["yaw:0"][metric]["value"] = value
    with pytest.raises(ValueError, match="metric value"):
        MetricsDocument.model_validate(rehash(body))


@pytest.mark.parametrize("metric", ["pixel_frame_p50_px", "edge_score_px", "recovery_rate_pct"])
def test_rehashed_improvement_must_match_before_after_in_declared_direction(
    payloads, metric: str
) -> None:
    from bevcalib.artifacts.documents import IntervalsDocument

    body = deepcopy(payloads["intervals"])
    body["comparisons"]["identity->classical"]["yaw:0"][metric]["before"] += (
        1.0 if metric != "edge_score_px" else -1.0
    )
    with pytest.raises(ValueError, match="paired improvement"):
        IntervalsDocument.model_validate(rehash(body))


def test_arithmetic_order_roundoff_is_allowed_without_changing_physical_tolerances() -> None:
    from bevcalib.artifacts.documents import check_improvement
    from bevcalib.artifacts.statistics import PairedEstimate

    estimate = PairedEstimate(
        before=0.1 + 0.2,
        after=0.3,
        improvement=0,
        interval={
            "estimate": 0,
            "low": 0,
            "high": 0,
            "resamples": 5000,
            "seed": 20260831,
            "confidence": 0.95,
        },
        support={
            "total_frames": 1,
            "frames": 1,
            "excluded_frames": 0,
            "total_scenes": 1,
            "scenes": 1,
            "exclusions": {},
        },
        reason=None,
    )
    assert estimate.before - estimate.after != estimate.improvement
    check_improvement("pixel_frame_p50_px", estimate)


@pytest.mark.parametrize("case", ["negative_error", "unavailable_selected"])
def test_timing_statistics_preserve_selection_semantics(payloads, case: str) -> None:
    from bevcalib.artifacts.documents import TimingDocument

    body = deepcopy(payloads["timing"])
    summary = body["offsets"]["0"]
    if case == "negative_error":
        summary["absolute_error_ms"]["mean"] = -1.0
    else:
        summary.update(valid=0, invalid=2, valid_fraction=0.0, reasons={"no_available_lidar": 2})
    with pytest.raises(ValueError, match=r"absolute_error_ms|selection counts"):
        TimingDocument.model_validate(rehash(body))


@pytest.mark.parametrize(
    "case,message",
    [
        ("policy", "policy"),
        ("observed_underfill", "thirty"),
        ("source_missing", "run inventory"),
        ("source_label", "label/checkpoint"),
        ("source_checkpoint", "checkpoint or producer"),
        ("producer", "checkpoint or producer"),
        ("run_missing", "run inventory"),
        ("condition_missing", "condition inventory"),
        ("metric_missing", "metric inventory"),
        ("support", "support denominator"),
        ("comparison_missing", "comparison inventory"),
        ("comparison_condition", "comparison condition"),
        ("comparison_metric", "comparison metric"),
        ("recovery_comparison", "recovery comparison"),
        ("timing_offset", "timing offset"),
        ("timing_request", "timing request"),
        ("timing_counts", "selection counts"),
        ("global_counts", "validity counts"),
        ("recovery_values", "recovery differs"),
        ("timing_values", "timing stress differs"),
        ("operator_support", "exclusions differ"),
        ("paired_support", "exclusions differ"),
    ],
)
def test_rehashed_incoherent_documents_are_refused(payloads, case: str, message: str) -> None:
    from bevcalib.artifacts.documents import FormalArtifactSet

    bodies = deepcopy(payloads)
    identity = bodies["metrics"]["identity"]
    metric = bodies["metrics"]["runs"]["identity"]["yaw:0"]
    comparison = bodies["intervals"]["comparisons"]["identity->classical"]
    timing = bodies["timing"]["offsets"]["0"]
    if case == "policy":
        identity["units"]["translation_norm_cm"] = "metre"
    elif case == "observed_underfill":
        identity["evidence_type"] = "observed"
    elif case == "source_missing":
        identity["source_runs"].pop("learned-17")
    elif case == "source_label":
        identity["source_runs"]["other"] = identity["source_runs"].pop("learned-17")
    elif case == "source_checkpoint":
        identity["source_runs"]["learned-42"]["checkpoint_sha256"] = "a" * 64
    elif case == "producer":
        identity["source_runs"]["learned-42"]["producer_lock_sha256"] = "d" * 64
    elif case == "run_missing":
        bodies["metrics"]["runs"].pop("identity")
    elif case == "condition_missing":
        bodies["metrics"]["runs"]["identity"].pop("yaw:0")
    elif case == "metric_missing":
        metric.pop("rotation_geodesic_deg")
    elif case == "support":
        metric["rotation_geodesic_deg"]["support"].update(total_frames=3, excluded_frames=1)
    elif case == "comparison_missing":
        bodies["intervals"]["comparisons"].pop("identity->classical")
    elif case == "comparison_condition":
        comparison.pop("yaw:0")
    elif case == "comparison_metric":
        comparison["yaw:0"].pop("edge_score_px")
    elif case == "recovery_comparison":
        bodies["recovery"]["comparisons"].pop("identity->classical")
    elif case == "timing_offset":
        bodies["timing"]["offsets"].pop("0")
    elif case == "timing_request":
        timing["requested_offset_ms"] = 50
    elif case == "timing_counts":
        timing["valid_fraction"] = 0.5
    elif case == "global_counts":
        bodies["exclusions"]["runs"]["identity"]["yaw:0"]["invalid"] = 1
    elif case == "recovery_values":
        bodies["recovery"]["runs"]["identity"]["yaw:0"]["value"] = 1.0
    elif case == "timing_values":
        timing["metrics"]["rotation_geodesic_deg"]["value"] = 2.0
    elif case == "operator_support":
        bodies["exclusions"]["runs"]["identity"]["yaw:0"]["operators"]["edge_score_px"].update(
            frames=1, excluded_frames=1
        )
    else:
        bodies["exclusions"]["comparisons"]["identity->classical"]["yaw:0"]["edge_score_px"].update(
            frames=1, excluded_frames=1
        )
    with pytest.raises(ValueError, match=message):
        FormalArtifactSet.model_validate({name: rehash(body) for name, body in bodies.items()})
