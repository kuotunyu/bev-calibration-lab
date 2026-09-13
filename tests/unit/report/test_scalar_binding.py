"""Figures and tables share exact source-snapshot binding, never rounded matching."""

import pytest

from bevcalib.analysis.claims import ClaimV1, ReportScalarBinding


def claim() -> ClaimV1:
    return ClaimV1(
        claim_id="formal.point",
        text="Synthetic value 0.125",
        evidence_type="synthetic",
        protocol_hash="a" * 64,
        dataset_manifest_hash="b" * 64,
        artifact_path="evidence/recovery.json",
        metric_path="/runs/identity/yaw:0/value",
        status="verified",
        report_binding=ReportScalarBinding(expected_summary_sha256="c" * 64, expected_value=0.125),
    )


def test_exact_point_returns_the_verified_claim_identity() -> None:
    from bevcalib.report.scalar_binding import bind_scalar

    source = claim()
    assert (
        bind_scalar(
            "recovery",
            source.metric_path,
            0.125,
            "c" * 64,
            {("recovery", source.metric_path): source},
        )
        == "formal.point"
    )


def test_unavailable_point_stays_unclaimed() -> None:
    from bevcalib.report.scalar_binding import bind_scalar

    assert bind_scalar("metrics", "/missing", None, "c" * 64, {}) is None


@pytest.mark.parametrize("case", ["absent", "draft", "unbound", "digest", "value", "pointer"])
def test_point_refuses_missing_or_mismatched_identity(case: str) -> None:
    from bevcalib.report.scalar_binding import bind_scalar

    source = claim()
    pointer = source.metric_path
    if case == "draft":
        source = source.model_copy(update={"status": "draft"})
    elif case == "unbound":
        source = source.model_copy(update={"report_binding": None})
    elif case == "digest":
        source = source.model_copy(
            update={
                "report_binding": ReportScalarBinding(
                    expected_summary_sha256="d" * 64, expected_value=0.125
                )
            }
        )
    elif case == "value":
        source = source.model_copy(
            update={
                "report_binding": ReportScalarBinding(
                    expected_summary_sha256="c" * 64, expected_value=0.125000000001
                )
            }
        )
    elif case == "pointer":
        source = source.model_copy(update={"metric_path": "/wrong"})
    indexed = {} if case == "absent" else {("recovery", pointer): source}
    with pytest.raises(
        ValueError, match=r"unclaimed displayed formal scalar|differs from verified source snapshot"
    ):
        bind_scalar("recovery", pointer, 0.125, "c" * 64, indexed)


def test_integer_and_equivalent_float_bind_without_rounding() -> None:
    from bevcalib.report.scalar_binding import bind_scalar

    source = claim().model_copy(
        update={
            "report_binding": ReportScalarBinding(
                expected_summary_sha256="c" * 64, expected_value=1
            )
        }
    )
    assert (
        bind_scalar(
            "recovery",
            source.metric_path,
            1.0,
            "c" * 64,
            {("recovery", source.metric_path): source},
        )
        == "formal.point"
    )
