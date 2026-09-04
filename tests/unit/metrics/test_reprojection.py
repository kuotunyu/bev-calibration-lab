"""Contracts for summarising reprojection error, and for saying what was excluded.

Two rules run through all of it. An aggregate over nothing is `None` rather than
zero, because zero pixels of error is the best possible score and an empty set is
not a good result. And the counts of what was excluded, with the reasons, travel
with every summary: a rate computed over an unknown denominator is not a rate.
"""

from __future__ import annotations

import numpy as np
import pytest

from bevcalib.artifacts.results import CalibrationFaultModel, CalibrationResultV1

ZERO_FAULT = CalibrationFaultModel(
    rotation_rpy_deg=(0.0, 0.0, 0.0),
    translation_xyz_m=(0.0, 0.0, 0.0),
    requested_time_offset_ms=0,
)


def result(
    valid: bool = True, reason: str | None = None, **overrides: object
) -> CalibrationResultV1:
    body: dict[str, object] = {
        "schema_version": "bev-calibration-result/v1",
        "sample_token": "sample-0",
        "scene_token": "scene-0",
        "fault": ZERO_FAULT,
        "estimate": ZERO_FAULT,
        "rotation_geodesic_error_deg": 0.1,
        "translation_error_m": 0.01,
        "pixel_error_median": 2.0,
        "pixel_error_p90": 5.0,
        "edge_alignment_score": 1.5,
        "bev_ground_contact_error_m": 0.2,
        "valid": valid,
        "invalid_reason": reason,
    }
    return CalibrationResultV1.model_validate(body | overrides)


@pytest.mark.parametrize(
    ("range_m", "expected"),
    [
        (0.0, "0-10"),
        (9.999, "0-10"),
        (10.0, "10-20"),
        (19.999, "10-20"),
        (20.0, "20-40"),
        (39.999, "20-40"),
        (40.0, "40-80"),
        (79.999, "40-80"),
        (80.0, "80+"),
        (1000.0, "80+"),
    ],
)
def test_each_range_bin_owns_its_lower_edge_and_not_its_upper(
    range_m: float, expected: str
) -> None:
    """Half-open everywhere: `[0,10)`, `[10,20)`, `[20,40)`, `[40,80)`, `>=80`.

    Stated once and tested at every boundary, because a box at exactly 40 m
    landing in two bins or none is the kind of thing that shows up as a strange
    dip in a chart and gets explained away as physics.
    """

    from bevcalib.metrics.reprojection import range_bin

    assert range_bin(range_m) == expected


@pytest.mark.parametrize("range_m", [-0.1, float("nan"), float("inf")])
def test_a_range_that_is_not_a_distance_is_refused(range_m: float) -> None:
    """Silently binning a NaN as `80+` would quietly inflate the far bin."""

    from bevcalib.metrics.reprojection import range_bin

    with pytest.raises(ValueError):
        range_bin(range_m)


def test_the_bins_are_the_five_the_report_will_show() -> None:
    """Adding a bin changes every chart, so the set is asserted rather than assumed."""

    from bevcalib.metrics.reprojection import RANGE_BINS

    assert RANGE_BINS == ("0-10", "10-20", "20-40", "40-80", "80+")


def test_the_median_and_the_ninetieth_percentile_are_hand_computable() -> None:
    """Errors 1 to 5: the median is 3 and the p90 interpolates to 4.6."""

    from bevcalib.metrics.reprojection import pixel_error_percentiles

    median, p90 = pixel_error_percentiles(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))

    assert median == pytest.approx(3.0)
    assert p90 == pytest.approx(4.6)


def test_the_percentiles_ignore_the_order_the_errors_arrive_in() -> None:
    """Sample order is an artefact of iteration and must not reach a reported number."""

    from bevcalib.metrics.reprojection import pixel_error_percentiles

    forward = pixel_error_percentiles(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
    shuffled = pixel_error_percentiles(np.array([4.0, 1.0, 5.0, 3.0, 2.0]))

    assert forward == shuffled


def test_invalid_samples_are_excluded_rather_than_counted_as_zero() -> None:
    """Counting an unmeasurable sample as perfect is the most flattering possible bug."""

    from bevcalib.metrics.reprojection import pixel_error_percentiles

    errors = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 999.0])
    valid = np.array([True, True, True, True, True, False])

    assert pixel_error_percentiles(errors, valid) == pixel_error_percentiles(
        np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    )


def test_an_aggregate_over_nothing_is_undefined_rather_than_zero() -> None:
    """Zero pixels of error is the best possible score; an empty set is not that."""

    from bevcalib.metrics.reprojection import pixel_error_percentiles

    assert pixel_error_percentiles(np.zeros(0)) == (None, None)
    assert pixel_error_percentiles(np.array([1.0, 2.0]), np.array([False, False])) == (None, None)


def test_a_non_finite_error_is_refused_rather_than_averaged() -> None:
    """One NaN turns a percentile into a NaN, and a NaN in a report is not a measurement."""

    from bevcalib.metrics.reprojection import pixel_error_percentiles

    with pytest.raises(ValueError, match=r"^pixel errors must be finite to be summarised$"):
        pixel_error_percentiles(np.array([1.0, float("nan")]))


def test_a_validity_mask_that_does_not_match_the_errors_is_refused() -> None:
    """Misaligned masks would exclude somebody else's sample."""

    from bevcalib.metrics.reprojection import pixel_error_percentiles

    with pytest.raises(ValueError, match="same"):
        pixel_error_percentiles(np.array([1.0, 2.0]), np.array([True]))


def test_the_validity_summary_counts_both_sides_and_names_every_reason() -> None:
    """A rate over an unknown denominator is not a rate, so the denominator travels with it."""

    from bevcalib.metrics.reprojection import summarize_validity

    summary = summarize_validity(
        [
            result(),
            result(),
            result(valid=False, reason="no LiDAR points project into the image"),
            result(valid=False, reason="no LiDAR points project into the image"),
            result(valid=False, reason="the timing request could not be met"),
        ]
    )

    assert summary.valid == 2
    assert summary.invalid == 3
    assert summary.reasons == (
        ("no LiDAR points project into the image", 2),
        ("the timing request could not be met", 1),
    )


def test_the_reasons_are_ordered_by_how_often_they_happened() -> None:
    """The reason that cost the most samples is the one worth reading first."""

    from bevcalib.metrics.reprojection import summarize_validity

    summary = summarize_validity(
        [result(valid=False, reason="rare")] + [result(valid=False, reason="common")] * 3
    )

    assert summary.reasons[0] == ("common", 3)


def test_a_summary_of_nothing_is_still_a_summary() -> None:
    """An empty cohort is a real outcome and must not need a special case at the call site."""

    from bevcalib.metrics.reprojection import summarize_validity

    summary = summarize_validity([])

    assert (summary.valid, summary.invalid, summary.reasons) == (0, 0, ())


def test_a_validity_summary_cannot_be_edited_after_the_fact() -> None:
    """It is the denominator of every rate in the report."""

    import dataclasses

    from bevcalib.metrics.reprojection import summarize_validity

    with pytest.raises(dataclasses.FrozenInstanceError):
        summarize_validity([result()]).valid = 99  # type: ignore[misc]


@pytest.mark.parametrize("shape", [(2, 3), (1, 1, 4)])
def test_errors_that_are_not_a_flat_list_are_refused(shape: tuple[int, ...]) -> None:
    """A per-sample-per-point matrix would silently be flattened into one distribution."""

    from bevcalib.metrics.reprojection import pixel_error_percentiles

    with pytest.raises(ValueError, match="flat"):
        pixel_error_percentiles(np.zeros(shape))
