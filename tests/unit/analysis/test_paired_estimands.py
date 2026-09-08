"""Analytical scene estimands: frames are matched, pixel arrays are not."""

from __future__ import annotations

import pytest
from tests.unit.artifacts.test_results_v2 import row_document

from bevcalib.artifacts.results import CalibrationResultV2


def row(scene: str, sample: str, pixels: list[float], **changes) -> CalibrationResultV2:
    return CalibrationResultV2.model_validate(
        row_document()
        | {
            "scene_token": scene,
            "sample_token": sample,
            "pixel_errors_px": pixels,
            "projection_count": max(3, len(pixels)),
            **changes,
        }
    )


def test_pixel_quantiles_are_frame_then_scene_means_on_sample_pairs() -> None:
    from bevcalib.analysis.estimands import paired_estimate

    before = [row("scene-a", "a", [0, 100]), row("scene-a", "b", [4]), row("scene-b", "c", [20])]
    after = [row("scene-b", "c", [10]), row("scene-a", "b", [2]), row("scene-a", "a", [10])]
    result = paired_estimate(
        {"identity": before, "classical": after}, "identity", ("classical",), "pixel_frame_p50_px"
    )
    assert result.before == 23.5 and result.after == 8
    assert result.improvement == 15.5
    assert (result.support.frames, result.support.scenes) == (3, 2)
    assert result.interval.resamples == 5000 and result.interval.seed == 20260831


def test_bev_pairs_box_identity_before_frame_averaging_and_refuses_range_drift() -> None:
    from bevcalib.analysis.estimands import paired_estimate

    def contact(token, distance, error):
        return {"box_token": token, "range_m": distance, "error_m": error, "invalid_reason": None}

    before = row("scene", "sample", [1], ground_contacts=[contact("a", 15, 1), contact("b", 18, 5)])
    after = row("scene", "sample", [1], ground_contacts=[contact("b", 18, 1), contact("a", 15, 2)])
    result = paired_estimate(
        {"identity": [before], "classical": [after]},
        "identity",
        ("classical",),
        "bev_frame_mean_m/10-20",
    )
    assert result.before == 3 and result.after == 1.5 and result.improvement == 1.5
    assert result.support.objects == 2
    changed = after.model_dump()
    changed["ground_contacts"][0]["range_m"] = 19
    with pytest.raises(ValueError, match="GT range"):
        paired_estimate(
            {"identity": [before], "classical": [CalibrationResultV2.model_validate(changed)]},
            "identity",
            ("classical",),
            "bev_frame_mean_m/10-20",
        )


def test_operator_partial_rows_survive_global_invalidity_and_empty_support_is_null() -> None:
    from bevcalib.analysis.estimands import paired_estimate

    partial = row(
        "scene",
        "sample",
        [4],
        valid=False,
        invalid_reason="no_image_edges",
        edge_alignment_score=None,
    )
    valid = row("scene", "sample", [2])
    values = {"identity": [partial], "classical": [valid]}
    assert (
        paired_estimate(values, "identity", ("classical",), "pixel_frame_p50_px").improvement == 2
    )
    edge = paired_estimate(values, "identity", ("classical",), "edge_score_px")
    assert edge.improvement is None and edge.interval is None
    assert edge.reason == "no_common_operator_valid_frames"
    assert edge.support.excluded_frames == 1


def test_three_seed_mean_uses_common_support_and_higher_is_better_for_edge() -> None:
    from bevcalib.analysis.estimands import paired_estimate

    values = {"identity": [row("scene", "sample", [1], edge_alignment_score=-10)]}
    for seed, score in ((17, -1), (42, -4), (73, -7)):
        values[f"learned-{seed}"] = [row("scene", "sample", [1], edge_alignment_score=score)]
    result = paired_estimate(
        values, "identity", ("learned-17", "learned-42", "learned-73"), "edge_score_px"
    )
    assert result.before == -10 and result.after == -4 and result.improvement == 6
    assert result.interval.estimate == 6
    assert result.support.frames == 1


def test_single_run_support_counts_each_unavailable_frame_once() -> None:
    from bevcalib.analysis.estimands import compact_frame, finish_estimate, scene_pairs

    partial = row(
        "scene",
        "sample",
        [1],
        valid=False,
        invalid_reason="no_image_edges",
        edge_alignment_score=None,
    )
    result = finish_estimate(
        scene_pairs(
            {"identity": [compact_frame(partial)]}, "identity", ("identity",), "edge_score_px"
        )
    )
    assert result.support.exclusions == {"identity:operator_unavailable": 1}


def test_missing_or_duplicate_frame_identity_is_not_an_operator_exclusion() -> None:
    from bevcalib.analysis.estimands import paired_estimate

    measured = row("scene", "sample", [1])
    for after in ([], [measured, measured]):
        with pytest.raises(ValueError, match="frame inventory"):
            paired_estimate(
                {"identity": [measured], "classical": after},
                "identity",
                ("classical",),
                "pixel_frame_p50_px",
            )


def test_bev_union_denominator_retains_missing_and_invalid_objects() -> None:
    from bevcalib.analysis.estimands import paired_estimate

    before = row(
        "scene",
        "sample",
        [1],
        ground_contacts=[
            {"box_token": "only-before", "range_m": 15, "error_m": 2, "invalid_reason": None},
            {"box_token": "invalid", "range_m": 16, "error_m": None, "invalid_reason": "range"},
        ],
        valid=False,
        invalid_reason="no_valid_ground_contact",
    )
    after = row(
        "scene",
        "sample",
        [1],
        ground_contacts=[
            {"box_token": "invalid", "range_m": 16, "error_m": 1, "invalid_reason": None},
        ],
    )
    result = paired_estimate(
        {"identity": [before], "classical": [after]},
        "identity",
        ("classical",),
        "bev_frame_mean_m/10-20",
    )
    assert result.improvement is None
    assert (
        result.support.total_objects,
        result.support.objects,
        result.support.excluded_objects,
    ) == (2, 0, 2)
    assert result.support.exclusions["object_missing_or_invalid_in_common_support"] == 2
