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


def test_compact_frame_reports_translation_norm_in_centimetres() -> None:
    from bevcalib.analysis.estimands import compact_frame

    measured = row("scene", "sample", [1])
    document = measured.model_dump()
    document["pose"]["translation_error_m"] = 0.05

    compact = compact_frame(CalibrationResultV2.model_validate(document))

    assert compact.values["translation_norm_cm"] == 5.0


def test_compact_frame_keeps_recovery_null_when_pose_is_unmeasured() -> None:
    from bevcalib.analysis.estimands import compact_frame

    document = row_document() | {
        "pose": None,
        "valid": False,
        "invalid_reason": "pose_unavailable",
    }
    result = compact_frame(CalibrationResultV2.model_validate(document))

    assert result.values["recovery_rate_pct"] is None


def test_compact_frame_keeps_recovery_null_for_timing_fault_with_pose() -> None:
    from bevcalib.analysis.estimands import compact_frame

    document = row_document()
    document.update(fault_axis="time", fault_level=0)
    document["fault"].update(
        rotation_rpy_deg=[0, 0, 0],
        translation_xyz_m=[0, 0, 0],
        requested_time_offset_ms=0,
    )
    document["timing"]["reason"] = "valid"
    result = compact_frame(CalibrationResultV2.model_validate(document))

    assert result.values["recovery_rate_pct"] is None


def test_compact_frame_reports_recovered_pose_as_one_hundred_percent() -> None:
    from bevcalib.analysis.estimands import compact_frame

    document = row_document()
    document["pose"].update(rotation_geodesic_error_deg=0.1, translation_error_m=0.01)
    result = compact_frame(CalibrationResultV2.model_validate(document))

    assert result.values["recovery_rate_pct"] == 100.0


def test_compact_frame_reports_absolute_signed_pose_components() -> None:
    from bevcalib.analysis.estimands import compact_frame

    document = row_document()
    document["pose"].update(
        rotation_rpy_error_deg=[-2.0, 3.0, -4.0],
        translation_xyz_error_m=[-0.01, 0.02, -0.03],
    )
    values = compact_frame(CalibrationResultV2.model_validate(document)).values

    assert values["rotation_abs_roll_deg"] == 2.0
    assert values["rotation_abs_pitch_deg"] == 3.0
    assert values["rotation_abs_yaw_deg"] == 4.0
    assert values["translation_abs_x_cm"] == 1.0
    assert values["translation_abs_y_cm"] == 2.0
    assert values["translation_abs_z_cm"] == 3.0


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


def test_bev_pair_keeps_later_target_range_object_after_earlier_out_of_range_token() -> None:
    from bevcalib.analysis.estimands import paired_estimate

    contacts_before = [
        {"box_token": "a-out", "range_m": 5, "error_m": 9, "invalid_reason": None},
        {"box_token": "z-in", "range_m": 15, "error_m": 3, "invalid_reason": None},
    ]
    contacts_after = [
        {"box_token": "a-out", "range_m": 5, "error_m": 8, "invalid_reason": None},
        {"box_token": "z-in", "range_m": 15, "error_m": 1, "invalid_reason": None},
    ]
    result = paired_estimate(
        {
            "identity": [row("scene", "sample", [1], ground_contacts=contacts_before)],
            "classical": [row("scene", "sample", [1], ground_contacts=contacts_after)],
        },
        "identity",
        ("classical",),
        "bev_frame_mean_m/10-20",
    )

    assert result.before == 3.0
    assert result.after == 1.0
    assert result.support.total_objects == 1
    assert result.support.objects == 1


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


def test_scene_support_counts_repeated_unavailability_and_keeps_later_valid_frame() -> None:
    from bevcalib.analysis.estimands import paired_estimate

    def unavailable(sample: str):
        return row(
            "scene",
            sample,
            [1],
            valid=False,
            invalid_reason="no_image_edges",
            edge_alignment_score=None,
        )

    before = [unavailable("a"), unavailable("b"), row("scene", "c", [1], edge_alignment_score=-10)]
    after = [unavailable("a"), unavailable("b"), row("scene", "c", [1], edge_alignment_score=-4)]
    result = paired_estimate(
        {"identity": before, "classical": after},
        "identity",
        ("classical",),
        "edge_score_px",
    )

    assert result.before == -10
    assert result.after == -4
    assert result.improvement == 6
    assert (result.support.total_frames, result.support.frames) == (3, 1)
    assert result.support.exclusions == {
        "classical:operator_unavailable": 2,
        "identity:operator_unavailable": 2,
    }


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
