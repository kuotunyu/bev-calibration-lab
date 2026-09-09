"""GT pairing tolerates bounded arithmetic noise, never category changes."""

from __future__ import annotations

import math

import pytest

from bevcalib.analysis.estimands import CompactFrame, scene_pairs
from bevcalib.artifacts.results import GroundContactResult
from bevcalib.metrics.reprojection import range_bin


def frames(distances: list[float]) -> dict[str, list[CompactFrame]]:
    return {
        str(index): [
            CompactFrame(
                "scene",
                "sample",
                {},
                {
                    "box": GroundContactResult(
                        box_token="box", range_m=distance, error_m=index + 1, invalid_reason=None
                    )
                },
            )
        ]
        for index, distance in enumerate(distances)
    }


@pytest.mark.parametrize("distances", [[15.0, 15.0 + 1e-12], [15.0 + 1e-12, 15.0]])
def test_roundoff_preserves_original_errors_support_and_ranges(distances: list[float]) -> None:
    runs = frames(distances)
    result = scene_pairs(runs, "0", ("1",), "bev_frame_mean_m/10-20")[0]
    assert (result.before, result.after, result.objects, result.frames) == (1, 2, 1, 1)
    assert [runs[str(i)][0].contacts["box"].range_m for i in range(2)] == distances


def test_absolute_budget_accepts_its_boundary_without_relative_scaling() -> None:
    accepted = scene_pairs(frames([0.0, 1e-9]), "0", ("1",), "bev_frame_mean_m/0-10")[0]
    assert accepted.objects == 1
    for distances in ([0.0, math.nextafter(1e-9, math.inf)], [1e8, 1e8 + 1e-5]):
        with pytest.raises(ValueError, match="GT range"):
            scene_pairs(
                frames(distances), "0", ("1",), f"bev_frame_mean_m/{range_bin(distances[0])}"
            )


@pytest.mark.parametrize("edge", [10.0, 20.0, 40.0, 80.0])
def test_tiny_difference_must_not_cross_any_range_bin(edge: float) -> None:
    with pytest.raises(ValueError, match="GT range"):
        scene_pairs(
            frames([math.nextafter(edge, -math.inf), edge]), "0", ("1",), "bev_frame_mean_m/0-10"
        )


def test_same_bin_must_not_cross_inclusive_operator_range_cutoff() -> None:
    assert range_bin(80.0) == range_bin(math.nextafter(80.0, math.inf))
    with pytest.raises(ValueError, match="GT range"):
        scene_pairs(
            frames([80.0, math.nextafter(80.0, math.inf)]), "0", ("1",), "bev_frame_mean_m/80+"
        )


def test_budget_bounds_whole_three_method_span_not_just_distance_to_reference() -> None:
    with pytest.raises(ValueError, match="GT range"):
        scene_pairs(
            frames([15.0, 15.0 - 0.75e-9, 15.0 + 0.75e-9]),
            "0",
            ("1", "2"),
            "bev_frame_mean_m/10-20",
        )
