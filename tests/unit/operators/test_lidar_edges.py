"""Contracts for finding depth discontinuities along LiDAR rings, and for scoring them.

A depth jump between two points is only meaningful if the two points are actually
neighbours, and on a spinning LiDAR that means the same ring at adjacent azimuths.
Comparing across rings, or comparing an unsorted point list in file order, produces
edges everywhere and an alignment score that cannot distinguish anything.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

RELATIVE = 0.08
ABSOLUTE_M = 0.5


def ring_row(range_m: float, azimuth_deg: float, ring: int, intensity: float = 10.0) -> list[float]:
    """One LiDAR return at a chosen range and azimuth, in the x, y, z, intensity, ring layout."""

    angle = math.radians(azimuth_deg)
    return [range_m * math.cos(angle), range_m * math.sin(angle), 0.0, intensity, float(ring)]


def sweep(*rows: list[float]) -> np.ndarray:
    return np.array(rows, dtype=np.float64)


def test_a_jump_at_both_thresholds_exactly_is_an_edge() -> None:
    """Inclusive on both criteria: 0.5 m absolute and 0.08 relative, stated once and tested."""

    from bevcalib.operators.lidar_edges import lidar_depth_edges

    # 6.25 to 6.75 is exactly 0.5 m and exactly 0.08 of the nearer range.
    edges = lidar_depth_edges(sweep(ring_row(6.25, 0.0, 0), ring_row(6.75, 1.0, 0)))

    assert edges.source_indices.tolist() == [0]


def test_a_large_absolute_jump_that_is_relatively_small_is_not_an_edge() -> None:
    """Half a metre at forty metres is the beam spreading, not an object boundary."""

    from bevcalib.operators.lidar_edges import lidar_depth_edges

    edges = lidar_depth_edges(sweep(ring_row(10.0, 0.0, 0), ring_row(10.5, 1.0, 0)))

    assert edges.source_indices.size == 0


def test_a_large_relative_jump_that_is_absolutely_small_is_not_an_edge() -> None:
    """Forty centimetres at one metre is close range noise, not a silhouette."""

    from bevcalib.operators.lidar_edges import lidar_depth_edges

    edges = lidar_depth_edges(sweep(ring_row(1.0, 0.0, 0), ring_row(1.4, 1.0, 0)))

    assert edges.source_indices.size == 0


def test_the_near_side_of_the_discontinuity_is_the_edge() -> None:
    """The near point is the object silhouette; the far one is background behind it."""

    from bevcalib.operators.lidar_edges import lidar_depth_edges

    near_first = lidar_depth_edges(sweep(ring_row(5.0, 0.0, 0), ring_row(20.0, 1.0, 0)))
    far_first = lidar_depth_edges(sweep(ring_row(20.0, 0.0, 0), ring_row(5.0, 1.0, 0)))

    assert near_first.source_indices.tolist() == [0]
    assert far_first.source_indices.tolist() == [1]


def test_points_are_compared_in_azimuth_order_and_not_in_file_order() -> None:
    """A sweep does not arrive sorted, and treating file order as azimuth order invents edges."""

    from bevcalib.operators.lidar_edges import lidar_depth_edges

    # In azimuth order the ranges are 5.0, 5.05, 20.0: one jump, between the last two.
    scrambled = sweep(
        ring_row(20.0, 2.0, 0),
        ring_row(5.0, 0.0, 0),
        ring_row(5.05, 1.0, 0),
    )

    edges = lidar_depth_edges(scrambled)

    assert edges.source_indices.tolist() == [2]


def test_two_rings_are_never_compared_with_each_other() -> None:
    """Adjacent rings look at different elevations, so a jump between them means nothing."""

    from bevcalib.operators.lidar_edges import lidar_depth_edges

    edges = lidar_depth_edges(sweep(ring_row(5.0, 0.0, 0), ring_row(20.0, 1.0, 1)))

    assert edges.source_indices.size == 0


def test_the_first_and_last_points_of_a_ring_are_not_treated_as_neighbours() -> None:
    """A sweep may be a cropped arc, and wrapping a crop would put an edge at every crop.

    A full revolution really does wrap, so this costs at most one edge per ring out
    of hundreds. A cropped arc wrapped by mistake produces a false edge on every
    ring of every frame, always in the same place, which is far worse.
    """

    from bevcalib.operators.lidar_edges import lidar_depth_edges

    edges = lidar_depth_edges(
        sweep(ring_row(5.0, -170.0, 0), ring_row(5.05, 0.0, 0), ring_row(20.0, 170.0, 0))
    )

    # The 20.0 to 5.0 wrap pair is ignored; only the 5.05 to 20.0 pair qualifies.
    assert edges.source_indices.tolist() == [1]


def test_intensity_does_not_change_which_points_are_edges() -> None:
    """The operator measures geometry; reflectivity is a different signal entirely."""

    from bevcalib.operators.lidar_edges import lidar_depth_edges

    dim = sweep(ring_row(5.0, 0.0, 0, intensity=1.0), ring_row(20.0, 1.0, 0, intensity=2.0))
    bright = sweep(ring_row(5.0, 0.0, 0, intensity=250.0), ring_row(20.0, 1.0, 0, intensity=9.0))

    assert lidar_depth_edges(dim).source_indices.tolist() == (
        lidar_depth_edges(bright).source_indices.tolist()
    )


def test_the_ring_each_edge_came_from_is_kept() -> None:
    """Per-ring diagnostics are how a systematic elevation problem becomes visible."""

    from bevcalib.operators.lidar_edges import lidar_depth_edges

    edges = lidar_depth_edges(
        sweep(
            ring_row(5.0, 0.0, 3),
            ring_row(20.0, 1.0, 3),
            ring_row(4.0, 0.0, 7),
            ring_row(30.0, 1.0, 7),
        )
    )

    assert edges.ring_ids.tolist() == [3, 7]
    assert edges.ring_ids.dtype == np.int64


def test_the_returned_points_are_the_original_coordinates() -> None:
    """The edge set feeds the projection chain, so it must carry real positions."""

    from bevcalib.operators.lidar_edges import lidar_depth_edges

    points = sweep(ring_row(5.0, 0.0, 0), ring_row(20.0, 1.0, 0))

    edges = lidar_depth_edges(points)

    assert edges.points_lidar_n3.shape == (1, 3)
    np.testing.assert_allclose(edges.points_lidar_n3[0], points[0, :3])


@pytest.mark.parametrize("rows", [[], [[1.0, 0.0, 0.0, 5.0, 0.0]]])
def test_a_ring_with_fewer_than_two_points_has_no_neighbours(rows: list[list[float]]) -> None:
    """No pairs means no jumps; this must be an empty answer rather than an error."""

    from bevcalib.operators.lidar_edges import lidar_depth_edges

    edges = lidar_depth_edges(np.array(rows, dtype=np.float64).reshape(-1, 5))

    assert edges.source_indices.size == 0
    assert edges.points_lidar_n3.shape == (0, 3)


def test_returns_that_cannot_be_used_are_dropped_before_any_comparison() -> None:
    """A zero range divides the relative test by zero, and a NaN ring cannot even be an integer."""

    from bevcalib.operators.lidar_edges import lidar_depth_edges

    edges = lidar_depth_edges(
        sweep(
            [0.0, 0.0, 0.0, 5.0, 0.0],
            [float("nan"), 1.0, 0.0, 5.0, 0.0],
            [3.0, 0.0, 0.0, 5.0, float("nan")],
            ring_row(5.0, 0.0, 0),
            ring_row(20.0, 1.0, 0),
        )
    )

    assert edges.source_indices.tolist() == [3]


def test_a_projected_point_that_is_not_finite_is_refused() -> None:
    """`floor(nan).astype(int)` is a silent garbage index into a real array."""

    from bevcalib.operators.lidar_edges import trimmed_distance_transform_score

    image_edges = np.zeros((4, 4), dtype=bool)
    image_edges[0, 0] = True

    with pytest.raises(ValueError, match=r"^projected points must be finite to be scored$"):
        trimmed_distance_transform_score(np.array([[float("nan"), 0.0]]), image_edges)


def test_the_thresholds_can_be_moved_for_a_sensitivity_check() -> None:
    """The defaults are protocol constants, but the operator must not hard-code them."""

    from bevcalib.operators.lidar_edges import lidar_depth_edges

    points = sweep(ring_row(10.0, 0.0, 0), ring_row(10.5, 1.0, 0))

    assert lidar_depth_edges(points).source_indices.size == 0
    assert lidar_depth_edges(points, relative_jump=0.04).source_indices.tolist() == [0]


@pytest.mark.parametrize("shape", [(3,), (4, 3), (4, 6), (2, 5, 1)])
def test_a_sweep_that_is_not_five_columns_is_rejected(shape: tuple[int, ...]) -> None:
    """Five columns is the contract the reader guarantees: x, y, z, intensity, ring."""

    from bevcalib.operators.lidar_edges import lidar_depth_edges

    with pytest.raises(ValueError, match=r"\[N, 5\]|shape"):
        lidar_depth_edges(np.zeros(shape))


def test_the_score_is_hand_computable() -> None:
    """One row, one edge pixel, distances 0 to 9: the trimmed mean is arithmetic anyone can check."""

    from bevcalib.operators.lidar_edges import trimmed_distance_transform_score

    image_edges = np.zeros((1, 10), dtype=bool)
    image_edges[0, 0] = True
    uv = np.array([[float(column), 0.0] for column in range(10)])

    # Distances are 0..9. Keeping the best nine gives 0..8, whose mean is 4.
    assert trimmed_distance_transform_score(uv, image_edges) == pytest.approx(-4.0)
    # Keeping all ten gives a mean of 4.5.
    assert trimmed_distance_transform_score(uv, image_edges, 1.0) == pytest.approx(-4.5)


def test_perfect_alignment_scores_zero_and_nothing_scores_higher() -> None:
    """Negated distance means larger is better and zero is the ceiling."""

    from bevcalib.operators.lidar_edges import trimmed_distance_transform_score

    image_edges = np.zeros((4, 4), dtype=bool)
    image_edges[2, 2] = True

    aligned = trimmed_distance_transform_score(np.array([[2.0, 2.0]]), image_edges)
    offset = trimmed_distance_transform_score(np.array([[0.0, 0.0]]), image_edges)

    assert aligned == pytest.approx(0.0)
    assert offset < aligned


def test_trimming_discards_the_worst_points_rather_than_the_best() -> None:
    """LiDAR sees edges the camera cannot, so the tail is occlusion and not misalignment."""

    from bevcalib.operators.lidar_edges import trimmed_distance_transform_score

    image_edges = np.zeros((1, 10), dtype=bool)
    image_edges[0, 0] = True
    # Nine points on the edge and one far away: trimming should remove the outlier.
    uv = np.array([[0.0, 0.0]] * 9 + [[9.0, 0.0]])

    assert trimmed_distance_transform_score(uv, image_edges) == pytest.approx(0.0)


@pytest.mark.parametrize("quantile", [0.0, -0.1, 1.1])
def test_a_trim_quantile_outside_its_range_is_rejected(quantile: float) -> None:
    """Zero would keep nothing and above one would keep points that do not exist."""

    from bevcalib.operators.lidar_edges import trimmed_distance_transform_score

    image_edges = np.zeros((1, 4), dtype=bool)
    image_edges[0, 0] = True

    with pytest.raises(ValueError, match="trim quantile"):
        trimmed_distance_transform_score(np.array([[0.0, 0.0]]), image_edges, quantile)


def test_scoring_nothing_against_something_fails_closed() -> None:
    """An empty projection is a real outcome, and it is not a score of zero."""

    from bevcalib.operators.lidar_edges import trimmed_distance_transform_score

    image_edges = np.zeros((1, 4), dtype=bool)
    image_edges[0, 0] = True

    with pytest.raises(ValueError, match=r"^there are no projected points to score$"):
        trimmed_distance_transform_score(np.zeros((0, 2)), image_edges)


def test_scoring_against_an_image_with_no_edges_fails_closed() -> None:
    """Every distance would be infinite, and the mean of infinities is not a measurement."""

    from bevcalib.operators.lidar_edges import trimmed_distance_transform_score

    with pytest.raises(ValueError, match=r"^the image has no edges to measure against$"):
        trimmed_distance_transform_score(np.array([[0.0, 0.0]]), np.zeros((4, 4), dtype=bool))


def test_a_point_outside_the_image_is_refused_rather_than_wrapped() -> None:
    """Negative indices wrap in numpy, which would score a point against the far edge."""

    from bevcalib.operators.lidar_edges import trimmed_distance_transform_score

    image_edges = np.zeros((4, 4), dtype=bool)
    image_edges[0, 0] = True

    with pytest.raises(ValueError, match=r"^a projected point falls outside the image bounds$"):
        trimmed_distance_transform_score(np.array([[-1.0, 0.0]]), image_edges)


@pytest.mark.parametrize("shape", [(4,), (2, 4, 3), (3, 4, 5, 6)])
def test_an_edge_mask_that_is_not_an_image_is_rejected(shape: tuple[int, ...]) -> None:
    """A three-dimensional mask would be a colour image, whose edges are undefined here."""

    from bevcalib.operators.lidar_edges import trimmed_distance_transform_score

    mask = np.ones(shape, dtype=bool)

    with pytest.raises(ValueError, match="two-dimensional"):
        trimmed_distance_transform_score(np.array([[0.0, 0.0]]), mask)


@pytest.mark.parametrize("shape", [(2,), (3, 3), (2, 2, 2)])
def test_projected_points_that_are_not_n_by_2_are_rejected(shape: tuple[int, ...]) -> None:
    """Pixels are two numbers; a three-column array is camera-frame points, not pixels."""

    from bevcalib.operators.lidar_edges import trimmed_distance_transform_score

    image_edges = np.zeros((4, 4), dtype=bool)
    image_edges[0, 0] = True

    with pytest.raises(ValueError, match=r"\[N, 2\]|shape"):
        trimmed_distance_transform_score(np.zeros(shape), image_edges)


@pytest.mark.parametrize(
    ("u", "v"),
    [(-0.5, 2.0), (4.0, 2.0), (2.0, -0.5), (2.0, 4.0), (4.5, 4.5)],
    ids=[
        "past the left edge",
        "exactly at the width",
        "above the top edge",
        "exactly at the height",
        "past the far corner in both axes",
    ],
)
def test_a_point_outside_any_edge_of_the_image_is_refused(u: float, v: float) -> None:
    """Four independent bounds, and each one alone must refuse the point.

    Every clause is joined by `or`, so a point over any single edge is out. Turn
    one of those into an `and` and that edge stops being checked, because the
    two halves of the pair can never both hold; make one comparison
    non-inclusive and the pixel exactly at the width or the height slips
    through. Either way the index reaches numpy, where a negative wraps to the
    opposite edge and an over-large one raises something that reads as a bug
    rather than as bad input.

    The image here is 4x4, so column and row 4 are the first outside it.
    """

    from bevcalib.operators.lidar_edges import trimmed_distance_transform_score

    image_edges = np.zeros((4, 4), dtype=bool)
    image_edges[0, 0] = True

    with pytest.raises(ValueError, match=r"^a projected point falls outside the image bounds$"):
        trimmed_distance_transform_score(np.array([[u, v]]), image_edges)


def test_a_single_point_is_scored_rather_than_trimmed_away() -> None:
    """The trim keeps at least one point, so the smallest cohort still has a score.

    `ceil(1 * 0.8)` is one, but a floor of two would ask for more points than
    exist and average over a padded tail. A single projected point is the
    degenerate case an optimiser hits when a fault pushes almost everything out
    of frame, and it must return that point's own distance.
    """

    from bevcalib.operators.lidar_edges import trimmed_distance_transform_score

    image_edges = np.zeros((4, 4), dtype=bool)
    image_edges[0, 0] = True

    score = trimmed_distance_transform_score(np.array([[3.0, 0.0]]), image_edges)

    assert score == pytest.approx(-3.0)
