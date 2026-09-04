"""Contracts for turning projected points into a sparse depth image.

Several LiDAR returns land on one pixel, and one of them is the surface the camera
actually saw; the rest are behind it. Keeping the nearest is what makes the depth
image comparable to the image, and it has to be independent of the order the
points arrive in, because that order is an artefact of the sweep.
"""

from __future__ import annotations

import numpy as np
import pytest

IMAGE_SIZE = (8, 4)


def test_one_point_writes_its_depth_at_its_pixel_and_nowhere_else() -> None:
    """The base case: everything else in the image stays unobserved."""

    from bevcalib.operators.rasterize import rasterize_min_depth

    depth_image, observed = rasterize_min_depth(
        np.array([[3.0, 2.0]]), np.array([7.5]), np.array([True]), IMAGE_SIZE
    )

    assert depth_image.shape == (4, 8)
    assert observed.sum() == 1
    assert bool(observed[2, 3])
    assert depth_image[2, 3] == pytest.approx(7.5)


def test_the_nearest_return_wins_when_several_land_on_one_pixel() -> None:
    """The far returns are surfaces the camera cannot see through."""

    from bevcalib.operators.rasterize import rasterize_min_depth

    uv = np.array([[3.0, 2.0], [3.2, 2.9], [3.9, 2.1]])
    depth = np.array([9.0, 4.0, 12.0])

    depth_image, observed = rasterize_min_depth(uv, depth, np.array([True] * 3), IMAGE_SIZE)

    assert observed.sum() == 1
    assert depth_image[2, 3] == pytest.approx(4.0)


def test_the_result_does_not_depend_on_the_order_the_points_arrive_in() -> None:
    """Sweep order is an artefact of the sensor and must not reach the output."""

    from bevcalib.operators.rasterize import rasterize_min_depth

    uv = np.array([[3.0, 2.0], [3.2, 2.9], [3.9, 2.1]])
    depth = np.array([9.0, 4.0, 12.0])
    valid = np.array([True] * 3)
    order = [2, 0, 1]

    forward, forward_mask = rasterize_min_depth(uv, depth, valid, IMAGE_SIZE)
    shuffled, shuffled_mask = rasterize_min_depth(uv[order], depth[order], valid[order], IMAGE_SIZE)

    np.testing.assert_array_equal(forward, shuffled)
    np.testing.assert_array_equal(forward_mask, shuffled_mask)


def test_points_marked_invalid_are_not_drawn() -> None:
    """The validity mask from projection is the only thing deciding what is drawn."""

    from bevcalib.operators.rasterize import rasterize_min_depth

    uv = np.array([[3.0, 2.0], [3.0, 2.0]])
    depth = np.array([1.0, 5.0])

    depth_image, observed = rasterize_min_depth(uv, depth, np.array([False, True]), IMAGE_SIZE)

    assert depth_image[2, 3] == pytest.approx(5.0)
    assert observed.sum() == 1


@pytest.mark.parametrize("depth", [0.0, -1.0, float("nan"), float("inf")])
def test_a_depth_that_is_not_a_positive_distance_is_not_drawn(depth: float) -> None:
    """`valid` comes from a caller; a non-positive depth is refused on its own merits."""

    from bevcalib.operators.rasterize import rasterize_min_depth

    _, observed = rasterize_min_depth(
        np.array([[3.0, 2.0]]), np.array([depth]), np.array([True]), IMAGE_SIZE
    )

    assert not observed.any()


def test_an_unobserved_pixel_is_zero_and_masked_out() -> None:
    """Zero is not a depth; the mask is what says whether the number means anything."""

    from bevcalib.operators.rasterize import rasterize_min_depth

    depth_image, observed = rasterize_min_depth(
        np.array([[3.0, 2.0]]), np.array([7.5]), np.array([True]), IMAGE_SIZE
    )

    assert depth_image[0, 0] == 0.0
    assert not bool(observed[0, 0])


def test_no_points_gives_an_empty_but_well_formed_image() -> None:
    """The empty case must not need a special branch at every call site."""

    from bevcalib.operators.rasterize import rasterize_min_depth

    depth_image, observed = rasterize_min_depth(
        np.zeros((0, 2)), np.zeros(0), np.zeros(0, dtype=bool), IMAGE_SIZE
    )

    assert depth_image.shape == (4, 8)
    assert not observed.any()


@pytest.mark.parametrize(
    ("u", "v", "column", "row"),
    [(0.0, 0.0, 0, 0), (5.9, 3.9, 5, 3), (7.0, 3.0, 7, 3), (0.999, 0.999, 0, 0)],
)
def test_a_pixel_owns_the_square_from_its_index_up_to_the_next(
    u: float, v: float, column: int, row: int
) -> None:
    """Flooring, not rounding: pixel `k` covers `[k, k+1)`, matching the bounds rule."""

    from bevcalib.operators.rasterize import rasterize_min_depth

    _, observed = rasterize_min_depth(
        np.array([[u, v]]), np.array([1.0]), np.array([True]), IMAGE_SIZE
    )

    assert bool(observed[row, column])
    assert observed.sum() == 1


def test_the_outputs_have_the_declared_types() -> None:
    """float32 halves the memory of a full-resolution depth image and is ample here."""

    from bevcalib.operators.rasterize import rasterize_min_depth

    depth_image, observed = rasterize_min_depth(
        np.array([[3.0, 2.0]]), np.array([7.5]), np.array([True]), IMAGE_SIZE
    )

    assert depth_image.dtype == np.float32
    assert observed.dtype == np.bool_


@pytest.mark.parametrize(
    ("uv_shape", "depth_size", "valid_size", "message"),
    [
        (
            (2, 2),
            3,
            2,
            r"^uv, depth and valid must describe the same points, got "
            r"\(2, 2\), \(3,\) and \(2,\)$",
        ),
        (
            (2, 2),
            2,
            3,
            r"^uv, depth and valid must describe the same points, got "
            r"\(2, 2\), \(2,\) and \(3,\)$",
        ),
        ((2, 3), 2, 2, r"^uv must have shape \[N, 2\], got \(2, 3\)$"),
        ((2,), 2, 2, r"^uv must have shape \[N, 2\], got \(2,\)$"),
    ],
)
def test_inputs_that_do_not_describe_the_same_points_are_rejected(
    uv_shape: tuple[int, ...],
    depth_size: int,
    valid_size: int,
    message: str,
) -> None:
    """Misaligned arrays would pair one point's pixel with another point's depth.

    Each row names its message, and the message carries all three shapes. That
    is the whole diagnosis: a caller who has just changed one stage of the
    projection chain needs to see which array disagrees with which, not merely
    that something did.
    """

    from bevcalib.operators.rasterize import rasterize_min_depth

    with pytest.raises(ValueError, match=message):
        rasterize_min_depth(
            np.zeros(uv_shape),
            np.ones(depth_size),
            np.ones(valid_size, dtype=bool),
            IMAGE_SIZE,
        )


@pytest.mark.parametrize("size", [(0, 4), (8, 0), (-8, 4)])
def test_an_image_size_that_cannot_contain_a_pixel_is_rejected(size: tuple[int, int]) -> None:
    """Same rule as projection, so the two cannot disagree about the canvas."""

    from bevcalib.operators.rasterize import rasterize_min_depth

    with pytest.raises(ValueError, match=r"^image size must be positive, got "):
        rasterize_min_depth(np.zeros((0, 2)), np.zeros(0), np.zeros(0, dtype=bool), size)


def test_a_valid_point_outside_the_canvas_is_refused_rather_than_wrapped() -> None:
    """Negative indices wrap in numpy, which would draw a point on the opposite edge."""

    from bevcalib.operators.rasterize import rasterize_min_depth

    with pytest.raises(ValueError, match=r"^a drawable point falls outside the image bounds$"):
        rasterize_min_depth(np.array([[-1.0, 2.0]]), np.array([5.0]), np.array([True]), IMAGE_SIZE)


def test_a_pixel_coordinate_that_is_not_finite_is_not_drawn() -> None:
    """`floor(nan).astype(int)` produces a silent garbage index, so it never gets there."""

    from bevcalib.operators.rasterize import rasterize_min_depth

    _, observed = rasterize_min_depth(
        np.array([[float("nan"), 2.0]]), np.array([5.0]), np.array([True]), IMAGE_SIZE
    )

    assert not observed.any()


@pytest.mark.parametrize(
    ("u", "v"),
    [(-0.5, 2.0), (8.0, 2.0), (2.0, -0.5), (2.0, 4.0), (8.5, 4.5)],
    ids=[
        "past the left edge",
        "exactly at the width",
        "above the top edge",
        "exactly at the height",
        "past the far corner in both axes",
    ],
)
def test_a_drawable_point_outside_any_edge_is_refused(u: float, v: float) -> None:
    """Four independent bounds, and each one alone must refuse the point.

    The clauses are joined by `or`, so a point over any single edge is out.
    Turning one into an `and` stops that edge being checked at all, because the
    two halves can never both hold; making a comparison non-inclusive lets the
    pixel exactly at the width or the height through. Either way the index
    reaches numpy, where a negative wraps round to the opposite edge and draws
    the point there.

    This is the same rule the edge scorer applies, tested the same way, because
    the two must not disagree about what the canvas is.
    """

    from bevcalib.operators.rasterize import rasterize_min_depth

    with pytest.raises(ValueError, match=r"^a drawable point falls outside the image bounds$"):
        rasterize_min_depth(np.array([[u, v]]), np.array([5.0]), np.array([True]), IMAGE_SIZE)


def test_a_validity_flag_of_zeros_and_ones_draws_the_same_points_a_bool_mask_draws() -> None:
    """The flags arrive as 0/1 from the projection stage, and must mean the same thing.

    Combined with the finiteness tests the flags become the index into the point
    arrays, and an integer index is positional rather than a mask: `[0, 1]` would
    read points 0 and 1 instead of point 1 alone, drawing the very return the
    projection marked invalid.
    """

    from bevcalib.operators.rasterize import rasterize_min_depth

    uv = np.array([[1.0, 1.0], [5.0, 2.0]])
    depth = np.array([3.0, 9.0])
    keep = [False, True]

    as_integers = rasterize_min_depth(uv, depth, np.array(keep, dtype=np.int64), IMAGE_SIZE)
    as_booleans = rasterize_min_depth(uv, depth, np.array(keep), IMAGE_SIZE)

    np.testing.assert_array_equal(as_integers[0], as_booleans[0])
    np.testing.assert_array_equal(as_integers[1], as_booleans[1])
    assert as_booleans[1].sum() == 1
