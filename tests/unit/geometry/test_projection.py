"""Contracts for pinhole projection, and above all for what "invalid" means.

Every point that goes in comes back out. A projection that silently drops points
makes per-sample errors incomparable across faults, because each fault would then
be scored over a different number of points, and a fault that pushes points out of
frame would look like it improved the alignment.

The camera frame here is the usual pinhole one: x right, y down, z forward along
the optical axis.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

# A plausible front-camera matrix: 800 px focal length on a 640x480 image.
INTRINSIC = np.array([[800.0, 0.0, 320.0], [0.0, 800.0, 240.0], [0.0, 0.0, 1.0]])
IMAGE_SIZE = (640, 480)


def test_a_point_on_the_optical_axis_lands_on_the_principal_point() -> None:
    """The one projection anybody can verify without arithmetic."""

    from bevcalib.geometry.projection import project_camera

    result = project_camera(np.array([[0.0, 0.0, 10.0]]), INTRINSIC, IMAGE_SIZE)

    np.testing.assert_allclose(result.uv, [[320.0, 240.0]], atol=1e-12)
    assert bool(result.valid[0])


def test_a_known_point_projects_to_its_computed_pixel() -> None:
    """Golden values: u = fx·x/z + cx, v = fy·y/z + cy, worked out by hand."""

    from bevcalib.geometry.projection import project_camera

    # 800·1/10 + 320 = 400, and 800·0.5/10 + 240 = 280.
    result = project_camera(np.array([[1.0, 0.5, 10.0]]), INTRINSIC, IMAGE_SIZE)

    np.testing.assert_allclose(result.uv, [[400.0, 280.0]], atol=1e-12)
    np.testing.assert_allclose(result.optical_depth, [10.0], atol=1e-12)


def test_a_point_behind_the_camera_can_land_inside_the_image_and_is_still_invalid() -> None:
    """This is why in_front and in_image are two flags rather than one.

    Negating a point negates both numerator and denominator, so the projection is
    unchanged and the mirrored point lands on exactly the same pixel. A pipeline
    that only checks the bounds would draw it.
    """

    from bevcalib.geometry.projection import project_camera

    result = project_camera(np.array([[-1.0, -0.5, -10.0]]), INTRINSIC, IMAGE_SIZE)

    np.testing.assert_allclose(result.uv, [[400.0, 280.0]], atol=1e-12)
    assert bool(result.in_image[0])
    assert not bool(result.in_front[0])
    assert not bool(result.valid[0])


def test_a_point_in_front_but_outside_the_frame_is_in_front_and_not_valid() -> None:
    """The other half of the distinction, so neither flag can stand in for the other."""

    from bevcalib.geometry.projection import project_camera

    result = project_camera(np.array([[100.0, 0.0, 1.0]]), INTRINSIC, IMAGE_SIZE)

    assert bool(result.in_front[0])
    assert not bool(result.in_image[0])
    assert not bool(result.valid[0])


@pytest.mark.parametrize(
    "point",
    [
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [float("nan"), 0.0, 10.0],
        [0.0, float("inf"), 10.0],
        [0.0, 0.0, float("nan")],
    ],
)
def test_a_point_with_no_defined_projection_is_invalid_without_raising(
    point: list[float],
) -> None:
    """Zero depth divides by zero and NaN poisons comparisons; neither may escape."""

    from bevcalib.geometry.projection import project_camera

    result = project_camera(np.array([point]), INTRINSIC, IMAGE_SIZE)

    assert not bool(result.valid[0])
    assert not bool(result.in_front[0])
    assert np.all(np.isnan(result.uv[0]))


def test_every_input_row_comes_back_even_when_most_are_invalid() -> None:
    """Filtering here would make two faults score over different point counts."""

    from bevcalib.geometry.projection import project_camera

    points = np.array(
        [
            [0.0, 0.0, 10.0],  # valid
            [0.0, 0.0, -10.0],  # behind
            [0.0, 0.0, 0.0],  # no projection
            [100.0, 0.0, 1.0],  # in front, out of frame
        ]
    )

    result = project_camera(points, INTRINSIC, IMAGE_SIZE)

    assert result.uv.shape == (4, 2)
    assert result.optical_depth.shape == (4,)
    np.testing.assert_array_equal(result.valid, [True, False, False, False])


@pytest.mark.parametrize(
    ("u_offset", "v_offset", "expected"),
    [
        (0.0, 0.0, True),  # exactly the top-left corner
        (639.999, 479.999, True),  # just inside the far corner
        (640.0, 240.0, False),  # u == width is outside
        (320.0, 480.0, False),  # v == height is outside
        (-0.001, 240.0, False),
        (320.0, -0.001, False),
    ],
)
def test_the_boundary_rule_is_zero_inclusive_and_size_exclusive(
    u_offset: float, v_offset: float, expected: bool
) -> None:
    """`0 <= u < width` and `0 <= v < height`: one convention, stated and tested."""

    from bevcalib.geometry.projection import project_camera

    # Choose a point at unit depth whose projection is exactly the wanted pixel.
    x = (u_offset - 320.0) / 800.0
    y = (v_offset - 240.0) / 800.0

    result = project_camera(np.array([[x, y, 1.0]]), INTRINSIC, IMAGE_SIZE)

    assert bool(result.in_image[0]) is expected


def test_the_result_arrays_have_the_declared_types() -> None:
    """Downstream metrics index these masks; a float mask would silently weight."""

    from bevcalib.geometry.projection import project_camera

    result = project_camera(np.array([[0.0, 0.0, 10.0]]), INTRINSIC, IMAGE_SIZE)

    assert result.uv.dtype == np.float64
    assert result.optical_depth.dtype == np.float64
    for mask in (result.in_front, result.in_image, result.valid):
        assert mask.dtype == np.bool_


def test_no_points_is_a_legitimate_input() -> None:
    """A sweep can miss the camera frustum entirely, and that must not be an error."""

    from bevcalib.geometry.projection import project_camera

    result = project_camera(np.zeros((0, 3)), INTRINSIC, IMAGE_SIZE)

    assert result.uv.shape == (0, 2)
    assert result.valid.shape == (0,)


SHAPE_MESSAGE = r"^intrinsic must have shape 3x3"
PINHOLE_MESSAGE = (
    r"^intrinsic must be an upper-triangular pinhole matrix with bottom row \[0, 0, 1\]$"
)
FOCAL_MESSAGE = r"^intrinsic focal lengths must be positive$"
FINITE_MESSAGE = r"^intrinsic entries must be finite$"


@pytest.mark.parametrize(
    ("intrinsic", "message"),
    [
        (np.eye(4), SHAPE_MESSAGE),
        (np.zeros((3, 3)), PINHOLE_MESSAGE),
        (np.array([[800.0, 0.0, 320.0], [0.0, 800.0, 240.0], [0.0, 0.0, 2.0]]), PINHOLE_MESSAGE),
        (np.array([[800.0, 0.0, 320.0], [0.0, 800.0, 240.0], [1.0, 0.0, 1.0]]), PINHOLE_MESSAGE),
        (np.array([[800.0, 0.0, 320.0], [5.0, 800.0, 240.0], [0.0, 0.0, 1.0]]), PINHOLE_MESSAGE),
        (np.array([[-800.0, 0.0, 320.0], [0.0, 800.0, 240.0], [0.0, 0.0, 1.0]]), FOCAL_MESSAGE),
        (np.array([[800.0, 0.0, 320.0], [0.0, 0.0, 240.0], [0.0, 0.0, 1.0]]), FOCAL_MESSAGE),
        (np.array([[np.nan, 0.0, 320.0], [0.0, 800.0, 240.0], [0.0, 0.0, 1.0]]), FINITE_MESSAGE),
    ],
)
def test_an_intrinsic_that_is_not_a_pinhole_camera_is_rejected(
    intrinsic: np.ndarray,
    message: str,
) -> None:
    """A wrong bottom row silently rescales depth, which is the worst kind of wrong.

    Each row names the message it expects. `match="intrinsic"` matched all four
    of this function's rejections, so it could not tell a non-finite entry from
    a negative focal length from a broken bottom row — and it kept passing when
    any of those messages was reworded.
    """

    from bevcalib.geometry.projection import project_camera

    with pytest.raises(ValueError, match=message):
        project_camera(np.array([[0.0, 0.0, 10.0]]), intrinsic, IMAGE_SIZE)


@pytest.mark.parametrize("shape", [(3,), (2, 2), (4, 3, 1), (3, 4)])
def test_points_that_are_not_n_by_3_are_rejected(shape: tuple[int, ...]) -> None:
    """The same row-vector rule as the rest of the geometry package."""

    from bevcalib.geometry.projection import project_camera

    with pytest.raises(ValueError, match=r"^points must have shape \[N, 3\], got "):
        project_camera(np.zeros(shape), INTRINSIC, IMAGE_SIZE)


@pytest.mark.parametrize("size", [(0, 480), (640, 0), (-1, 480), (640, -1)])
def test_an_image_size_that_cannot_contain_a_pixel_is_rejected(size: tuple[int, int]) -> None:
    """An empty image would make every point invalid for a reason nobody would find."""

    from bevcalib.geometry.projection import project_camera

    with pytest.raises(ValueError, match=r"^image size must be positive, got "):
        project_camera(np.array([[0.0, 0.0, 10.0]]), INTRINSIC, size)


@given(
    x=st.floats(min_value=-50.0, max_value=50.0),
    y=st.floats(min_value=-50.0, max_value=50.0),
    z=st.floats(min_value=0.1, max_value=100.0),
    scale=st.floats(min_value=0.01, max_value=100.0),
)
@settings(max_examples=200, deadline=None)
def test_moving_a_point_along_its_own_ray_does_not_move_its_pixel(
    x: float, y: float, z: float, scale: float
) -> None:
    """A pinhole camera measures direction, not distance; only the depth changes."""

    from bevcalib.geometry.projection import project_camera

    points = np.array([[x, y, z], [x * scale, y * scale, z * scale]])

    result = project_camera(points, INTRINSIC, IMAGE_SIZE)

    np.testing.assert_allclose(result.uv[0], result.uv[1], rtol=1e-9, atol=1e-9)
    assert result.optical_depth[1] == pytest.approx(z * scale)


@pytest.mark.parametrize(
    "size",
    [(1, 1), (1, 4), (8, 1)],
    ids=["a single pixel", "a one-pixel-wide column", "a one-pixel-high row"],
)
def test_an_image_one_pixel_across_is_a_valid_canvas(size: tuple[int, int]) -> None:
    """The bound is positivity, so one is the smallest canvas and must be accepted.

    Written `<= 1` in either dimension the validator would refuse a single row
    or column. That is not hypothetical: a debug crop and a one-row test
    fixture both hit it, and the refusal would read as a malformed image size
    rather than as an off-by-one bound. The rasteriser shares this validator,
    so the two cannot disagree about the canvas.
    """

    from bevcalib.geometry.projection import validate_image_size

    assert validate_image_size(size) == size


@pytest.mark.parametrize(
    ("fx", "fy"),
    [(1.0, 800.0), (800.0, 1.0), (1.0, 1.0)],
    ids=["a one-pixel focal length in x", "in y", "in both"],
)
def test_a_focal_length_of_one_pixel_is_accepted(fx: float, fy: float) -> None:
    """The bound is positivity, and one pixel is positive.

    A focal length that small describes an extremely wide lens rather than a
    broken camera, and nothing downstream divides by anything that vanishes
    because of it. Written `<= 1.0` the validator would refuse it, and the
    refusal would name the focal length as the fault when the bound is what is
    wrong.
    """

    from bevcalib.geometry.projection import project_camera

    intrinsic = np.array([[fx, 0.0, 320.0], [0.0, fy, 240.0], [0.0, 0.0, 1.0]])

    result = project_camera(np.array([[0.0, 0.0, 10.0]]), intrinsic, IMAGE_SIZE)

    assert bool(result.in_front[0])


@pytest.mark.parametrize(
    ("fx", "fy"),
    [(0.0, 800.0), (800.0, 0.0)],
    ids=["a zero focal length in x", "in y"],
)
def test_a_focal_length_of_exactly_zero_is_refused_in_either_axis(
    fx: float,
    fy: float,
) -> None:
    """Zero is the value the bound exists to exclude, and both axes are checked.

    A zero focal length collapses that axis of the image onto the principal
    point, so every projected column or row would be identical and the
    reprojection error would read as suspiciously good. Written `< 0.0` in
    either clause the zero slips through.
    """

    from bevcalib.geometry.projection import project_camera

    intrinsic = np.array([[fx, 0.0, 320.0], [0.0, fy, 240.0], [0.0, 0.0, 1.0]])

    with pytest.raises(ValueError, match=r"^intrinsic focal lengths must be positive$"):
        project_camera(np.array([[0.0, 0.0, 10.0]]), intrinsic, IMAGE_SIZE)


def test_one_non_finite_point_invalidates_only_itself() -> None:
    """Finiteness is decided per point, and a batch is not all-or-nothing.

    A LiDAR sweep carries tens of thousands of returns and a handful of them
    are junk. Reducing the finiteness check across the whole array instead of
    along each row turns one bad return into an empty projection, and the
    caller sees a sweep with no valid points rather than a sweep with one bad
    one. Every downstream metric would then report nothing to measure.
    """

    from bevcalib.geometry.projection import project_camera

    points = np.array(
        [[0.0, 0.0, 10.0], [np.nan, 0.0, 10.0], [1.0, 1.0, 12.0]],
        dtype=np.float64,
    )
    intrinsic = np.array([[800.0, 0.0, 320.0], [0.0, 800.0, 240.0], [0.0, 0.0, 1.0]])

    result = project_camera(points, intrinsic, IMAGE_SIZE)

    assert result.valid.tolist() == [True, False, True]
    assert result.in_front.tolist() == [True, False, True]


def test_single_precision_points_are_promoted_before_projection() -> None:
    """A float32 input is promoted, not computed in single precision.

    Callers get their arrays from torch and from the nuScenes devkit, and both
    hand out float32 routinely. The promotion at the top of each entry point is
    what makes the answer independent of that: without it the whole computation
    runs in single precision and drifts by a few parts in 1e8 — far too small to
    fail any tolerance in this suite, and far too large for a value that is
    stored in an artifact and compared by hash.

    Comparing the two calls against each other rather than against a recorded
    number is what makes this a test of the promotion rather than of a
    particular input.
    """

    from bevcalib.geometry.projection import project_camera

    points = np.array([[1.0, 0.5, 10.0], [-2.0, 3.0, 7.0]], dtype=np.float32)

    single = project_camera(points, INTRINSIC, IMAGE_SIZE)
    promoted = project_camera(points.astype(np.float64), INTRINSIC, IMAGE_SIZE)

    np.testing.assert_array_equal(single.uv, promoted.uv)
    np.testing.assert_array_equal(single.optical_depth, promoted.optical_depth)
    assert single.uv.dtype == np.float64


def test_a_single_precision_intrinsic_is_promoted_before_inversion() -> None:
    """A float32 input is promoted, not computed in single precision.

    Callers get their arrays from torch and from the nuScenes devkit, and both
    hand out float32 routinely. The promotion at the top of each entry point is
    what makes the answer independent of that: without it the whole computation
    runs in single precision and drifts by a few parts in 1e8 — far too small to
    fail any tolerance in this suite, and far too large for a value that is
    stored in an artifact and compared by hash.

    Comparing the two calls against each other rather than against a recorded
    number is what makes this a test of the promotion rather than of a
    particular input.
    """

    from bevcalib.geometry.projection import project_camera

    points = np.array([[1.0, 0.5, 10.0]])

    single = project_camera(points, INTRINSIC.astype(np.float32), IMAGE_SIZE)
    promoted = project_camera(points, INTRINSIC.astype(np.float32).astype(np.float64), IMAGE_SIZE)

    np.testing.assert_array_equal(single.uv, promoted.uv)
