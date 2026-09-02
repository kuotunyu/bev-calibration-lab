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


@pytest.mark.parametrize(
    "intrinsic",
    [
        np.eye(4),
        np.zeros((3, 3)),
        np.array([[800.0, 0.0, 320.0], [0.0, 800.0, 240.0], [0.0, 0.0, 2.0]]),
        np.array([[800.0, 0.0, 320.0], [0.0, 800.0, 240.0], [1.0, 0.0, 1.0]]),
        np.array([[800.0, 0.0, 320.0], [5.0, 800.0, 240.0], [0.0, 0.0, 1.0]]),
        np.array([[-800.0, 0.0, 320.0], [0.0, 800.0, 240.0], [0.0, 0.0, 1.0]]),
        np.array([[800.0, 0.0, 320.0], [0.0, 0.0, 240.0], [0.0, 0.0, 1.0]]),
        np.array([[np.nan, 0.0, 320.0], [0.0, 800.0, 240.0], [0.0, 0.0, 1.0]]),
    ],
)
def test_an_intrinsic_that_is_not_a_pinhole_camera_is_rejected(intrinsic: np.ndarray) -> None:
    """A wrong bottom row silently rescales depth, which is the worst kind of wrong."""

    from bevcalib.geometry.projection import project_camera

    with pytest.raises(ValueError, match="intrinsic"):
        project_camera(np.array([[0.0, 0.0, 10.0]]), intrinsic, IMAGE_SIZE)


@pytest.mark.parametrize("shape", [(3,), (2, 2), (4, 3, 1), (3, 4)])
def test_points_that_are_not_n_by_3_are_rejected(shape: tuple[int, ...]) -> None:
    """The same row-vector rule as the rest of the geometry package."""

    from bevcalib.geometry.projection import project_camera

    with pytest.raises(ValueError, match=r"\[N, 3\]|shape"):
        project_camera(np.zeros(shape), INTRINSIC, IMAGE_SIZE)


@pytest.mark.parametrize("size", [(0, 480), (640, 0), (-1, 480), (640, -1)])
def test_an_image_size_that_cannot_contain_a_pixel_is_rejected(size: tuple[int, int]) -> None:
    """An empty image would make every point invalid for a reason nobody would find."""

    from bevcalib.geometry.projection import project_camera

    with pytest.raises(ValueError, match="image size"):
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
