"""Contracts for rigid transforms and, above all, for composition order.

`T_target_source` maps points expressed in `source` into `target`. Every function
here is named from that rule, and the composition test is the important one: a
reversed order still returns three finite numbers per point, still round-trips
through its own inverse, and is wrong in a way that looks like a calibration
error rather than a bug.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

if TYPE_CHECKING:
    from bevcalib.geometry.se3 import SE3

ROOT_HALF = math.sqrt(0.5)
QUARTER_TURN_Z = (ROOT_HALF, 0.0, 0.0, ROOT_HALF)
QUARTER_TURN_X = (ROOT_HALF, ROOT_HALF, 0.0, 0.0)
IDENTITY_QUATERNION = (1.0, 0.0, 0.0, 0.0)

POINTS = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 2.0, 3.0]])


def build_se3(
    rotation: tuple[float, float, float, float], translation: tuple[float, float, float]
) -> SE3:
    """Construct through a late import so the strategy can be built before the module exists."""

    from bevcalib.geometry.se3 import SE3

    return SE3(rotation_wxyz=rotation, translation_xyz_m=translation)


def se3_transforms() -> st.SearchStrategy[SE3]:
    """Draw rigid transforms with bounded translations and unit quaternions."""

    component = st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False)
    metres = st.floats(min_value=-50.0, max_value=50.0, allow_nan=False, allow_infinity=False)
    return st.builds(
        build_se3,
        rotation=st.tuples(component, component, component, component).filter(
            lambda q: math.sqrt(sum(value * value for value in q)) > 1e-3
        ),
        translation=st.tuples(metres, metres, metres),
    )


def points_arrays() -> st.SearchStrategy[np.ndarray]:
    metres = st.floats(min_value=-100.0, max_value=100.0, allow_nan=False, allow_infinity=False)
    return st.lists(st.tuples(metres, metres, metres), min_size=0, max_size=8).map(
        lambda rows: np.array(rows, dtype=np.float64).reshape(-1, 3)
    )


def test_the_identity_transform_leaves_every_point_where_it_was() -> None:
    """The base case the rest of the algebra is checked against."""

    from bevcalib.geometry.se3 import SE3, transform_points

    identity = SE3(rotation_wxyz=IDENTITY_QUATERNION, translation_xyz_m=(0.0, 0.0, 0.0))

    np.testing.assert_allclose(transform_points(identity, POINTS), POINTS, atol=1e-15)


def test_a_transform_rotates_before_it_translates() -> None:
    """`R @ p + t`, not `R @ (p + t)`. Both are plausible; only one is a rigid transform."""

    from bevcalib.geometry.se3 import SE3, transform_points

    # A quarter turn about z sends x to y, then the translation shifts along x.
    transform = SE3(rotation_wxyz=QUARTER_TURN_Z, translation_xyz_m=(10.0, 0.0, 0.0))

    moved = transform_points(transform, np.array([[1.0, 0.0, 0.0]]))

    np.testing.assert_allclose(moved, [[10.0, 1.0, 0.0]], atol=1e-15)


def test_composition_applies_the_right_hand_transform_first() -> None:
    """`compose(T_a_b, T_b_c)` is `T_a_c`, which means c is mapped through b into a."""

    from bevcalib.geometry.se3 import SE3, compose, transform_points

    a_from_b = SE3(rotation_wxyz=QUARTER_TURN_Z, translation_xyz_m=(1.0, 0.0, 0.0))
    b_from_c = SE3(rotation_wxyz=QUARTER_TURN_X, translation_xyz_m=(0.0, 2.0, 0.0))

    composed = transform_points(compose(a_from_b, b_from_c), POINTS)
    stepwise = transform_points(a_from_b, transform_points(b_from_c, POINTS))

    np.testing.assert_allclose(composed, stepwise, atol=1e-12)


def test_composing_in_the_wrong_order_gives_a_different_answer() -> None:
    """Proof that the previous test can fail: order matters for these two transforms."""

    from bevcalib.geometry.se3 import SE3, compose, transform_points

    a_from_b = SE3(rotation_wxyz=QUARTER_TURN_Z, translation_xyz_m=(1.0, 0.0, 0.0))
    b_from_c = SE3(rotation_wxyz=QUARTER_TURN_X, translation_xyz_m=(0.0, 2.0, 0.0))

    forward = transform_points(compose(a_from_b, b_from_c), POINTS)
    reversed_order = transform_points(compose(b_from_c, a_from_b), POINTS)

    assert not np.allclose(forward, reversed_order)


def test_composition_carries_the_translation_through_the_left_rotation() -> None:
    """A worked numeric case, so the rule is pinned by arithmetic and not only by a property."""

    from bevcalib.geometry.se3 import SE3, compose

    a_from_b = SE3(rotation_wxyz=QUARTER_TURN_Z, translation_xyz_m=(1.0, 0.0, 0.0))
    b_from_c = SE3(rotation_wxyz=IDENTITY_QUATERNION, translation_xyz_m=(0.0, 2.0, 0.0))

    # R_ab @ (0,2,0) is (-2,0,0); adding t_ab gives (-1,0,0).
    assert compose(a_from_b, b_from_c).translation_xyz_m == pytest.approx((-1.0, 0.0, 0.0))


def test_the_inverse_undoes_the_transform() -> None:
    """`T_source_target` is exactly the transform that takes the points back."""

    from bevcalib.geometry.se3 import SE3, inverse, transform_points

    transform = SE3(rotation_wxyz=QUARTER_TURN_Z, translation_xyz_m=(1.0, -2.0, 3.0))

    there_and_back = transform_points(inverse(transform), transform_points(transform, POINTS))

    np.testing.assert_allclose(there_and_back, POINTS, atol=1e-12)


def test_a_transform_composed_with_its_inverse_is_the_identity() -> None:
    """Stated on the transform rather than on points, which is a stronger claim."""

    from bevcalib.geometry.se3 import SE3, compose, inverse

    transform = SE3(rotation_wxyz=QUARTER_TURN_X, translation_xyz_m=(4.0, 5.0, 6.0))

    identity = compose(transform, inverse(transform))

    assert identity.rotation_wxyz == pytest.approx(IDENTITY_QUATERNION, abs=1e-12)
    assert identity.translation_xyz_m == pytest.approx((0.0, 0.0, 0.0), abs=1e-12)


@pytest.mark.parametrize("count", [0, 1, 3])
def test_any_number_of_points_is_accepted_and_the_shape_is_preserved(count: int) -> None:
    """An empty cohort is a real case: a sweep can project entirely out of frame."""

    from bevcalib.geometry.se3 import SE3, transform_points

    transform = SE3(rotation_wxyz=QUARTER_TURN_Z, translation_xyz_m=(1.0, 0.0, 0.0))
    points = np.zeros((count, 3))

    result = transform_points(transform, points)

    assert result.shape == (count, 3)
    assert result.dtype == np.float64


@pytest.mark.parametrize("shape", [(3,), (2, 2), (4, 3, 1), (3, 4)])
def test_points_that_are_not_n_by_3_are_rejected(shape: tuple[int, ...]) -> None:
    """A `[3, N]` array broadcasts against a rotation and returns confident nonsense."""

    from bevcalib.geometry.se3 import SE3, transform_points

    transform = SE3(rotation_wxyz=IDENTITY_QUATERNION, translation_xyz_m=(0.0, 0.0, 0.0))

    with pytest.raises(ValueError, match=r"^points must have shape \[N, 3\], got "):
        transform_points(transform, np.zeros(shape))


def test_a_transform_normalises_its_quaternion_on_construction() -> None:
    """An SE3 that is not rigid would scale points, and nothing downstream would notice."""

    from bevcalib.geometry.se3 import SE3

    transform = SE3(rotation_wxyz=(2.0, 0.0, 0.0, 0.0), translation_xyz_m=(0.0, 0.0, 0.0))

    assert transform.rotation_wxyz == (1.0, 0.0, 0.0, 0.0)


@pytest.mark.parametrize(
    ("rotation", "translation", "expected"),
    [
        (
            (0.0, 0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            r"^quaternion norm .* is too small to define a rotation$",
        ),
        (
            (1.0, 0.0, 0.0, 0.0),
            (float("nan"), 0.0, 0.0),
            r"^translation components must be finite, got ",
        ),
        (
            (1.0, 0.0, 0.0, 0.0),
            (float("inf"), 0.0, 0.0),
            r"^translation components must be finite, got ",
        ),
    ],
)
def test_a_transform_that_is_not_rigid_is_rejected_at_construction(
    rotation: tuple[float, float, float, float],
    translation: tuple[float, float, float],
    expected: str,
) -> None:
    """Failing at construction means no later function has to re-check it.

    The rotation and the translation are refused by separate checks, and a
    caller who handed over a NaN translation needs to be sent to the
    translation rather than told its rotation was degenerate.
    """

    from bevcalib.geometry.se3 import SE3

    with pytest.raises(ValueError, match=expected):
        SE3(rotation_wxyz=rotation, translation_xyz_m=translation)


def test_a_transform_cannot_be_mutated_after_construction() -> None:
    """Validation at construction is only meaningful if the value then stays put."""

    import dataclasses

    from bevcalib.geometry.se3 import SE3

    transform = SE3(rotation_wxyz=IDENTITY_QUATERNION, translation_xyz_m=(0.0, 0.0, 0.0))

    with pytest.raises(dataclasses.FrozenInstanceError):
        transform.translation_xyz_m = (1.0, 0.0, 0.0)  # type: ignore[misc]


@given(transform=se3_transforms(), points=points_arrays())
@settings(max_examples=150, deadline=None)
def test_the_inverse_returns_every_point_for_any_transform(
    transform: SE3, points: np.ndarray
) -> None:
    """The property the golden cases above are examples of."""

    from bevcalib.geometry.se3 import inverse, transform_points

    there_and_back = transform_points(inverse(transform), transform_points(transform, points))

    np.testing.assert_allclose(there_and_back, points, atol=1e-9)


@given(a=se3_transforms(), b=se3_transforms(), c=se3_transforms(), points=points_arrays())
@settings(max_examples=150, deadline=None)
def test_composition_is_associative(a: SE3, b: SE3, c: SE3, points: np.ndarray) -> None:
    """The four-frame chain is composed in whatever order is convenient, so it must not matter."""

    from bevcalib.geometry.se3 import compose, transform_points

    left = transform_points(compose(compose(a, b), c), points)
    right = transform_points(compose(a, compose(b, c)), points)

    np.testing.assert_allclose(left, right, atol=1e-9)


@pytest.mark.parametrize("translation", [(0.0, 0.0), (0.0, 0.0, 0.0, 0.0), ()])
def test_a_translation_that_is_not_three_components_is_rejected(
    translation: tuple[float, ...],
) -> None:
    """A two-component translation would broadcast into the third axis as a zero."""

    from bevcalib.geometry.se3 import SE3

    with pytest.raises(ValueError, match=r"^translation must have three components, got "):
        SE3(rotation_wxyz=IDENTITY_QUATERNION, translation_xyz_m=translation)  # type: ignore[arg-type]


def test_single_precision_points_are_promoted_before_transforming() -> None:
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

    from bevcalib.geometry.se3 import SE3, transform_points

    pose = SE3(rotation_wxyz=(0.5, 0.5, 0.5, 0.5), translation_xyz_m=(1.25, -0.5, 3.75))
    points = np.array([[1.0, 2.0, 3.0], [-4.0, 5.5, 0.25]], dtype=np.float32)

    np.testing.assert_array_equal(
        transform_points(pose, points), transform_points(pose, points.astype(np.float64))
    )
