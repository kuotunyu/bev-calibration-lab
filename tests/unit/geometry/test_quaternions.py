"""Contracts for the quaternion representation of rotation.

Two things make quaternions worth testing carefully rather than trusting. A
quaternion and its negation are the same rotation, so equality is meaningless
without a canonical sign; and the textbook matrix-to-quaternion formula divides
by something that approaches zero near 180 degrees, which is exactly where a
calibration fault study spends no time and a coordinate-frame bug hides forever.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

ROOT_HALF = math.sqrt(0.5)

# Golden pairs: right-handed active rotations, verified by where they send a basis vector.
NINETY_DEGREES = {
    "x": ((ROOT_HALF, ROOT_HALF, 0.0, 0.0), [[1, 0, 0], [0, 0, -1], [0, 1, 0]]),
    "y": ((ROOT_HALF, 0.0, ROOT_HALF, 0.0), [[0, 0, 1], [0, 1, 0], [-1, 0, 0]]),
    "z": ((ROOT_HALF, 0.0, 0.0, ROOT_HALF), [[0, -1, 0], [1, 0, 0], [0, 0, 1]]),
}
HALF_TURNS = {
    "x": ((0.0, 1.0, 0.0, 0.0), [[1, 0, 0], [0, -1, 0], [0, 0, -1]]),
    "y": ((0.0, 0.0, 1.0, 0.0), [[-1, 0, 0], [0, 1, 0], [0, 0, -1]]),
    "z": ((0.0, 0.0, 0.0, 1.0), [[-1, 0, 0], [0, -1, 0], [0, 0, 1]]),
}


def to_unit(q: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    norm = math.sqrt(sum(value * value for value in q))
    return (q[0] / norm, q[1] / norm, q[2] / norm, q[3] / norm)


def unit_quaternions() -> st.SearchStrategy[tuple[float, float, float, float]]:
    """Draw unit quaternions by normalising four bounded reals away from zero."""

    component = st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False)
    return (
        st.tuples(component, component, component, component)
        .filter(lambda q: math.sqrt(sum(value * value for value in q)) > 1e-3)
        .map(to_unit)
    )


def test_the_identity_quaternion_is_the_identity_rotation() -> None:
    """If this is wrong, nothing downstream can be right."""

    from bevcalib.geometry.quaternions import quaternion_to_matrix

    np.testing.assert_allclose(quaternion_to_matrix((1.0, 0.0, 0.0, 0.0)), np.eye(3), atol=1e-15)


@pytest.mark.parametrize("axis", ["x", "y", "z"])
def test_a_quarter_turn_about_each_axis_matches_its_known_matrix(axis: str) -> None:
    """Golden values, not a reimplementation of the formula being tested."""

    from bevcalib.geometry.quaternions import quaternion_to_matrix

    quaternion, expected = NINETY_DEGREES[axis]

    np.testing.assert_allclose(quaternion_to_matrix(quaternion), expected, atol=1e-15)


def test_a_quarter_turn_about_z_sends_x_to_y() -> None:
    """The sign convention stated in one concrete, checkable sentence."""

    from bevcalib.geometry.quaternions import quaternion_to_matrix

    rotated = quaternion_to_matrix(NINETY_DEGREES["z"][0]) @ np.array([1.0, 0.0, 0.0])

    np.testing.assert_allclose(rotated, [0.0, 1.0, 0.0], atol=1e-15)


@pytest.mark.parametrize("axis", ["x", "y", "z"])
def test_a_half_turn_round_trips_where_the_naive_formula_would_divide_by_zero(axis: str) -> None:
    """At 180 degrees the trace is -1, which is where the textbook branch fails."""

    from bevcalib.geometry.quaternions import matrix_to_quaternion, quaternion_to_matrix

    quaternion, expected = HALF_TURNS[axis]

    matrix = quaternion_to_matrix(quaternion)
    np.testing.assert_allclose(matrix, expected, atol=1e-15)
    assert matrix_to_quaternion(matrix) == pytest.approx(quaternion, abs=1e-12)


def test_a_rotation_just_short_of_a_half_turn_round_trips() -> None:
    """Near 180 degrees is worse than at it: the small term is tiny but not zero."""

    from bevcalib.geometry.quaternions import matrix_to_quaternion, quaternion_to_matrix

    angle = math.radians(179.9999)
    axis = np.array([1.0, 2.0, 3.0])
    axis = axis / np.linalg.norm(axis)
    quaternion = (math.cos(angle / 2), *(axis * math.sin(angle / 2)))

    recovered = matrix_to_quaternion(quaternion_to_matrix(quaternion))

    assert recovered == pytest.approx(quaternion, abs=1e-9)


def test_a_quaternion_and_its_negation_normalise_to_the_same_representation() -> None:
    """q and -q are one rotation, so equality without a canonical sign is a trap."""

    from bevcalib.geometry.quaternions import normalize_quaternion_wxyz

    quaternion = (0.5, -0.5, 0.5, 0.5)
    negated = (-quaternion[0], -quaternion[1], -quaternion[2], -quaternion[3])

    assert normalize_quaternion_wxyz(quaternion) == normalize_quaternion_wxyz(negated)


def test_the_canonical_sign_makes_the_first_nonzero_component_positive() -> None:
    """A stated rule, so two runs on two machines produce comparable records."""

    from bevcalib.geometry.quaternions import normalize_quaternion_wxyz

    assert normalize_quaternion_wxyz((-1.0, 0.0, 0.0, 0.0)) == (1.0, 0.0, 0.0, 0.0)
    assert normalize_quaternion_wxyz((0.0, -1.0, 0.0, 0.0)) == (0.0, 1.0, 0.0, 0.0)
    assert normalize_quaternion_wxyz((0.0, 0.0, 0.0, -1.0)) == (0.0, 0.0, 0.0, 1.0)


def test_a_non_unit_quaternion_is_normalised_rather_than_scaling_the_rotation() -> None:
    """nuScenes metadata is not guaranteed to arrive at exactly unit norm."""

    from bevcalib.geometry.quaternions import normalize_quaternion_wxyz, quaternion_to_matrix

    normalized = normalize_quaternion_wxyz((3.0, 0.0, 0.0, 0.0))

    assert normalized == (1.0, 0.0, 0.0, 0.0)
    np.testing.assert_allclose(quaternion_to_matrix((3.0, 0.0, 0.0, 0.0)), np.eye(3), atol=1e-15)


@pytest.mark.parametrize(
    "quaternion",
    [
        (0.0, 0.0, 0.0, 0.0),
        (float("nan"), 0.0, 0.0, 1.0),
        (float("inf"), 0.0, 0.0, 0.0),
        (1e-20, 0.0, 0.0, 0.0),
    ],
)
def test_a_quaternion_that_does_not_describe_a_rotation_is_rejected(
    quaternion: tuple[float, float, float, float],
) -> None:
    """Zero and non-finite norms have no direction to normalise towards."""

    from bevcalib.geometry.quaternions import normalize_quaternion_wxyz

    with pytest.raises(ValueError):
        normalize_quaternion_wxyz(quaternion)


@pytest.mark.parametrize(
    "matrix",
    [
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.001]],
        [[2.0, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 1.0]],
        [[1.0, 0.1, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
    ],
)
def test_a_matrix_that_is_not_a_rotation_is_rejected(matrix: list[list[float]]) -> None:
    """A scaled or skewed matrix would silently stretch every transformed point."""

    from bevcalib.geometry.quaternions import matrix_to_quaternion

    with pytest.raises(ValueError, match="orthonormal"):
        matrix_to_quaternion(np.array(matrix))


def test_a_reflection_is_rejected_even_though_it_is_orthonormal() -> None:
    """Determinant -1 is orthonormal and still flips handedness, which is not a rotation."""

    from bevcalib.geometry.quaternions import matrix_to_quaternion

    reflection = np.diag([1.0, 1.0, -1.0])

    with pytest.raises(ValueError, match=r"right-handed|determinant"):
        matrix_to_quaternion(reflection)


@pytest.mark.parametrize("shape", [(3,), (2, 3), (3, 4), (1, 3, 3)])
def test_a_matrix_of_the_wrong_shape_is_rejected(shape: tuple[int, ...]) -> None:
    """Catching a bad shape here is cheaper than a broadcast that produces numbers."""

    from bevcalib.geometry.quaternions import matrix_to_quaternion

    with pytest.raises(ValueError, match=r"3x3|shape"):
        matrix_to_quaternion(np.zeros(shape))


def test_the_rotation_matrix_is_float64() -> None:
    """Metre-scale translations and 1e-9 tolerances leave no room for float32."""

    from bevcalib.geometry.quaternions import quaternion_to_matrix

    assert quaternion_to_matrix((1.0, 0.0, 0.0, 0.0)).dtype == np.float64


@given(quaternion=unit_quaternions())
@settings(max_examples=200, deadline=None)
def test_matrix_and_quaternion_round_trip_for_any_rotation(
    quaternion: tuple[float, float, float, float],
) -> None:
    """The round trip is the property; the golden cases above pin the convention."""

    from bevcalib.geometry.quaternions import (
        matrix_to_quaternion,
        normalize_quaternion_wxyz,
        quaternion_to_matrix,
    )

    canonical = normalize_quaternion_wxyz(quaternion)

    recovered = matrix_to_quaternion(quaternion_to_matrix(canonical))

    assert recovered == pytest.approx(canonical, abs=1e-9)


@given(quaternion=unit_quaternions())
@settings(max_examples=200, deadline=None)
def test_every_produced_matrix_is_orthonormal_and_right_handed(
    quaternion: tuple[float, float, float, float],
) -> None:
    """The generator and the validator must agree, or valid input fails validation."""

    from bevcalib.geometry.quaternions import quaternion_to_matrix

    matrix = quaternion_to_matrix(quaternion)

    np.testing.assert_allclose(matrix @ matrix.T, np.eye(3), atol=1e-12)
    assert np.linalg.det(matrix) == pytest.approx(1.0, abs=1e-12)


@pytest.mark.parametrize("shape", [(3,), (5,), (2, 2), (1, 4)])
def test_a_quaternion_of_the_wrong_shape_is_rejected(shape: tuple[int, ...]) -> None:
    """Three components is a rotation vector, not a quaternion, and they are easy to swap."""

    from bevcalib.geometry.quaternions import normalize_quaternion_wxyz

    with pytest.raises(ValueError, match="four components"):
        normalize_quaternion_wxyz(np.ones(shape))  # type: ignore[arg-type]


def test_a_matrix_containing_a_non_finite_entry_is_rejected() -> None:
    """NaN passes an orthonormality check by making every comparison false."""

    from bevcalib.geometry.quaternions import matrix_to_quaternion

    matrix = np.eye(3)
    matrix[1, 1] = float("nan")

    with pytest.raises(ValueError, match="finite"):
        matrix_to_quaternion(matrix)
