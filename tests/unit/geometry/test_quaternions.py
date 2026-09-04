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
    ("quaternion", "expected"),
    [
        ((0.0, 0.0, 0.0, 0.0), r"^quaternion norm .* is too small to define a rotation$"),
        ((float("nan"), 0.0, 0.0, 1.0), r"^quaternion components must be finite$"),
        ((float("inf"), 0.0, 0.0, 0.0), r"^quaternion components must be finite$"),
        ((1e-20, 0.0, 0.0, 0.0), r"^quaternion norm .* is too small to define a rotation$"),
    ],
)
def test_a_quaternion_that_does_not_describe_a_rotation_is_rejected(
    quaternion: tuple[float, float, float, float],
    expected: str,
) -> None:
    """Zero and non-finite norms have no direction to normalise towards.

    They are refused by two different checks and each row says which. A NaN
    component is caught before any norm is taken, because a norm computed from
    it would be NaN and would compare false against every bound rather than
    failing the smallness test the other rows exercise.
    """

    from bevcalib.geometry.quaternions import normalize_quaternion_wxyz

    with pytest.raises(ValueError, match=expected):
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

    with pytest.raises(
        ValueError, match=r"^matrix is not orthonormal within 1e-09: singular values "
    ):
        matrix_to_quaternion(np.array(matrix))


def test_a_reflection_is_rejected_even_though_it_is_orthonormal() -> None:
    """Determinant -1 is orthonormal and still flips handedness, which is not a rotation."""

    from bevcalib.geometry.quaternions import matrix_to_quaternion

    reflection = np.diag([1.0, 1.0, -1.0])

    with pytest.raises(
        ValueError, match=r"^rotation must be right-handed, but the determinant is "
    ):
        matrix_to_quaternion(reflection)


@pytest.mark.parametrize("shape", [(3,), (2, 3), (3, 4), (1, 3, 3)])
def test_a_matrix_of_the_wrong_shape_is_rejected(shape: tuple[int, ...]) -> None:
    """Catching a bad shape here is cheaper than a broadcast that produces numbers."""

    from bevcalib.geometry.quaternions import matrix_to_quaternion

    with pytest.raises(ValueError, match=r"^rotation matrix must have shape 3x3, got "):
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

    with pytest.raises(ValueError, match=r"^quaternion must have four components, got shape "):
        normalize_quaternion_wxyz(np.ones(shape))  # type: ignore[arg-type]


def test_a_matrix_containing_a_non_finite_entry_is_rejected() -> None:
    """NaN passes an orthonormality check by making every comparison false."""

    from bevcalib.geometry.quaternions import matrix_to_quaternion

    matrix = np.eye(3)
    matrix[1, 1] = float("nan")

    with pytest.raises(ValueError, match=r"^rotation matrix entries must be finite$"):
        matrix_to_quaternion(matrix)


def test_a_quaternion_whose_norm_is_exactly_the_minimum_is_still_a_rotation() -> None:
    """The smallness guard is strict, so the threshold itself is a usable input.

    `_MINIMUM_NORM` is the norm BELOW which a direction is numerical noise. A
    quaternion sitting exactly on it still has a well-defined direction, and
    `(1e-12, 0, 0, 0)` normalises to the identity. Written `<=` the guard would
    refuse it, and the refusal would read as degenerate data rather than as an
    off-by-one comparison.
    """

    from bevcalib.geometry import quaternions as module

    assert module.normalize_quaternion_wxyz((1e-12, 0.0, 0.0, 0.0)) == (1.0, 0.0, 0.0, 0.0)


def test_a_matrix_that_drifts_inside_numpy_default_tolerances_is_still_refused() -> None:
    """The orthonormality bound is the declared one, not whatever `np.allclose` defaults to.

    `np.allclose` defaults to `rtol=1e-5, atol=1e-8`, both of which are orders
    of magnitude looser than the 1e-9 this project declares. Dropping either
    argument therefore admits a matrix that is not a rotation to within the
    stated tolerance, and every transform derived from it inherits the error
    silently.

    A uniform scaling cannot show this, because it moves the determinant
    further than it moves the singular values and the right-handedness check
    fires first. Post-multiplying by `diag(1+d, 1, 1/(1+d))` instead leaves the
    determinant at exactly one while shifting two singular values by `d`, so
    the orthonormality check is the only one that can refuse it.
    """

    from bevcalib.geometry import quaternions as module

    rotation = module.quaternion_to_matrix((0.5, 0.5, 0.5, 0.5))
    drift = 5e-9
    skewed = rotation @ np.diag([1.0 + drift, 1.0, 1.0 / (1.0 + drift)])

    assert float(np.linalg.det(skewed)) == pytest.approx(1.0, rel=0.0, abs=1e-12)
    assert drift > module.ORTHONORMALITY_TOLERANCE
    assert drift < 1e-8, "must sit inside numpy's default atol for the test to mean anything"

    with pytest.raises(ValueError, match=r"^matrix is not orthonormal within 1e-09"):
        module.matrix_to_quaternion(skewed)


@pytest.mark.parametrize(
    ("quaternion", "expected"),
    [
        (
            (math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5)),
            (0.7071067811865475, 0.0, 0.0, 0.7071067811865478),
        ),
        ((0.0, 1.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0)),
        ((0.0, 0.0, 1.0, 0.0), (0.0, 0.0, 1.0, 0.0)),
        ((0.0, 0.0, 0.0, 1.0), (0.0, 0.0, 0.0, 1.0)),
    ],
    ids=[
        "branch W: positive trace",
        "branch X: half turn about x",
        "branch Y: half turn about y",
        "branch Z: half turn about z",
    ],
)
def test_each_conversion_branch_produces_its_exact_documented_value(
    quaternion: tuple[float, float, float, float],
    expected: tuple[float, float, float, float],
) -> None:
    """One matrix per branch, with the result pinned to the bit.

    Shepperd's four branches are algebraically equal wherever more than one is
    valid, so a mis-selected branch usually returns the right answer with worse
    conditioning — which is exactly why every approximate test passes under a
    mutated branch condition. They are NOT equal to the last bit, and this
    project stores quaternions in artifacts that are compared by hash, so the
    last bit is part of the contract rather than an implementation detail.

    Each row also documents which branch it reaches: a positive trace takes W,
    and the three half turns each make one diagonal entry the strict maximum,
    which is the only way to reach X, Y and Z respectively.
    """

    from bevcalib.geometry import quaternions as module

    result = module.matrix_to_quaternion(module.quaternion_to_matrix(quaternion))

    assert result == expected


def test_a_trace_of_exactly_zero_does_not_take_the_positive_trace_branch() -> None:
    """The trace test is strict, and zero is the boundary it decides.

    A 120 degree turn about (1, 1, 1) has an exactly zero trace and an all-zero
    diagonal, so it falls past both the trace branch and the largest-diagonal
    branches to the last one. Written `>=` the first branch would claim it and
    return `(0.5, 0.5, 0.5, 0.5)` exactly; the branch that actually runs returns
    a value one bit away from that in two components.

    Both are the same rotation to fourteen digits. They are different artifacts.
    """

    from bevcalib.geometry import quaternions as module

    axis = np.array([1.0, 1.0, 1.0]) / math.sqrt(3.0)
    matrix = module.quaternion_to_matrix(
        (math.cos(math.radians(60.0)), *(axis * math.sin(math.radians(60.0))))
    )

    assert float(matrix[0, 0] + matrix[1, 1] + matrix[2, 2]) == 0.0
    assert module.matrix_to_quaternion(matrix) == (0.5000000000000001, 0.5, 0.5, 0.5000000000000001)


def test_a_tie_between_the_second_and_third_diagonal_entries_takes_the_last_branch() -> None:
    """The second comparison is strict too, and a half turn about (0, 1, 1) ties it.

    That rotation makes the second and third diagonal entries exactly equal, so
    `values[1, 1] > values[2, 2]` is false and the final branch runs. Written
    `>=` the third branch would claim it instead. The two answers differ by one
    bit in each of the last two components, and swap which of them carries it.
    """

    from bevcalib.geometry import quaternions as module

    axis = np.array([0.0, 1.0, 1.0]) / math.sqrt(2.0)
    matrix = module.quaternion_to_matrix((0.0, *axis))

    assert matrix[1, 1] == matrix[2, 2]
    assert module.matrix_to_quaternion(matrix) == (0.0, 0.0, 0.7071067811865476, 0.7071067811865475)


def test_the_finiteness_messages_name_what_they_reject() -> None:
    """Two different arrays, two different messages, and an operator reads them.

    A quaternion and a rotation matrix arrive from different places in the
    calibration chain, so an error that says only "not finite" leaves the
    caller to work out which input was at fault. Both messages are asserted in
    full, which is also what makes them detectable when reworded.
    """

    from bevcalib.geometry import quaternions as module

    with pytest.raises(ValueError, match=r"^quaternion components must be finite$"):
        module.normalize_quaternion_wxyz((math.nan, 0.0, 0.0, 1.0))
    with pytest.raises(ValueError, match=r"^rotation matrix entries must be finite$"):
        module.matrix_to_quaternion(np.full((3, 3), math.inf, dtype=np.float64))


def test_a_norm_too_small_to_be_a_rotation_reports_the_norm_it_saw() -> None:
    """The number is the diagnosis: it says whether the input was zero or merely tiny.

    A caller handed a quaternion assembled from a failed fit needs to know
    whether it collapsed to nothing or is just badly scaled, and only the value
    distinguishes those.
    """

    from bevcalib.geometry import quaternions as module

    with pytest.raises(
        ValueError, match=r"^quaternion norm 0\.0 is too small to define a rotation$"
    ):
        module.normalize_quaternion_wxyz((0.0, 0.0, 0.0, 0.0))


@pytest.mark.parametrize(
    ("axis", "expected"),
    [
        ((0.5, 0.8, -math.sqrt(0.11)), (0.0, 0.5, 0.8, -0.33166247903553997)),
        (
            (math.sqrt(0.49), math.sqrt(0.02), math.sqrt(0.49)),
            (0.0, 0.7000000000000001, 0.14142135623730953, 0.7000000000000001),
        ),
        (
            (math.sqrt(0.49), math.sqrt(0.49), math.sqrt(0.02)),
            (0.0, 0.7000000000000001, 0.7000000000000001, 0.14142135623730953),
        ),
    ],
    ids=[
        "second axis largest, third negative",
        "first and third axes tie",
        "first and second axes tie",
    ],
)
def test_the_largest_diagonal_branch_reads_the_diagonal_and_nothing_else(
    axis: tuple[float, float, float],
    expected: tuple[float, float, float, float],
) -> None:
    """The second branch compares `m00` against `m11` and `m22`, and both must be strict.

    That condition has six moving parts — two comparisons, four indices and the
    conjunction — and a half turn is the cheapest rotation that can pin all of
    them, because for a half turn about a unit axis the quaternion is exactly
    `(0, axis)` and the diagonal is `2 n_i^2 - 1`. Choosing the axis chooses the
    diagonal ordering, and choosing the sign of a component chooses the sign of
    an off-diagonal term.

    The three axes here were picked so that between them every single-token
    change to that condition flips it:

    * the first makes an off-diagonal entry smaller than `m00`, which is what
      an index reading `m21` or `m12` instead of `m11` would compare against,
      and makes `m00 > m22` true so a conjunction weakened to a disjunction
      takes the branch;
    * the second ties `m00` and `m22`, which is the only way a `>=` in the
      second comparison shows, and puts an off-diagonal above `m22` for the
      indices that read the wrong row;
    * the third ties `m00` and `m11`, which is the only way a `>=` in the first
      comparison shows.

    A half turn about a unit axis returns that axis exactly, so the expected
    values are the axis itself rather than a recorded output. Taking the wrong
    branch still returns the same rotation, but not the same bits.
    """

    from bevcalib.geometry import quaternions as module

    unit = np.asarray(axis, dtype=np.float64)
    unit = unit / np.linalg.norm(unit)
    matrix = module.quaternion_to_matrix((0.0, *unit))

    assert float(matrix[0, 0] + matrix[1, 1] + matrix[2, 2]) < 0.0
    assert module.matrix_to_quaternion(matrix) == expected


def test_the_answer_does_not_depend_on_the_dtype_the_caller_handed_in() -> None:
    """A float32 quaternion is promoted, not computed in single precision.

    Callers get arrays from torch and from the nuScenes devkit, and both hand
    out float32 routinely. The promotion at the top of each entry point is what
    makes the answer independent of that: without it the norm, the division and
    the sign test all run in single precision and the result drifts by about
    2e-08 — far too small to fail any tolerance in this suite, and far too
    large for a value that is stored in an artifact and compared by hash.

    Asserting equality between the two calls rather than against a recorded
    number is what makes this test about the promotion rather than about a
    particular quaternion.
    """

    from bevcalib.geometry import quaternions as module

    single = np.array([0.3, 0.4, 0.5, 0.7], dtype=np.float32)

    assert module.normalize_quaternion_wxyz(single) == module.normalize_quaternion_wxyz(
        single.astype(np.float64)
    )


def test_a_single_precision_rotation_matrix_is_not_orthonormal_enough() -> None:
    """float32 cannot hold a rotation to 1e-9, and the loader says so rather than coping.

    Rounding a rotation matrix to single precision moves its singular values by
    a few parts in 1e9, which is outside the tolerance this project declares.
    That is a real limit callers meet: a matrix that arrives from torch or from
    a devkit as float32 must be recomposed from a quaternion rather than cast,
    because casting has already destroyed the property being validated.

    Refusing it is the right behaviour and is pinned here so that loosening the
    tolerance to accommodate float32 becomes a visible decision instead of a
    quiet one.
    """

    from bevcalib.geometry import quaternions as module

    rotation = module.quaternion_to_matrix((0.3, 0.4, 0.5, 0.7))

    assert module.matrix_to_quaternion(rotation) == pytest.approx(
        module.normalize_quaternion_wxyz((0.3, 0.4, 0.5, 0.7)), abs=1e-12
    )
    with pytest.raises(ValueError, match=r"^matrix is not orthonormal within 1e-09"):
        module.matrix_to_quaternion(rotation.astype(np.float32))
