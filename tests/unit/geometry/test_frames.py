"""Contracts for attaching frame names to transforms so mismatches cannot compose.

The arithmetic in `se3.py` is happy to compose any two transforms. Whether that
composition means anything depends on whether the right-hand transform lands in
the frame the left-hand one starts from, and that is a fact about the data, not
about the matrices. Carrying the frame names is what turns a silent wrong answer
into an exception.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

ROOT_HALF = math.sqrt(0.5)
QUARTER_TURN_Z = (ROOT_HALF, 0.0, 0.0, ROOT_HALF)
IDENTITY_QUATERNION = (1.0, 0.0, 0.0, 0.0)


def framed(target: str, source: str, translation: tuple[float, float, float] = (0.0, 0.0, 0.0)):  # type: ignore[no-untyped-def]
    from bevcalib.geometry.frames import FramedTransform
    from bevcalib.geometry.se3 import SE3

    return FramedTransform(
        target=target,  # type: ignore[arg-type]
        source=source,  # type: ignore[arg-type]
        value=SE3(rotation_wxyz=IDENTITY_QUATERNION, translation_xyz_m=translation),
    )


def test_composing_matching_frames_names_the_new_transform_correctly() -> None:
    """`T_a_b` composed with `T_b_c` is `T_a_c`, and the label must follow the algebra."""

    from bevcalib.geometry.frames import compose_framed

    composed = compose_framed(framed("global", "lidar_ego"), framed("lidar_ego", "lidar_sensor"))

    assert composed.target == "global"
    assert composed.source == "lidar_sensor"


def test_composing_across_a_frame_gap_is_refused_and_names_both_frames() -> None:
    """This is the whole reason the names are carried at all."""

    from bevcalib.geometry.frames import compose_framed

    with pytest.raises(
        ValueError,
        match=r"^cannot compose '.*'<-'.*' with '.*'<-'.*': the left transform starts from '.*' but the right one lands in '.*'$",
    ) as raised:
        compose_framed(framed("global", "lidar_ego"), framed("camera_ego", "camera_sensor"))

    message = str(raised.value)
    assert "lidar_ego" in message and "camera_ego" in message


def test_the_full_sensor_chain_composes_down_to_one_transform() -> None:
    """The formal chain this project runs, asserted end to end rather than described."""

    from bevcalib.geometry.frames import compose_framed

    camera_from_camera_ego = framed("camera_sensor", "camera_ego", (1.0, 0.0, 0.0))
    camera_ego_from_global = framed("camera_ego", "global", (0.0, 2.0, 0.0))
    global_from_lidar_ego = framed("global", "lidar_ego", (0.0, 0.0, 3.0))
    lidar_ego_from_lidar = framed("lidar_ego", "lidar_sensor", (4.0, 0.0, 0.0))

    chain = compose_framed(
        compose_framed(camera_from_camera_ego, camera_ego_from_global),
        compose_framed(global_from_lidar_ego, lidar_ego_from_lidar),
    )

    assert (chain.target, chain.source) == ("camera_sensor", "lidar_sensor")
    # Every step here is a pure translation, so the totals add componentwise.
    assert chain.value.translation_xyz_m == pytest.approx((5.0, 2.0, 3.0))


def test_the_composed_transform_moves_points_the_same_way_as_the_two_steps() -> None:
    """The frame labels must not change the arithmetic, only guard it."""

    from bevcalib.geometry.frames import FramedTransform, compose_framed
    from bevcalib.geometry.se3 import SE3, transform_points

    left = FramedTransform(
        target="global",
        source="lidar_ego",
        value=SE3(rotation_wxyz=QUARTER_TURN_Z, translation_xyz_m=(1.0, 0.0, 0.0)),
    )
    right = FramedTransform(
        target="lidar_ego",
        source="lidar_sensor",
        value=SE3(rotation_wxyz=IDENTITY_QUATERNION, translation_xyz_m=(0.0, 2.0, 0.0)),
    )
    points = np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]])

    composed = transform_points(compose_framed(left, right).value, points)
    stepwise = transform_points(left.value, transform_points(right.value, points))

    np.testing.assert_allclose(composed, stepwise, atol=1e-12)


@pytest.mark.parametrize("frame", ["lidar", "world", "cam_front", "", "LIDAR_SENSOR"])
def test_a_frame_name_outside_the_declared_set_is_rejected(frame: str) -> None:
    """A typed alias documents the set; only a runtime check enforces it."""

    with pytest.raises(
        ValueError,
        match=r"^unknown target frame .*; expected one of \['lidar_sensor', 'lidar_ego', 'global', 'camera_ego', 'camera_sensor'\]$",
    ):
        framed(frame, "global")


def test_a_transform_from_a_frame_to_itself_is_allowed() -> None:
    """An identity within one frame is legitimate and appears when a fault is zero."""

    from bevcalib.geometry.frames import compose_framed

    same = framed("global", "global")

    composed = compose_framed(same, same)

    assert (composed.target, composed.source) == ("global", "global")


def test_a_framed_transform_cannot_be_mutated() -> None:
    """Relabelling a transform after the fact is exactly the mistake being prevented."""

    import dataclasses

    with pytest.raises(dataclasses.FrozenInstanceError):
        framed("global", "lidar_ego").target = "camera_ego"  # type: ignore[misc]


def test_the_declared_frames_are_the_five_the_protocol_names() -> None:
    """Adding a frame is a protocol change, so the set is asserted rather than assumed."""

    from bevcalib.geometry.frames import FRAME_NAMES

    assert FRAME_NAMES == (
        "lidar_sensor",
        "lidar_ego",
        "global",
        "camera_ego",
        "camera_sensor",
    )


def test_inverting_a_framed_transform_swaps_its_two_frames() -> None:
    """Reversing a link in the sensor chain is not optional; the camera half runs backwards."""

    from bevcalib.geometry.frames import inverse_framed

    inverted = inverse_framed(framed("global", "lidar_ego", (1.0, 2.0, 3.0)))

    assert (inverted.target, inverted.source) == ("lidar_ego", "global")
    assert inverted.value.translation_xyz_m == pytest.approx((-1.0, -2.0, -3.0))


def test_a_framed_transform_composed_with_its_inverse_is_an_identity_in_one_frame() -> None:
    """The frame labels must survive the round trip, not only the numbers."""

    from bevcalib.geometry.frames import compose_framed, inverse_framed

    forward = framed("camera_sensor", "camera_ego", (4.0, -5.0, 6.0))

    identity = compose_framed(forward, inverse_framed(forward))

    assert (identity.target, identity.source) == ("camera_sensor", "camera_sensor")
    assert identity.value.translation_xyz_m == pytest.approx((0.0, 0.0, 0.0), abs=1e-12)
