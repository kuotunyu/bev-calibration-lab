"""Contracts for reconstructing where a box touches the ground, in bird's-eye view.

The operator is deliberately oracle-controlled. The contact point comes from the
ground-truth 3D box and is projected with the TRUE calibration, so the detection is
perfect by construction. Only the back-projection uses the assumed, possibly faulty
calibration. Whatever error appears in the reconstructed position is therefore
calibration error and nothing else, which is the only way the study can attribute
it.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from bevcalib.geometry.quaternions import matrix_to_quaternion
from bevcalib.geometry.se3 import SE3

INTRINSIC = np.array([[800.0, 0.0, 320.0], [0.0, 800.0, 240.0], [0.0, 0.0, 1.0]])
IMAGE_SIZE = (640, 480)
GROUND_Z = 0.0

# A camera 1.5 m above the ground looking along global +x, with the usual camera
# axes: x right, y down, z forward.
_CAMERA_ROTATION = np.array([[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]])
CAMERA_FROM_GLOBAL = SE3(
    rotation_wxyz=matrix_to_quaternion(_CAMERA_ROTATION),
    translation_xyz_m=(0.0, 1.5, 0.0),
)
IDENTITY_QUATERNION = (1.0, 0.0, 0.0, 0.0)


def quaternion_about(axis: str, degrees: float) -> tuple[float, float, float, float]:
    angle = math.radians(degrees) / 2.0
    component = {"x": 1, "y": 2, "z": 3}[axis]
    values = [math.cos(angle), 0.0, 0.0, 0.0]
    values[component] = math.sin(angle)
    return (values[0], values[1], values[2], values[3])


def test_the_bottom_centre_of_an_upright_box_is_half_its_height_below_the_centre() -> None:
    """The base case, and the one a reader can check without touching a rotation."""

    from bevcalib.operators.ground_contact import bottom_center_global

    bottom = bottom_center_global(np.array([10.0, 2.0, 0.75]), (1.8, 4.2, 1.5), IDENTITY_QUATERNION)

    np.testing.assert_allclose(bottom, [10.0, 2.0, 0.0], atol=1e-12)


def test_turning_a_box_on_the_spot_does_not_move_its_contact_point() -> None:
    """A yaw keeps the box's own up axis vertical, so the bottom stays where it was.

    This is the test that catches a rotation applied to the wrong vector: with only
    yaw, a correct implementation and one that rotates the offset in the world frame
    happen to agree, so the case below with a roll is what separates them.
    """

    from bevcalib.operators.ground_contact import bottom_center_global

    for yaw_deg in (0.0, 37.0, 180.0, -95.0):
        bottom = bottom_center_global(
            np.array([10.0, 2.0, 0.75]), (1.8, 4.2, 1.5), quaternion_about("z", yaw_deg)
        )
        np.testing.assert_allclose(bottom, [10.0, 2.0, 0.0], atol=1e-12)


def test_a_box_tipped_onto_its_side_has_its_bottom_somewhere_else_entirely() -> None:
    """The offset is along the box's own up axis, not along global z."""

    from bevcalib.operators.ground_contact import bottom_center_global

    # A quarter turn about x sends the box's local -z onto global +y.
    bottom = bottom_center_global(
        np.array([10.0, 2.0, 0.75]), (1.8, 4.2, 1.5), quaternion_about("x", 90.0)
    )

    np.testing.assert_allclose(bottom, [10.0, 2.75, 0.75], atol=1e-12)


@pytest.mark.parametrize("size", [(1.8, 4.2, 0.0), (1.8, 4.2, -1.5), (0.0, 4.2, 1.5)])
def test_a_box_with_no_extent_is_rejected(size: tuple[float, float, float]) -> None:
    """A zero or negative dimension is a parsing error, not a very flat car."""

    from bevcalib.operators.ground_contact import bottom_center_global

    with pytest.raises(ValueError, match="positive"):
        bottom_center_global(np.array([10.0, 2.0, 0.75]), size, IDENTITY_QUATERNION)


@pytest.mark.parametrize("shape", [(2,), (4,), (1, 3)])
def test_a_box_centre_that_is_not_three_numbers_is_rejected(shape: tuple[int, ...]) -> None:
    """Same rule as everywhere else: a `[1, 3]` centre would broadcast."""

    from bevcalib.operators.ground_contact import bottom_center_global

    with pytest.raises(ValueError, match="three"):
        bottom_center_global(np.zeros(shape), (1.8, 4.2, 1.5), IDENTITY_QUATERNION)


def test_a_pixel_back_projects_onto_the_ground_point_it_came_from() -> None:
    """Hand-computable: the ground point (10, 2) projects to (160, 360) and back again.

    The camera sits 1.5 m up looking along +x, so (10, 2, 0) is at (-2, 1.5, 10) in
    camera coordinates, which is u = 800·(-2)/10 + 320 = 160 and
    v = 800·1.5/10 + 240 = 360.
    """

    from bevcalib.operators.ground_contact import reconstruct_ground_contact

    contact = reconstruct_ground_contact((160.0, 360.0), CAMERA_FROM_GLOBAL, INTRINSIC, GROUND_Z)

    assert contact is not None
    assert contact == pytest.approx((10.0, 2.0), abs=1e-9)


def test_the_reconstruction_inverts_the_projection_for_any_ground_point() -> None:
    """Round trip through the real projection code, not through a restatement of it."""

    from bevcalib.geometry.projection import project_camera
    from bevcalib.geometry.se3 import transform_points
    from bevcalib.operators.ground_contact import reconstruct_ground_contact

    # A ground point at x = 5 projects to v = 480 exactly, which the half-open
    # image bounds rule out, so the near case starts at six metres.
    for x, y in ((6.0, 0.0), (20.0, -3.0), (40.0, 8.0), (12.5, 1.25)):
        ground = np.array([[x, y, GROUND_Z]])
        projection = project_camera(
            transform_points(CAMERA_FROM_GLOBAL, ground), INTRINSIC, IMAGE_SIZE
        )
        assert projection.valid[0]

        contact = reconstruct_ground_contact(
            (float(projection.uv[0, 0]), float(projection.uv[0, 1])),
            CAMERA_FROM_GLOBAL,
            INTRINSIC,
            GROUND_Z,
        )

        assert contact == pytest.approx((x, y), abs=1e-9)


def test_a_ray_along_the_horizon_never_reaches_the_ground() -> None:
    """At the principal row the ray is exactly horizontal, and dividing by its slope is not on."""

    from bevcalib.operators.ground_contact import reconstruct_ground_contact

    assert (
        reconstruct_ground_contact((320.0, 240.0), CAMERA_FROM_GLOBAL, INTRINSIC, GROUND_Z) is None
    )


def test_a_ray_above_the_horizon_meets_the_plane_behind_the_camera() -> None:
    """The plane is infinite, so the arithmetic succeeds and answers with a point behind you."""

    from bevcalib.operators.ground_contact import reconstruct_ground_contact

    assert (
        reconstruct_ground_contact((320.0, 100.0), CAMERA_FROM_GLOBAL, INTRINSIC, GROUND_Z) is None
    )


@pytest.mark.parametrize("uv", [(float("nan"), 360.0), (160.0, float("inf"))])
def test_a_pixel_that_is_not_a_pixel_is_rejected(uv: tuple[float, float]) -> None:
    """Distinguish "no intersection exists" from "the question was malformed"."""

    from bevcalib.operators.ground_contact import reconstruct_ground_contact

    with pytest.raises(ValueError, match="finite"):
        reconstruct_ground_contact(uv, CAMERA_FROM_GLOBAL, INTRINSIC, GROUND_Z)


def test_a_ground_height_that_is_not_a_number_is_rejected() -> None:
    """The plane has to exist before a ray can miss it."""

    from bevcalib.operators.ground_contact import reconstruct_ground_contact

    with pytest.raises(ValueError, match="finite"):
        reconstruct_ground_contact((160.0, 360.0), CAMERA_FROM_GLOBAL, INTRINSIC, float("nan"))


def test_an_intrinsic_that_is_not_a_pinhole_camera_is_rejected() -> None:
    """The same validator the projection uses, so the two cannot disagree."""

    from bevcalib.operators.ground_contact import reconstruct_ground_contact

    with pytest.raises(ValueError, match="intrinsic"):
        reconstruct_ground_contact((160.0, 360.0), CAMERA_FROM_GLOBAL, np.eye(4), GROUND_Z)


def observe(**overrides: object):  # type: ignore[no-untyped-def]
    from bevcalib.operators.ground_contact import observe_ground_contact

    arguments: dict[str, object] = {
        "box_token": "box-0",
        "box_center_global": np.array([10.0, 2.0, 0.75]),
        "size_wlh": (1.8, 4.2, 1.5),
        "orientation_wxyz": IDENTITY_QUATERNION,
        "true_camera_from_global": CAMERA_FROM_GLOBAL,
        "assumed_camera_from_global": CAMERA_FROM_GLOBAL,
        "intrinsic": INTRINSIC,
        "image_size_wh": IMAGE_SIZE,
        "ground_z_global": GROUND_Z,
    }
    return observe_ground_contact(**(arguments | overrides))  # type: ignore[arg-type]


def test_a_perfect_calibration_reconstructs_the_box_exactly_where_it_is() -> None:
    """Zero fault must give zero error, or every measured error includes a constant bias."""

    observation = observe()

    assert observation.valid
    assert observation.box_token == "box-0"
    assert observation.true_uv == pytest.approx((160.0, 360.0), abs=1e-9)
    assert observation.oracle_ground_xy_m == pytest.approx((10.0, 2.0), abs=1e-12)
    assert observation.assumed_ground_xy_m == pytest.approx((10.0, 2.0), abs=1e-9)


def test_a_horizontal_translation_fault_shifts_the_reconstruction_by_exactly_that_much() -> None:
    """A hand-computable error: move the assumed camera sideways and the ground point follows.

    A purely horizontal shift leaves the camera height unchanged, so the ray meets
    the plane at the same distance along itself and the intersection moves by the
    same vector the camera did.
    """

    from bevcalib.geometry.se3 import compose

    shifted = compose(
        CAMERA_FROM_GLOBAL,
        SE3(rotation_wxyz=IDENTITY_QUATERNION, translation_xyz_m=(0.2, 0.0, 0.0)),
    )

    observation = observe(assumed_camera_from_global=shifted)

    assert observation.valid
    error = np.array(observation.assumed_ground_xy_m) - np.array(observation.oracle_ground_xy_m)
    assert float(np.linalg.norm(error)) == pytest.approx(0.2, abs=1e-9)


def test_a_box_beyond_the_range_limit_is_excluded() -> None:
    """Past 80 m one pixel of reconstruction error is metres of ground position."""

    near = observe(box_center_global=np.array([79.0, 0.0, 0.75]))
    far = observe(box_center_global=np.array([81.0, 0.0, 0.75]))

    assert near.valid
    assert not far.valid
    assert far.range_m > 80.0


def test_a_box_the_true_camera_cannot_see_is_excluded() -> None:
    """The oracle observation has to exist before a reconstruction of it can mean anything."""

    behind = observe(box_center_global=np.array([-10.0, 0.0, 0.75]))
    out_of_frame = observe(box_center_global=np.array([10.0, 40.0, 0.75]))

    assert not behind.valid
    assert not out_of_frame.valid
    assert all(math.isnan(value) for value in behind.true_uv)


def test_an_invalid_observation_reports_no_reconstructed_position() -> None:
    """`valid` is the only thing that says the numbers mean anything, so the rest says nothing."""

    observation = observe(box_center_global=np.array([-10.0, 0.0, 0.75]))

    assert not observation.valid
    assert all(math.isnan(value) for value in observation.assumed_ground_xy_m)


def test_an_assumed_calibration_whose_ray_misses_the_ground_is_invalid() -> None:
    """A large enough pitch fault turns a downward ray into one that never lands."""

    from bevcalib.geometry.se3 import compose

    tipped = compose(
        CAMERA_FROM_GLOBAL,
        SE3(rotation_wxyz=quaternion_about("y", 45.0), translation_xyz_m=(0.0, 0.0, 0.0)),
    )

    observation = observe(assumed_camera_from_global=tipped)

    assert not observation.valid


def test_the_observation_cannot_be_edited_after_the_fact() -> None:
    """It is a measurement, and measurements are frozen everywhere in this project."""

    import dataclasses

    with pytest.raises(dataclasses.FrozenInstanceError):
        observe().valid = False  # type: ignore[misc]


def test_the_ground_plane_comes_from_the_ego_height() -> None:
    """One documented place for the flat-world simplification, rather than a literal per call."""

    from bevcalib.geometry.frames import FramedTransform
    from bevcalib.operators.ground_contact import ground_plane_z_from_ego

    ego_pose = FramedTransform(
        target="global",
        source="camera_ego",
        value=SE3(rotation_wxyz=IDENTITY_QUATERNION, translation_xyz_m=(100.0, 200.0, 3.5)),
    )

    assert ground_plane_z_from_ego(ego_pose) == pytest.approx(3.5)


def test_the_ground_plane_must_come_from_a_pose_in_global() -> None:
    """A calibrated sensor transform has a z too, and it is not the ground height."""

    from bevcalib.geometry.frames import FramedTransform
    from bevcalib.operators.ground_contact import ground_plane_z_from_ego

    sensor = FramedTransform(
        target="camera_ego",
        source="camera_sensor",
        value=SE3(rotation_wxyz=IDENTITY_QUATERNION, translation_xyz_m=(1.7, 0.0, 1.5)),
    )

    with pytest.raises(ValueError, match="global"):
        ground_plane_z_from_ego(sensor)


# A camera 1.5 m up and 5 m along +x, so its origin has more than one non-zero
# component and a sign error in the range cannot cancel itself out.
OFFSET_CAMERA_FROM_GLOBAL = SE3(
    rotation_wxyz=matrix_to_quaternion(_CAMERA_ROTATION),
    translation_xyz_m=(0.0, 1.5, -5.0),
)


def observation_at(box_center: tuple[float, float, float], camera: SE3, **overrides: object):
    """One upright pedestrian-sized box, observed with the same calibration twice."""

    from bevcalib.operators.ground_contact import observe_ground_contact

    arguments: dict[str, object] = {
        "box_token": "token",
        "box_center_global": np.array(box_center),
        "size_wlh": (0.6, 0.6, 1.7),
        "orientation_wxyz": IDENTITY_QUATERNION,
        "true_camera_from_global": camera,
        "assumed_camera_from_global": camera,
        "intrinsic": INTRINSIC,
        "image_size_wh": IMAGE_SIZE,
        "ground_z_global": GROUND_Z,
    }
    arguments.update(overrides)
    return observe_ground_contact(**arguments)  # type: ignore[arg-type]


def test_a_pedestrian_sized_box_is_a_valid_box() -> None:
    """The bound on a box dimension is positivity, not a minimum of one metre.

    A pedestrian is about 0.6 m across and nuScenes is full of them, as well as
    of traffic cones and bicycles narrower still. Written `> 1.0` the guard
    would refuse most of the vulnerable road users the study exists to measure,
    and the refusal would name the box as malformed.
    """

    from bevcalib.operators.ground_contact import bottom_center_global

    contact = bottom_center_global(np.array([1.0, 2.0, 0.85]), (0.6, 0.6, 1.7), IDENTITY_QUATERNION)

    np.testing.assert_allclose(contact, [1.0, 2.0, 0.0], atol=1e-12)


def test_the_range_is_measured_from_the_camera_to_the_contact_point() -> None:
    """A sign error here reads as a distant object and is filtered out as one.

    The range gates which observations are usable at all, so getting it wrong
    silently changes the cohort rather than any single number. The camera used
    here is displaced along two axes: with the usual fixture, whose origin
    differs from the contact only in z, adding and subtracting give the same
    length and the error is invisible.

    The expected value is recomputed here from the two positions rather than
    recorded, so the test states the definition rather than an output.
    """

    from bevcalib.geometry.se3 import inverse

    observation = observation_at((12.0, 2.0, 0.85), OFFSET_CAMERA_FROM_GLOBAL)

    origin = np.asarray(inverse(OFFSET_CAMERA_FROM_GLOBAL).translation_xyz_m)
    contact = np.array([12.0, 2.0, GROUND_Z])
    assert observation.valid
    assert observation.range_m == pytest.approx(float(np.linalg.norm(contact - origin)))
    assert observation.range_m < float(np.linalg.norm(contact + origin))


def test_a_contact_at_exactly_the_range_limit_is_kept(monkeypatch: pytest.MonkeyPatch) -> None:
    """The range bound is exclusive above, so the limit itself is still usable.

    Written `>=` the observation exactly at the limit would be discarded, which
    moves the cohort boundary by one observation in every scene and does it
    silently, because a discarded observation is a legitimate outcome rather
    than an error.

    The limit is moved onto the measured range rather than the geometry being
    contrived to land on 80 m: no exact-arithmetic configuration puts a ground
    point exactly 80 m from a raised camera AND inside the image, and a range
    that is 80.000000000001 would be discarded by the correct code too.
    """

    from bevcalib.operators import ground_contact

    baseline = observation_at((12.0, 2.0, 0.85), OFFSET_CAMERA_FROM_GLOBAL)
    assert baseline.valid

    monkeypatch.setattr(ground_contact, "MAX_RANGE_M", baseline.range_m)
    at_the_limit = observation_at((12.0, 2.0, 0.85), OFFSET_CAMERA_FROM_GLOBAL)

    assert at_the_limit.range_m == baseline.range_m
    assert at_the_limit.valid


def test_a_ray_that_reaches_the_ground_at_unit_distance_still_reconstructs() -> None:
    """The distance bound refuses zero and behind, and one metre is neither.

    The pixel one focal length below the principal point looks down at exactly
    45 degrees, so with the camera 1.5 m up and the plane at 0.5 m the ray meets
    the ground at a distance of exactly one. Written `<= 1.0` the reconstruction
    would return `None` there and the caller would read it as a pixel above the
    horizon.
    """

    from bevcalib.operators.ground_contact import reconstruct_ground_contact

    contact = reconstruct_ground_contact((320.0, 1040.0), CAMERA_FROM_GLOBAL, INTRINSIC, 0.5)

    assert contact == pytest.approx((1.0, 0.0), abs=1e-12)


def test_a_plane_through_the_camera_itself_has_no_reconstruction() -> None:
    """A distance of exactly zero is not a contact point; it is the camera.

    Setting the ground plane at the camera's own height makes every ray meet it
    at zero distance. Written `< 0.0` the guard would let that through and
    return the camera's own position as a ground contact, which is a perfectly
    plausible pair of coordinates and wrong for every pixel at once.
    """

    from bevcalib.operators.ground_contact import reconstruct_ground_contact

    assert reconstruct_ground_contact((320.0, 1040.0), CAMERA_FROM_GLOBAL, INTRINSIC, 1.5) is None
