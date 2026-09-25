"""The whole edge-alignment chain, from a synthetic sweep to a score.

This is the first test that runs the pieces together in the order the study will:
find LiDAR depth edges, carry them through a calibration that may be faulty,
project them into the camera, and score them against the image edges. It exists to
check the one claim the study rests on, that a calibration fault makes the score
worse, before any of it is pointed at real data.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from bevcalib.artifacts.results import CalibrationFaultModel
from bevcalib.geometry.projection import project_camera
from bevcalib.geometry.quaternions import matrix_to_quaternion
from bevcalib.geometry.se3 import SE3, transform_points
from bevcalib.operators.lidar_edges import lidar_depth_edges, trimmed_distance_transform_score
from bevcalib.perturbations.apply import apply_metadata_fault

INTRINSIC = np.array([[800.0, 0.0, 320.0], [0.0, 800.0, 240.0], [0.0, 0.0, 1.0]])
IMAGE_SIZE = (640, 480)

# LiDAR looks along +x with +z up; the camera looks along +z with +y down. This is
# the true extrinsic the faults are applied to.
CAMERA_FROM_LIDAR = SE3(
    rotation_wxyz=matrix_to_quaternion(
        np.array([[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]])
    ),
    translation_xyz_m=(0.0, 0.0, 0.0),
)


def synthetic_sweep() -> np.ndarray:
    """Two rings sweeping across the camera's field of view, each with one near object.

    Ranges step gently except where the object silhouette is, so exactly the
    silhouette returns qualify as depth edges.
    """

    rows: list[list[float]] = []
    for ring, elevation_deg in enumerate((-2.0, 2.0)):
        for index, azimuth_deg in enumerate(np.linspace(-15.0, 15.0, 31)):
            # A near object occupies the middle third of the arc.
            range_m = 6.0 if 10 <= index <= 20 else 25.0
            azimuth = math.radians(azimuth_deg)
            elevation = math.radians(elevation_deg)
            rows.append(
                [
                    range_m * math.cos(elevation) * math.cos(azimuth),
                    range_m * math.cos(elevation) * math.sin(azimuth),
                    range_m * math.sin(elevation),
                    12.0,
                    float(ring),
                ]
            )
    return np.array(rows, dtype=np.float64)


def score_for(fault: CalibrationFaultModel, sweep: np.ndarray, image_edges: np.ndarray) -> float:
    assumed = apply_metadata_fault(CAMERA_FROM_LIDAR, fault)
    edges = lidar_depth_edges(sweep)
    camera_points = transform_points(assumed, edges.points_lidar_n3)
    projection = project_camera(camera_points, INTRINSIC, IMAGE_SIZE)
    return trimmed_distance_transform_score(projection.uv[projection.valid], image_edges)


def zero_fault() -> CalibrationFaultModel:
    return CalibrationFaultModel(
        rotation_rpy_deg=(0.0, 0.0, 0.0),
        translation_xyz_m=(0.0, 0.0, 0.0),
        requested_time_offset_ms=0,
    )


@pytest.fixture(scope="module")
def aligned_scene() -> tuple[np.ndarray, np.ndarray]:
    """A sweep, and the image edges that the true calibration puts the LiDAR edges on."""

    sweep = synthetic_sweep()
    edges = lidar_depth_edges(sweep)
    truth = project_camera(
        transform_points(CAMERA_FROM_LIDAR, edges.points_lidar_n3), INTRINSIC, IMAGE_SIZE
    )
    assert truth.valid.all(), "the synthetic scene must land entirely inside the image"

    image_edges = np.zeros((IMAGE_SIZE[1], IMAGE_SIZE[0]), dtype=bool)
    for u, v in truth.uv:
        image_edges[int(v), int(u)] = True
    return sweep, image_edges


def test_the_silhouette_is_what_the_operator_finds(aligned_scene) -> None:  # type: ignore[no-untyped-def]
    """Four edges: the near and far boundary of the object, on each of the two rings."""

    sweep, _ = aligned_scene

    edges = lidar_depth_edges(sweep)

    assert edges.points_lidar_n3.shape == (4, 3)
    assert sorted(edges.ring_ids.tolist()) == [0, 0, 1, 1]


def test_the_true_calibration_scores_perfectly(aligned_scene) -> None:  # type: ignore[no-untyped-def]
    """By construction the image edges are where the true chain puts the LiDAR edges."""

    sweep, image_edges = aligned_scene

    assert score_for(zero_fault(), sweep, image_edges) == pytest.approx(0.0)


@pytest.mark.parametrize("yaw_deg", [0.25, 1.0, 2.0])
def test_a_yaw_fault_makes_the_alignment_score_worse(aligned_scene, yaw_deg: float) -> None:  # type: ignore[no-untyped-def]
    """The claim the whole study rests on, checked before any real data is touched."""

    sweep, image_edges = aligned_scene
    faulted = CalibrationFaultModel(
        rotation_rpy_deg=(0.0, 0.0, yaw_deg),
        translation_xyz_m=(0.0, 0.0, 0.0),
        requested_time_offset_ms=0,
    )

    assert score_for(faulted, sweep, image_edges) < score_for(zero_fault(), sweep, image_edges)


def test_a_larger_yaw_fault_scores_worse_than_a_smaller_one(aligned_scene) -> None:  # type: ignore[no-untyped-def]
    """Monotonicity is what makes the score usable as the classical optimiser's objective."""

    sweep, image_edges = aligned_scene

    def yaw(degrees: float) -> float:
        return score_for(
            CalibrationFaultModel(
                rotation_rpy_deg=(0.0, 0.0, degrees),
                translation_xyz_m=(0.0, 0.0, 0.0),
                requested_time_offset_ms=0,
            ),
            sweep,
            image_edges,
        )

    assert yaw(2.0) < yaw(1.0) < yaw(0.25) < yaw(0.0)


def test_the_fault_leaves_the_sweep_bytes_untouched(aligned_scene) -> None:  # type: ignore[no-untyped-def]
    """End to end, the study still perturbs metadata only."""

    import hashlib

    sweep, image_edges = aligned_scene
    before = hashlib.sha256(sweep.tobytes()).hexdigest()

    score_for(
        CalibrationFaultModel(
            rotation_rpy_deg=(2.0, -1.0, 0.5),
            translation_xyz_m=(0.2, -0.1, 0.05),
            requested_time_offset_ms=0,
        ),
        sweep,
        image_edges,
    )

    assert hashlib.sha256(sweep.tobytes()).hexdigest() == before
