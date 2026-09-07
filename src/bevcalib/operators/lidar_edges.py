"""Depth discontinuities along LiDAR rings, and how well they line up with image edges."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy.ndimage import distance_transform_edt

from bevcalib.geometry.projection import BoolArray, Float64Array

# Protocol defaults. Both criteria must hold: half a metre is the beam spreading at
# forty metres, and eight percent is close-range noise at one metre. Either test
# alone therefore fires on ordinary surfaces.
DEFAULT_RELATIVE_JUMP = 0.08
DEFAULT_ABSOLUTE_JUMP_M = 0.5
DEFAULT_TRIM_QUANTILE = 0.90


@dataclass(frozen=True)
class EdgePointSet:
    """The returns that sit on a depth discontinuity, and where they came from."""

    points_lidar_n3: Float64Array
    source_indices: npt.NDArray[np.int64]
    ring_ids: npt.NDArray[np.int64]


def lidar_depth_edges(
    points_n5: Float64Array,
    *,
    relative_jump: float = DEFAULT_RELATIVE_JUMP,
    absolute_jump_m: float = DEFAULT_ABSOLUTE_JUMP_M,
) -> EdgePointSet:
    """Find the object silhouettes in one sweep of `[N, 5]` x, y, z, intensity, ring.

    Two returns are neighbours only if they share a ring and are adjacent in
    azimuth, so the points are grouped by ring and sorted by `atan2(y, x)`, stably
    by original index. A sweep does not arrive sorted, and treating file order as
    azimuth order invents edges everywhere.

    The FIRST and LAST return of a ring are not compared. A full revolution really
    does wrap, but a sweep may be a cropped arc and we cannot tell which we have.
    Wrapping a crop puts a false edge on every ring of every frame, always in the
    same place; not wrapping a full revolution costs at most one edge per ring.

    Of each qualifying pair the NEAR return is kept, because that is the object
    silhouette. The far return is background sitting behind it, and it projects
    past the boundary rather than onto it.
    """

    points = np.asarray(points_n5, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 5:
        raise ValueError(f"a sweep must have shape [N, 5], got {points.shape}")

    xyz = points[:, :3]
    ranges = np.linalg.norm(xyz, axis=1)
    ring_column = points[:, 4]
    usable = np.all(np.isfinite(xyz), axis=1) & np.isfinite(ring_column) & (ranges > 0.0)
    # Cast only after masking the non-finite rings: casting a NaN to an integer is
    # a silent garbage value, and here it would also raise on the warning filter.
    rings = np.where(np.isfinite(ring_column), ring_column, 0.0).astype(np.int64)
    azimuth = np.arctan2(points[:, 1], points[:, 0])

    usable_indices = np.flatnonzero(usable)
    selected: list[int] = []
    for ring in np.unique(rings[usable_indices]):
        members = usable_indices[rings[usable_indices] == ring]
        if members.size < 2:
            continue
        order = members[np.argsort(azimuth[members], kind="stable")]
        first, second = ranges[order[:-1]], ranges[order[1:]]
        jump = np.abs(second - first)
        qualifies = (jump >= absolute_jump_m) & (jump / np.minimum(first, second) >= relative_jump)
        nearer = np.where(first <= second, order[:-1], order[1:])
        selected.extend(int(value) for value in nearer[qualifies])

    source_indices = np.array(sorted(set(selected)), dtype=np.int64)
    return EdgePointSet(
        points_lidar_n3=xyz[source_indices].reshape(-1, 3),
        source_indices=source_indices,
        ring_ids=rings[source_indices],
    )


def trimmed_distance_transform_score(
    projected_uv: Float64Array,
    image_edges: BoolArray,
    trim_quantile: float = DEFAULT_TRIM_QUANTILE,
) -> float:
    """Score projected LiDAR edges against image edges: higher is better, zero is perfect.

    The score is the negated mean distance from each projected point to the
    nearest image edge, over the best `trim_quantile` of the points. Negated so
    that larger is better and an optimiser can maximise it; trimmed because a
    LiDAR sees silhouettes a camera cannot, and those points are occlusion rather
    than misalignment. Without the trim the tail would dominate the mean and a
    correct calibration would score no better than a wrong one.
    """

    edges = np.asarray(image_edges, dtype=bool)
    if edges.ndim != 2:
        raise ValueError(f"image edges must be a two-dimensional mask, got shape {edges.shape}")
    if not edges.any():
        raise ValueError("the image has no edges to measure against")

    return trimmed_distance_field_score(
        projected_uv, np.asarray(distance_transform_edt(~edges), dtype=np.float64), trim_quantile
    )


def trimmed_distance_field_score(
    projected_uv: Float64Array,
    distance_field: Float64Array,
    trim_quantile: float = DEFAULT_TRIM_QUANTILE,
) -> float:
    """Same score using a precomputed field; validate accessed distances in O(points).

    The image adapter computes the field once. Only sampled field entries enter
    this measurement, so validation does not scan every image pixel per candidate.
    """
    if not 0.0 < trim_quantile <= 1.0:
        raise ValueError(f"trim quantile must be within (0, 1], got {trim_quantile}")
    field = np.asarray(distance_field, dtype=np.float64)
    if field.ndim != 2 or 0 in field.shape:
        raise ValueError("distance field must be a nonempty two-dimensional array")
    uv = np.asarray(projected_uv, dtype=np.float64)
    if uv.ndim != 2 or uv.shape[1] != 2:
        raise ValueError(f"projected points must have shape [N, 2], got {uv.shape}")
    if uv.shape[0] == 0:
        raise ValueError("there are no projected points to score")
    if not np.all(np.isfinite(uv)):
        raise ValueError("projected points must be finite to be scored")

    height, width = field.shape
    columns = np.floor(uv[:, 0]).astype(np.int64)
    rows = np.floor(uv[:, 1]).astype(np.int64)
    if np.any((columns < 0) | (columns >= width) | (rows < 0) | (rows >= height)):
        raise ValueError("a projected point falls outside the image bounds")

    distances = field[rows, columns]
    if not np.isfinite(distances).all() or np.any(distances < 0):
        raise ValueError("sampled distance field values must be finite and nonnegative")
    keep = max(1, math.ceil(uv.shape[0] * trim_quantile))
    return -float(np.sort(distances)[:keep].mean())
