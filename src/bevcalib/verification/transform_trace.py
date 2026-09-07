"""Typed evidence that a nuScenes transform/projection check was executed."""

from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, StrictFloat, StrictInt, model_validator

from bevcalib.geometry.frames import FrameName

TimestampField = Literal["lidar_timestamp_us", "camera_timestamp_us"]
FiniteFloat = Annotated[StrictFloat, Field(allow_inf_nan=False)]
PARITY_TOLERANCE = 1e-6
PARITY_CASES = frozenset(
    {
        "lidar_projection:sample_0",
        "lidar_projection:sample_1",
        "box_centers:sample_0",
        "box_centers:sample_1",
    }
)
EXPECTED_CHAIN = (
    ("T_lidar_ego_lidar_sensor", "lidar_ego", "lidar_sensor", None),
    ("T_global_lidar_ego", "global", "lidar_ego", "lidar_timestamp_us"),
    ("T_camera_ego_global", "camera_ego", "global", "camera_timestamp_us"),
    ("T_camera_sensor_camera_ego", "camera_sensor", "camera_ego", None),
)


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class NativeLookup(_Strict):
    """The repository adapter resolves channel pairs from scene records."""

    sample_pair_source: Literal["scene_records"]
    camera_channel: Literal["CAM_FRONT"]
    lidar_channel: Literal["LIDAR_TOP"]


class SensorTimestamps(_Strict):
    """Actual sensor timestamps; neither pose may borrow the other's time."""

    lidar_timestamp_us: Annotated[StrictInt, Field(ge=0)]
    camera_timestamp_us: Annotated[StrictInt, Field(ge=0)]


class TransformEdge(_Strict):
    """One directed, typed link in the LiDAR-to-camera chain."""

    name: str = Field(min_length=1)
    target: FrameName
    source: FrameName
    timestamp_field: TimestampField | None


class BehindCameraTest(_Strict):
    """A point may land inside the image numerically and still be invalid."""

    camera_point_xyz: tuple[FiniteFloat, FiniteFloat, FiniteFloat]
    in_front: bool
    in_image: bool
    valid: bool

    @model_validator(mode="after")
    def proves_depth_mask(self) -> Self:
        if self.camera_point_xyz[2] >= 0 or self.in_front or not self.in_image or self.valid:
            raise ValueError(
                "behind-camera test must use negative optical depth, remain in-image, "
                "and be invalid because in_front is false"
            )
        return self


class ParityCase(_Strict):
    """One nonempty comparison against the official devkit."""

    case_id: Literal[
        "lidar_projection:sample_0",
        "lidar_projection:sample_1",
        "box_centers:sample_0",
        "box_centers:sample_1",
    ]
    comparison_count: Annotated[StrictInt, Field(gt=0)]
    max_abs_error: Annotated[FiniteFloat, Field(ge=0)]


class DevkitParity(_Strict):
    """Absolute-only parity across both pinned samples and both operators."""

    absolute_tolerance: FiniteFloat
    relative_tolerance: FiniteFloat
    cases: tuple[ParityCase, ...]

    @model_validator(mode="after")
    def complete_and_within_tolerance(self) -> Self:
        if self.absolute_tolerance != PARITY_TOLERANCE:
            raise ValueError("absolute tolerance must be exactly 1e-6")
        if self.relative_tolerance != 0.0:
            raise ValueError("relative tolerance must be zero")
        case_ids = [case.case_id for case in self.cases]
        if len(case_ids) != 4 or set(case_ids) != PARITY_CASES:
            raise ValueError("trace must contain exactly the four pinned parity cases")
        failures = [
            case.case_id for case in self.cases if case.max_abs_error > self.absolute_tolerance
        ]
        if failures:
            raise ValueError(f"parity gap exceeds absolute tolerance for {failures}")
        return self


class TransformVerificationTrace(_Strict):
    """Complete evidence for native resolution, geometry, masking and parity."""

    schema_version: Literal["bev-transform-verification/v1"]
    native_lookup: NativeLookup
    timestamps_us: SensorTimestamps
    point_array_shape: tuple[Literal["N"], Literal[3]]
    chain: tuple[TransformEdge, ...]
    behind_camera_test: BehindCameraTest
    parity: DevkitParity

    @model_validator(mode="after")
    def directed_chain(self) -> Self:
        observed = tuple(
            (edge.name, edge.target, edge.source, edge.timestamp_field) for edge in self.chain
        )
        if observed != EXPECTED_CHAIN:
            raise ValueError(
                "trace must contain the directed four-link chain from lidar_sensor "
                "through both timestamped ego frames to camera_sensor"
            )
        return self
