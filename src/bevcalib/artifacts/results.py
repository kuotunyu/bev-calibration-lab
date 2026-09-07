"""One calibration result: the row every figure in this study reduces to."""

from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

# JSON has no NaN or Infinity. Refusing them at the boundary is what keeps a
# single bad sample from turning an entire aggregate into NaN much later, in a
# place where the cause is no longer visible.
FiniteFloat = Annotated[float, Field(allow_inf_nan=False)]
NonNegativeFiniteFloat = Annotated[float, Field(allow_inf_nan=False, ge=0.0)]
Vector3 = tuple[FiniteFloat, FiniteFloat, FiniteFloat]


class CalibrationFaultModel(BaseModel):
    """A perturbation of the sensor calibration chain, or an estimate of one."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    rotation_rpy_deg: Vector3
    translation_xyz_m: Vector3
    requested_time_offset_ms: int


class CalibrationResultV1(BaseModel):
    """What one fault did to one sample, and how much of it a corrector recovered.

    Both the injected fault and the corrector's estimate are kept, rather than
    only their difference, so recovery can be recomputed from the artifact by a
    reader who does not trust the arithmetic that produced it.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["bev-calibration-result/v1"]
    sample_token: str = Field(min_length=1)
    scene_token: str = Field(min_length=1)
    fault: CalibrationFaultModel
    estimate: CalibrationFaultModel
    rotation_geodesic_error_deg: NonNegativeFiniteFloat
    translation_error_m: NonNegativeFiniteFloat
    pixel_error_median: NonNegativeFiniteFloat
    pixel_error_p90: NonNegativeFiniteFloat
    edge_alignment_score: NonNegativeFiniteFloat
    bev_ground_contact_error_m: NonNegativeFiniteFloat
    valid: bool
    invalid_reason: str | None

    @model_validator(mode="after")
    def validate_result_coherence(self) -> Self:
        """Reject rows that are internally impossible rather than merely surprising."""

        if self.pixel_error_p90 < self.pixel_error_median:
            raise ValueError("pixel_error_p90 must not be below pixel_error_median")
        if self.valid and self.invalid_reason is not None:
            raise ValueError("a valid result must not carry an invalid_reason")
        if not self.valid and not self.invalid_reason:
            raise ValueError("an invalid result requires a non-empty invalid_reason")
        return self


class SensorIdentity(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    token: str = Field(min_length=1)
    timestamp_us: int = Field(ge=0, strict=True)
    channel: Literal["CAM_FRONT", "LIDAR_TOP"]


class TimingEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    requested_offset_ms: int
    realized_offset_ms: FiniteFloat | None
    absolute_error_ms: NonNegativeFiniteFloat | None
    reason: Literal["nominal_pair", "valid", "outside_tolerance", "no_available_lidar"]


class PoseErrors(BaseModel):
    """Signed component errors and nonnegative pose magnitudes in explicit units."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    rotation_rpy_error_deg: Vector3
    translation_xyz_error_m: Vector3
    rotation_geodesic_error_deg: NonNegativeFiniteFloat
    translation_error_m: NonNegativeFiniteFloat


class GroundContactResult(BaseModel):
    """Per-object GT range survives aggregation, including unmeasurable objects."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    box_token: str = Field(min_length=1)
    range_m: NonNegativeFiniteFloat
    error_m: NonNegativeFiniteFloat | None
    invalid_reason: str | None

    @model_validator(mode="after")
    def coherent(self) -> Self:
        if (self.error_m is None and not self.invalid_reason) or (
            self.error_m is not None and self.invalid_reason is not None
        ):
            raise ValueError("ground-contact measurement and invalid reason disagree")
        return self


class CalibrationResultV2(BaseModel):
    """Formal measurements; method/run/protocol binding lives in the scene document.

    V1 is legacy-only. A null is unmeasured, never a perfect zero. Projection
    counts and per-object GT ranges preserve denominators and paired scene units.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: Literal["bev-calibration-result/v2"]
    sample_token: str = Field(min_length=1)
    scene_token: str = Field(min_length=1)
    fault_axis: Literal["roll", "pitch", "yaw", "x", "y", "z", "time"]
    fault_level: FiniteFloat
    fault: CalibrationFaultModel
    estimate: CalibrationFaultModel | None
    camera: SensorIdentity
    lidar: SensorIdentity | None
    timing: TimingEvidence
    pose: PoseErrors | None
    pixel_errors_px: tuple[NonNegativeFiniteFloat, ...]
    projection_count: int = Field(ge=0)
    edge_alignment_score: Annotated[float, Field(allow_inf_nan=False, le=0)] | None
    ground_contacts: tuple[GroundContactResult, ...]
    valid: bool
    invalid_reason: str | None

    @model_validator(mode="after")
    def coherent(self) -> Self:
        if self.camera.channel != "CAM_FRONT":
            raise ValueError("fixed camera must be CAM_FRONT")
        if self.timing.requested_offset_ms != self.fault.requested_time_offset_ms:
            raise ValueError("timing request differs from fault")
        if self.lidar is None:
            if (
                self.timing.reason != "no_available_lidar"
                or self.timing.realized_offset_ms is not None
                or self.timing.absolute_error_ms is not None
            ):
                raise ValueError("missing LiDAR must retain null timing evidence")
        else:
            if self.lidar.channel != "LIDAR_TOP" or self.lidar.token == self.camera.token:
                raise ValueError("selected sensor must be a distinct LIDAR_TOP")
            realized = (self.lidar.timestamp_us - self.camera.timestamp_us) / 1000
            error = abs(realized - self.timing.requested_offset_ms)
            if (
                self.timing.realized_offset_ms != realized
                or self.timing.absolute_error_ms != error
                or self.timing.reason == "no_available_lidar"
            ):
                raise ValueError("timing measurements disagree with actual sensor timestamps")
            if self.fault_axis == "time":
                expected = "valid" if error <= 25 else "outside_tolerance"
                if self.timing.reason != expected:
                    raise ValueError("timing reason disagrees with fixed tolerance")
            elif self.timing.reason != "nominal_pair":
                raise ValueError("metadata faults must retain the nominal sensor pair")
        if self.timing.reason in ("outside_tolerance", "no_available_lidar") and any(
            (
                self.estimate is not None,
                self.pose is not None,
                bool(self.pixel_errors_px),
                self.projection_count != 0,
                self.edge_alignment_score is not None,
                bool(self.ground_contacts),
            )
        ):
            raise ValueError("unavailable timing requires unmeasured operators and zero counts")
        if len(self.pixel_errors_px) > self.projection_count:
            raise ValueError("projection valid count exceeds available count")
        if len({box.box_token for box in self.ground_contacts}) != len(self.ground_contacts):
            raise ValueError("duplicate ground-contact object")
        if self.valid:
            if self.timing.reason in ("outside_tolerance", "no_available_lidar"):
                raise ValueError("invalid timing cannot be a valid result")
            if (
                self.estimate is None
                or self.pose is None
                or not self.pixel_errors_px
                or self.edge_alignment_score is None
                or not any(box.error_m is not None for box in self.ground_contacts)
            ):
                raise ValueError("a valid row requires complete measurable operators")
            if self.invalid_reason is not None:
                raise ValueError("valid result cannot carry invalid reason")
        elif not self.invalid_reason:
            raise ValueError("invalid result needs an explanation")
        return self
