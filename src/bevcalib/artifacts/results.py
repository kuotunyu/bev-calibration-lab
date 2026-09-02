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
