"""Finite estimates and disclosed support for the predeclared scene estimands."""

from __future__ import annotations

import math
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from bevcalib.artifacts.results import FiniteFloat
from bevcalib.metrics.bootstrap import BootstrapInterval


class Support(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    total_frames: int = Field(ge=0)
    frames: int = Field(ge=0)
    excluded_frames: int = Field(ge=0)
    total_scenes: int = Field(ge=0)
    scenes: int = Field(ge=0)
    total_objects: int | None = Field(default=None, ge=0)
    objects: int | None = Field(default=None, ge=0)
    excluded_objects: int | None = Field(default=None, ge=0)
    exclusions: dict[str, int]

    @model_validator(mode="after")
    def coherent(self) -> Self:
        if (
            self.frames + self.excluded_frames != self.total_frames
            or self.scenes > self.total_scenes
            or self.scenes > self.frames
            or (self.scenes == 0) != (self.frames == 0)
            or any(count < 0 for count in self.exclusions.values())
        ):
            raise ValueError("incoherent frame/scene support")
        if self.total_objects is None:
            if self.objects is not None or self.excluded_objects is not None:
                raise ValueError("incoherent object support")
        elif (
            self.objects is None
            or self.excluded_objects is None
            or self.objects + self.excluded_objects != self.total_objects
            or self.objects < self.frames
        ):
            raise ValueError("incoherent object support")
        return self


class Estimate(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    value: FiniteFloat | None
    support: Support
    reason: str | None

    @model_validator(mode="after")
    def coherent(self) -> Self:
        if (self.value is None) != (self.support.frames == 0) or (self.value is None) != bool(
            self.reason
        ):
            raise ValueError("null estimate requires empty support and reason")
        return self


class PairedEstimate(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    before: FiniteFloat | None
    after: FiniteFloat | None
    improvement: FiniteFloat | None
    interval: BootstrapInterval | None
    support: Support
    reason: str | None

    @model_validator(mode="after")
    def coherent(self) -> Self:
        missing = self.support.frames == 0
        if (
            any(
                (value is None) != missing
                for value in (self.before, self.after, self.improvement, self.interval)
            )
            or bool(self.reason) != missing
        ):
            raise ValueError("paired null estimate requires empty support and reason")
        if self.interval is not None and (
            self.interval.estimate != self.improvement
            or self.interval.resamples != 5000
            or self.interval.seed != 20260831
            or self.interval.confidence != 0.95
            or self.interval.low > self.interval.high
            or not all(math.isfinite(value) for value in (self.interval.low, self.interval.high))
        ):
            raise ValueError("interval differs from the fixed bootstrap contract")
        return self
