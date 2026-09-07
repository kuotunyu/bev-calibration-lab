"""Versioned measurement policy and native input identity, without runtime adapters."""

from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, Field

from bevcalib.artifacts.results import NonNegativeFiniteFloat
from bevcalib.cohort.manifest import Digest

IMAGE_EDGE_POLICY: Final = "bev-image-edges/v1:pillow-RGB-to-L:float64-sobel-axes01-reflect:hypot:positive-quantile0.90-linear:positive-and-ge"


class ImageMeasurement(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    rgb_sha256: Digest
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    edge_threshold: NonNegativeFiniteFloat | None


class EvaluationMeasurements(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: Literal["bev-calibration-measurements/v1"] = "bev-calibration-measurements/v1"
    image_edge_policy: Literal[
        "bev-image-edges/v1:pillow-RGB-to-L:float64-sobel-axes01-reflect:hypot:positive-quantile0.90-linear:positive-and-ge"
    ] = IMAGE_EDGE_POLICY
    native_pixel_coordinates: Literal[True] = True
    empty_projection_search_policy: Literal["minus-native-image-diagonal;never-measurement"] = (
        "minus-native-image-diagonal;never-measurement"
    )
    table_sha256: dict[str, Digest]
    images: dict[str, ImageMeasurement]
