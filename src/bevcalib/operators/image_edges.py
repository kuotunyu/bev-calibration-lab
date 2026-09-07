"""The fixed, native-resolution image-edge algorithm; no cohort-fitted threshold."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from PIL import Image
from scipy.ndimage import distance_transform_edt, sobel

IMAGE_EDGE_POLICY = "bev-image-edges/v1:pillow-RGB-to-L:float64-sobel-axes01-reflect:hypot:positive-quantile0.90-linear:positive-and-ge"


@dataclass(frozen=True)
class ImageEdgeEvidence:
    mask: npt.NDArray[np.bool_]
    threshold: float | None
    distance_field: npt.NDArray[np.float64] | None
    policy: str = IMAGE_EDGE_POLICY


def image_edge_evidence(rgb: npt.NDArray[np.uint8]) -> ImageEdgeEvidence:
    if rgb.ndim != 3 or rgb.shape[2] != 3 or 0 in rgb.shape or rgb.dtype != np.uint8:
        raise ValueError("image edges require a nonempty RGB uint8 image")
    gray = np.asarray(Image.fromarray(rgb).convert("L"), dtype=np.float64)
    magnitude = np.hypot(sobel(gray, axis=0, mode="reflect"), sobel(gray, axis=1, mode="reflect"))
    positive = magnitude[magnitude > 0]
    if not positive.size:
        return ImageEdgeEvidence(np.zeros(gray.shape, dtype=bool), None, None)
    threshold = float(np.quantile(positive, 0.90, method="linear"))
    mask = (magnitude > 0) & (magnitude >= threshold)
    return ImageEdgeEvidence(
        mask, threshold, np.asarray(distance_transform_edt(~mask), dtype=np.float64)
    )
