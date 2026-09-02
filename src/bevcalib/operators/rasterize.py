"""Turn projected points into a sparse depth image, keeping the nearest surface."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from bevcalib.geometry.projection import BoolArray, Float64Array, validate_image_size


def rasterize_min_depth(
    uv: Float64Array,
    depth: Float64Array,
    valid: BoolArray,
    image_size_wh: tuple[int, int],
) -> tuple[npt.NDArray[np.float32], BoolArray]:
    """Rasterise points into `(depth_image, observed)` of shape `[height, width]`.

    Several LiDAR returns land on one pixel and only the nearest is the surface
    the camera saw, so the minimum positive depth wins. `np.minimum.at` does that
    unbuffered, which is what makes the result independent of the order the
    points arrive in; sweep order is an artefact of the sensor and must not reach
    a metric.

    An unobserved pixel is zero, and `observed` is the only thing that says
    whether the number means anything. Zero is not a depth.
    """

    pixels = np.asarray(uv, dtype=np.float64)
    depths = np.asarray(depth, dtype=np.float64)
    flags = np.asarray(valid, dtype=bool)
    if pixels.ndim != 2 or pixels.shape[1] != 2:
        raise ValueError(f"uv must have shape [N, 2], got {pixels.shape}")
    if depths.shape != (pixels.shape[0],) or flags.shape != (pixels.shape[0],):
        raise ValueError(
            "uv, depth and valid must describe the same points, got "
            f"{pixels.shape}, {depths.shape} and {flags.shape}"
        )
    width, height = validate_image_size(image_size_wh)

    # A non-finite coordinate would survive `floor` and then become a silent
    # garbage index in `astype`, so it is excluded here rather than checked later.
    drawable = (
        flags
        & np.isfinite(depths)
        & (depths > 0.0)
        & np.isfinite(pixels[:, 0])
        & np.isfinite(pixels[:, 1])
    )
    columns = np.floor(pixels[drawable, 0]).astype(np.int64)
    rows = np.floor(pixels[drawable, 1]).astype(np.int64)
    if np.any((columns < 0) | (columns >= width) | (rows < 0) | (rows >= height)):
        # Negative indices wrap in numpy, which would draw the point on the
        # opposite edge of the image instead of failing.
        raise ValueError("a drawable point falls outside the image bounds")

    canvas = np.full(height * width, np.inf, dtype=np.float64)
    np.minimum.at(canvas, rows * width + columns, depths[drawable])
    observed = np.isfinite(canvas)
    canvas[~observed] = 0.0
    return canvas.reshape(height, width).astype(np.float32), observed.reshape(height, width)
