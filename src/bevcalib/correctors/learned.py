"""The learned corrector's input tensor and the stem adapted to carry it."""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from bevcalib.geometry.projection import BoolArray

# The same limit the ground-contact operator uses, so the depth encoding saturates
# exactly where the study stops reporting.
MAX_DEPTH_M = 80.0
CHANNEL_NAMES: tuple[str, ...] = ("red", "green", "blue", "depth", "valid")


def build_five_channel_input(
    rgb_chw: npt.NDArray[np.float32],
    projected_depth_hw: npt.NDArray[np.float32],
    valid_hw: BoolArray,
) -> npt.NDArray[np.float32]:
    """Stack colour, projected depth and a validity mask into `[5, H, W]`.

    Depth is encoded as `log1p(d) / log1p(80)` and clipped to one. A log scale
    because the returns that matter, silhouettes and near objects, are bunched at
    short range, and a linear encoding would spend most of its resolution on the
    far field the study excludes anyway.

    A pixel with no usable depth carries exactly zero in BOTH the depth and the
    validity channel. Zero is not a distance, and the validity channel is the only
    thing that distinguishes an unobserved pixel from one at the sensor. A depth
    that is not a positive finite number is treated as unobserved rather than
    passed through: a single NaN would spread through the first convolution to
    every pixel in its receptive field.
    """

    rgb = np.asarray(rgb_chw, dtype=np.float32)
    depth = np.asarray(projected_depth_hw, dtype=np.float32)
    valid = np.asarray(valid_hw, dtype=bool)

    if rgb.ndim != 3 or rgb.shape[0] != 3:
        raise ValueError(f"colour must have shape [3, H, W], got {rgb.shape}")
    image_shape = rgb.shape[1:]
    if depth.shape != image_shape or valid.shape != image_shape:
        raise ValueError(
            f"depth {depth.shape} and validity {valid.shape} must both match the image "
            f"{image_shape}"
        )
    if not np.all(np.isfinite(rgb)):
        raise ValueError("colour channels must be finite; there is no mask to hide one behind")

    usable = valid & np.isfinite(depth) & (depth > 0.0)
    tensor = np.zeros((5, *image_shape), dtype=np.float32)
    tensor[:3] = rgb
    tensor[3][usable] = np.clip(np.log1p(depth[usable]) / np.log1p(MAX_DEPTH_M), 0.0, 1.0).astype(
        np.float32
    )
    tensor[4] = usable.astype(np.float32)
    return tensor


def five_channel_stem_weight(existing_weight: npt.NDArray[Any]) -> npt.NDArray[np.float64]:
    """Widen a `[out, 3, k, k]` stem kernel to `[out, 5, k, k]`.

    Append two mean-RGB kernels, then scale each output's five-channel kernel to
    preserve its original RGB squared weight energy. This preserves convolution
    output variance for independent unit-variance inputs, not a claim that real
    RGB, projected depth and validity masks have identical distributions. A zero
    output kernel remains zero.
    """

    weight = np.asarray(existing_weight, dtype=np.float64)
    if weight.ndim != 4 or weight.shape[1] != 3:
        raise ValueError(
            f"the stem must be a three-channel convolution of shape [out, 3, k, k], "
            f"got {weight.shape}"
        )
    colour_mean = weight.mean(axis=1, keepdims=True)
    expanded = np.concatenate([weight, colour_mean, colour_mean], axis=1)
    original_energy = np.square(weight).sum(axis=(1, 2, 3), keepdims=True)
    expanded_energy = np.square(expanded).sum(axis=(1, 2, 3), keepdims=True)
    ratio = np.divide(
        original_energy,
        expanded_energy,
        out=np.zeros_like(original_energy),
        where=expanded_energy > 0,
    )
    return expanded * np.sqrt(ratio)


def _find_three_channel_stem(model: Any) -> tuple[str, Any]:
    from torch import nn

    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) and module.in_channels == 3:
            return name, module
    raise ValueError("the model has no three-channel convolution to adapt as a stem")


def initialize_convnextv2_five_channel(model: object) -> object:
    """Replace a ConvNeXtV2 stem with a five-channel one, in place, and return the model.

    Torch is imported inside the function so that the input builder above stays
    usable, and testable, without the training extra installed.
    """

    import torch
    from torch import nn

    name, stem = _find_three_channel_stem(model)
    adapted = nn.Conv2d(
        in_channels=5,
        out_channels=stem.out_channels,
        kernel_size=stem.kernel_size,
        stride=stem.stride,
        padding=stem.padding,
        bias=stem.bias is not None,
        device=stem.weight.device,
        dtype=stem.weight.dtype,
    )
    adapted_bias = adapted.bias
    with torch.no_grad():
        adapted.weight.copy_(
            torch.as_tensor(
                five_channel_stem_weight(stem.weight.detach().cpu().numpy()),
                dtype=adapted.weight.dtype,
            )
        )
        if stem.bias is not None and adapted_bias is not None:
            adapted_bias.copy_(stem.bias)

    parent = model
    parts = name.split(".")
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], adapted)
    return model
