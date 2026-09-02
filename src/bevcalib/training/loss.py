"""The loss that trains the learned corrector."""

from __future__ import annotations

from typing import Any

# The widest fault the study injects, used to put rotation and translation on one
# scale. An error at the edge of the range costs the same on either.
DEFAULT_ROTATION_SCALE_DEG = 2.0
DEFAULT_TRANSLATION_SCALE_M = 0.2
DEFAULT_DELTA = 1.0


def normalized_huber_loss(
    prediction_n6: Any,
    target_n6: Any,
    rotation_scale_deg: float = DEFAULT_ROTATION_SCALE_DEG,
    translation_scale_m: float = DEFAULT_TRANSLATION_SCALE_M,
    delta: float = DEFAULT_DELTA,
) -> Any:
    """Huber loss over six degrees of freedom, after dividing each by its own scale.

    Rotation is in degrees and translation in metres, so without normalisation the
    loss would be dominated by whichever unit happened to carry the larger numbers:
    two degrees and twenty centimetres are the same size of mistake in this study,
    and they differ by a factor of ten as raw numbers.

    Normalisation happens HERE and nowhere else. The dataset emits its target in
    physical units so that a reader can check a target against the fault it came
    from by eye, and only this function needs to remember the scales.

    Huber rather than squared error because one wild sample, a frame where the
    LiDAR saw almost nothing, must not dominate a batch.

    Torch is imported inside the function so importing the package does not require
    the training extra.
    """

    import torch

    if rotation_scale_deg <= 0.0 or translation_scale_m <= 0.0 or delta <= 0.0:
        raise ValueError(
            "the scales and delta must be positive, got "
            f"{rotation_scale_deg}, {translation_scale_m} and {delta}"
        )

    prediction = torch.as_tensor(prediction_n6)
    target = torch.as_tensor(target_n6)
    if prediction.shape != target.shape:
        raise ValueError(
            f"prediction {tuple(prediction.shape)} and target {tuple(target.shape)} "
            "must have the same shape"
        )
    if prediction.ndim != 2 or prediction.shape[1] != 6:
        raise ValueError(
            f"predictions must have shape [N, 6] in the order roll, pitch, yaw, x, y, z, "
            f"got {tuple(prediction.shape)}"
        )

    scales = torch.tensor(
        [rotation_scale_deg] * 3 + [translation_scale_m] * 3,
        dtype=prediction.dtype,
        device=prediction.device,
    )
    error = (prediction - target) / scales
    magnitude = error.abs()
    quadratic = 0.5 * error.pow(2)
    linear = delta * (magnitude - 0.5 * delta)
    return torch.where(magnitude <= delta, quadratic, linear).mean()
