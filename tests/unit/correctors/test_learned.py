"""Contracts for the learned corrector's input tensor and its adapted stem.

The input builder is pure NumPy on purpose, so the channel order and the depth
encoding can be checked without installing a deep-learning framework. Channel
order is the kind of thing that produces a model which trains, converges, and is
measuring something else entirely.
"""

from __future__ import annotations

import numpy as np
import pytest

HEIGHT, WIDTH = 4, 6


def scene() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """An RGB image, a sparse depth image, and the mask saying which depths are real."""

    rgb = np.zeros((3, HEIGHT, WIDTH), dtype=np.float32)
    rgb[0] = 0.25  # red
    rgb[1] = 0.50  # green
    rgb[2] = 0.75  # blue
    depth = np.zeros((HEIGHT, WIDTH), dtype=np.float32)
    valid = np.zeros((HEIGHT, WIDTH), dtype=bool)
    depth[1, 2] = 10.0
    valid[1, 2] = True
    return rgb, depth, valid


def test_the_five_channels_are_red_green_blue_depth_then_valid() -> None:
    """A silently permuted channel would train happily and measure something else."""

    from bevcalib.correctors.learned import build_five_channel_input

    rgb, depth, valid = scene()

    tensor = build_five_channel_input(rgb, depth, valid)

    assert tensor.shape == (5, HEIGHT, WIDTH)
    assert tensor.dtype == np.float32
    np.testing.assert_allclose(tensor[0], 0.25)
    np.testing.assert_allclose(tensor[1], 0.50)
    np.testing.assert_allclose(tensor[2], 0.75)
    assert tensor[3, 1, 2] > 0.0
    assert tensor[4, 1, 2] == 1.0


def test_the_valid_channel_is_one_where_there_is_a_depth_and_zero_elsewhere() -> None:
    """Sparse LiDAR leaves most pixels empty; the network has to be told which."""

    from bevcalib.correctors.learned import build_five_channel_input

    rgb, depth, valid = scene()

    tensor = build_five_channel_input(rgb, depth, valid)

    assert tensor[4].sum() == 1.0
    assert tensor[4, 0, 0] == 0.0


def test_an_unobserved_pixel_carries_a_depth_of_exactly_zero() -> None:
    """Zero is not a depth, and the valid channel is what distinguishes it from one."""

    from bevcalib.correctors.learned import build_five_channel_input

    rgb, depth, valid = scene()
    depth[0, 0] = 999.0  # left over in the buffer, but never observed

    tensor = build_five_channel_input(rgb, depth, valid)

    assert tensor[3, 0, 0] == 0.0
    assert tensor[4, 0, 0] == 0.0


def test_depth_is_encoded_on_a_log_scale_normalised_to_the_range_limit() -> None:
    """Hand-computable: `log1p(d) / log1p(80)`, so 80 m maps to exactly one.

    A log scale because the returns that matter, silhouettes and near objects,
    are bunched at short range, and a linear encoding would spend most of its
    resolution on the far field where the study excludes points anyway.
    """

    from bevcalib.correctors.learned import MAX_DEPTH_M, build_five_channel_input

    rgb, depth, valid = scene()
    for metres in (1.0, 10.0, 80.0):
        depth[1, 2] = metres
        tensor = build_five_channel_input(rgb, depth, valid)
        assert tensor[3, 1, 2] == pytest.approx(np.log1p(metres) / np.log1p(MAX_DEPTH_M), abs=1e-6)
    assert MAX_DEPTH_M == 80.0


def test_a_depth_beyond_the_range_limit_saturates_rather_than_exceeding_one() -> None:
    """Every other channel lives in [0, 1]; the depth channel must not be the exception."""

    from bevcalib.correctors.learned import build_five_channel_input

    rgb, depth, valid = scene()
    depth[1, 2] = 400.0

    tensor = build_five_channel_input(rgb, depth, valid)

    assert tensor[3, 1, 2] == pytest.approx(1.0)


@pytest.mark.parametrize("bad_depth", [float("nan"), float("inf"), -1.0, 0.0])
def test_a_depth_that_is_not_a_positive_distance_is_treated_as_unobserved(
    bad_depth: float,
) -> None:
    """A NaN would spread through a convolution to every pixel in its receptive field."""

    from bevcalib.correctors.learned import build_five_channel_input

    rgb, depth, valid = scene()
    depth[1, 2] = bad_depth

    tensor = build_five_channel_input(rgb, depth, valid)

    assert tensor[3, 1, 2] == 0.0
    assert tensor[4, 1, 2] == 0.0
    assert np.all(np.isfinite(tensor))


@pytest.mark.parametrize(
    ("rgb_shape", "depth_shape", "valid_shape", "message"),
    [
        (
            (4, HEIGHT, WIDTH),
            (HEIGHT, WIDTH),
            (HEIGHT, WIDTH),
            rf"^colour must have shape \[3, H, W\], got \(4, {HEIGHT}, {WIDTH}\)$",
        ),
        (
            (3, HEIGHT),
            (HEIGHT, WIDTH),
            (HEIGHT, WIDTH),
            rf"^colour must have shape \[3, H, W\], got \(3, {HEIGHT}\)$",
        ),
        (
            (3, HEIGHT, WIDTH),
            (HEIGHT, WIDTH + 1),
            (HEIGHT, WIDTH),
            rf"^depth \({HEIGHT}, {WIDTH + 1}\) and validity \({HEIGHT}, {WIDTH}\) must both "
            rf"match the image \({HEIGHT}, {WIDTH}\)",
        ),
        (
            (3, HEIGHT, WIDTH),
            (HEIGHT, WIDTH),
            (HEIGHT + 1, WIDTH),
            rf"^depth \({HEIGHT}, {WIDTH}\) and validity \({HEIGHT + 1}, {WIDTH}\) must both "
            rf"match the image \({HEIGHT}, {WIDTH}\)",
        ),
    ],
)
def test_inputs_that_do_not_describe_one_image_are_rejected(
    rgb_shape: tuple[int, ...],
    depth_shape: tuple[int, ...],
    valid_shape: tuple[int, ...],
    message: str,
) -> None:
    """Mismatched shapes would broadcast into a tensor nobody intended.

    Each row names its message, and each message carries the shapes it saw. Two
    separate checks run here — the colour tensor's own rank and channel count,
    then the depth and validity maps against the image the colour tensor
    describes — and a bare `pytest.raises(ValueError)` cannot tell which of them
    fired. That matters when the two disagree: a caller who cropped the depth
    map but not the image needs to be sent to the depth map, and a caller who
    passed an RGBA image needs to be sent to the channel count.
    """

    from bevcalib.correctors.learned import build_five_channel_input

    with pytest.raises(ValueError, match=message):
        build_five_channel_input(
            np.zeros(rgb_shape, dtype=np.float32),
            np.zeros(depth_shape, dtype=np.float32),
            np.zeros(valid_shape, dtype=bool),
        )


def test_a_colour_channel_that_is_not_finite_is_rejected() -> None:
    """Unlike depth there is no mask for colour, so a NaN here has nowhere to go."""

    from bevcalib.correctors.learned import build_five_channel_input

    rgb, depth, valid = scene()
    rgb[0, 0, 0] = np.float32("nan")

    with pytest.raises(
        ValueError, match=r"^colour channels must be finite; there is no mask to hide one behind$"
    ):
        build_five_channel_input(rgb, depth, valid)


def test_the_new_stem_keeps_the_pretrained_colour_kernels_untouched() -> None:
    """The whole point of adapting rather than rebuilding is to keep what was learned."""

    from bevcalib.correctors.learned import five_channel_stem_weight

    existing = np.arange(96 * 3 * 4 * 4, dtype=np.float64).reshape(96, 3, 4, 4)

    adapted = five_channel_stem_weight(existing)

    assert adapted.shape == (96, 5, 4, 4)
    np.testing.assert_array_equal(adapted[:, :3], existing)


def test_each_new_stem_channel_starts_as_the_average_of_the_colour_channels() -> None:
    """A zero kernel would learn from nothing; a random one would inject noise into a
    pretrained stem. The colour mean starts each new channel as another intensity-like
    input, which is the closest thing to what the stem already knows how to read.

    It does raise the stem's total response to a uniform input by two thirds, since
    five channels now carry what three did. That is left as it is because the layer
    that follows is normalised, and because rescaling would change the pretrained
    colour response this adaptation exists to preserve.
    """

    from bevcalib.correctors.learned import five_channel_stem_weight

    existing = np.random.default_rng(20260902).normal(size=(96, 3, 4, 4))

    adapted = five_channel_stem_weight(existing)

    colour_mean = existing.mean(axis=1)
    np.testing.assert_allclose(adapted[:, 3], colour_mean, atol=1e-15)
    np.testing.assert_allclose(adapted[:, 4], colour_mean, atol=1e-15)


@pytest.mark.parametrize("shape", [(96, 4, 4, 4), (96, 3, 4), (3, 4, 4)])
def test_a_stem_that_is_not_a_three_channel_convolution_is_rejected(
    shape: tuple[int, ...],
) -> None:
    """Adapting a stem that is already five channels would silently do it twice."""

    from bevcalib.correctors.learned import five_channel_stem_weight

    with pytest.raises(
        ValueError,
        match=r"^the stem must be a three-channel convolution of shape \[out, 3, k, k\]",
    ):
        five_channel_stem_weight(np.zeros(shape))


torch = pytest.importorskip("torch", reason="adapting a real stem needs the train extra")


def convnext_like():  # type: ignore[no-untyped-def]
    """A model shaped like ConvNeXtV2-Tiny's first two layers, stem included."""

    from torch import nn

    class Embeddings(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            # ConvNeXtV2-Tiny's stem: 96 features, 4x4 kernel, stride 4.
            self.patch_embeddings = nn.Conv2d(3, 96, kernel_size=4, stride=4)
            self.layernorm = nn.GroupNorm(1, 96)

    class Backbone(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embeddings = Embeddings()
            self.head = nn.Conv2d(96, 6, kernel_size=1)

        def forward(self, tensor):  # type: ignore[no-untyped-def]
            hidden = self.embeddings.layernorm(self.embeddings.patch_embeddings(tensor))
            return self.head(hidden).mean(dim=(2, 3))

    return Backbone()


def test_adapting_the_stem_widens_it_to_five_channels_in_place() -> None:
    """The model keeps its identity; only the stem's input width changes."""

    from bevcalib.correctors.learned import initialize_convnextv2_five_channel

    model = convnext_like()

    returned = initialize_convnextv2_five_channel(model)

    assert returned is model
    assert model.embeddings.patch_embeddings.in_channels == 5
    assert tuple(model.embeddings.patch_embeddings.weight.shape) == (96, 5, 4, 4)


def test_the_adapted_stem_carries_the_pretrained_weights_and_bias_across() -> None:
    """An adaptation that quietly reinitialised the stem would throw away the pretraining."""

    from bevcalib.correctors.learned import initialize_convnextv2_five_channel

    model = convnext_like()
    original = model.embeddings.patch_embeddings.weight.detach().clone()
    original_bias = model.embeddings.patch_embeddings.bias.detach().clone()

    initialize_convnextv2_five_channel(model)

    adapted = model.embeddings.patch_embeddings
    torch.testing.assert_close(adapted.weight[:, :3], original)
    torch.testing.assert_close(adapted.weight[:, 3], original.mean(dim=1))
    torch.testing.assert_close(adapted.weight[:, 4], original.mean(dim=1))
    torch.testing.assert_close(adapted.bias, original_bias)


def test_the_adapted_model_consumes_a_five_channel_image_and_predicts_six_numbers() -> None:
    """End to end: the tensor the input builder produces goes in, a 6DoF vector comes out."""

    from bevcalib.correctors.learned import (
        build_five_channel_input,
        initialize_convnextv2_five_channel,
    )

    model = initialize_convnextv2_five_channel(convnext_like())
    rgb = np.zeros((3, 32, 32), dtype=np.float32)
    depth = np.full((32, 32), 12.0, dtype=np.float32)
    valid = np.ones((32, 32), dtype=bool)

    tensor = torch.from_numpy(build_five_channel_input(rgb, depth, valid)).unsqueeze(0)
    prediction = model(tensor)

    assert tuple(prediction.shape) == (1, 6)
    assert bool(torch.isfinite(prediction).all())


def test_a_model_with_no_three_channel_stem_is_refused() -> None:
    """Adapting an already-adapted model would widen it to seven channels."""

    from bevcalib.correctors.learned import initialize_convnextv2_five_channel

    model = initialize_convnextv2_five_channel(convnext_like())

    with pytest.raises(
        ValueError,
        match=r"^the model has no three-channel convolution to adapt as a stem$",
    ):
        initialize_convnextv2_five_channel(model)


def test_a_stem_without_a_bias_is_adapted_without_inventing_one() -> None:
    """Some ConvNeXt variants drop the stem bias, and adding one would change the model."""

    from torch import nn

    from bevcalib.correctors.learned import initialize_convnextv2_five_channel

    class Bare(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.stem = nn.Conv2d(3, 8, kernel_size=4, stride=4, bias=False)

    model = initialize_convnextv2_five_channel(Bare())

    assert model.stem.in_channels == 5
    assert model.stem.bias is None


def test_a_validity_mask_of_zeros_and_ones_marks_the_same_pixels_a_bool_mask_marks() -> None:
    """The mask arrives as 0/1 from the rasteriser, and must select the same pixels.

    The mask becomes a two-dimensional index into the depth channel, and an
    integer index selects ROWS by position rather than pixels by truth. The two
    valid pixels are deliberately in DIFFERENT rows: with one pixel the two
    readings coincide by accident, and the test would pass while asserting
    nothing. With two, the positional reading writes rows 0 and 1 and never
    reaches row 2, so the second depth is dropped entirely and the model trains
    on a depth channel missing the returns it was given.
    """

    from bevcalib.correctors.learned import build_five_channel_input

    rgb, depth, valid = scene()
    depth[2, 4] = 20.0
    valid[2, 4] = True

    as_integers = build_five_channel_input(rgb, depth, valid.astype(np.int64))
    as_booleans = build_five_channel_input(rgb, depth, valid)

    np.testing.assert_array_equal(as_integers, as_booleans)
    assert np.argwhere(as_booleans[3] != 0.0).tolist() == [[1, 2], [2, 4]]
