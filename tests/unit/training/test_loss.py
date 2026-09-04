"""Contracts for the loss that trains the corrector.

Rotation is in degrees and translation is in metres, so an unnormalised loss would
be dominated by whichever unit happens to have the larger numbers. Normalising by
the widest fault the study injects, two degrees and twenty centimetres, makes an
error at the edge of the range count the same on either.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch", reason="the learned corrector needs the train extra")


def tensors(difference: list[float]):  # type: ignore[no-untyped-def]
    """A prediction that misses a zero target by exactly `difference`."""

    target = torch.zeros((1, 6), dtype=torch.float64)
    prediction = torch.tensor([difference], dtype=torch.float64)
    return prediction, target


def test_a_perfect_prediction_costs_nothing() -> None:
    """The floor has to be reachable, or every reported loss carries an offset."""

    from bevcalib.training.loss import normalized_huber_loss

    prediction, target = tensors([0.0] * 6)

    assert float(normalized_huber_loss(prediction, target)) == pytest.approx(0.0)


def test_an_error_at_the_edge_of_the_range_sits_exactly_on_the_huber_kink() -> None:
    """Two degrees normalises to one, which is delta, so the cost is half of one squared."""

    from bevcalib.training.loss import normalized_huber_loss

    prediction, target = tensors([2.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    # 0.5 * 1**2 on one component out of six.
    assert float(normalized_huber_loss(prediction, target)) == pytest.approx(0.5 / 6.0)


def test_an_error_past_the_kink_grows_linearly_rather_than_quadratically() -> None:
    """That is the point of Huber: one wild sample must not dominate a batch."""

    from bevcalib.training.loss import normalized_huber_loss

    prediction, target = tensors([4.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    # |e| = 2, so delta * (|e| - 0.5 * delta) = 1.5, over six components.
    assert float(normalized_huber_loss(prediction, target)) == pytest.approx(1.5 / 6.0)


def test_two_degrees_and_twenty_centimetres_cost_exactly_the_same() -> None:
    """This is what the normalisation is for, stated as a single equality."""

    from bevcalib.training.loss import normalized_huber_loss

    rotation_error, target = tensors([2.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    translation_error, _ = tensors([0.0, 0.0, 0.0, 0.2, 0.0, 0.0])

    assert float(normalized_huber_loss(rotation_error, target)) == pytest.approx(
        float(normalized_huber_loss(translation_error, target))
    )


def test_without_normalisation_the_two_would_not_be_comparable() -> None:
    """Proof the previous test is not vacuous: the raw numbers differ by a hundredfold."""

    from bevcalib.training.loss import normalized_huber_loss

    rotation_error, target = tensors([2.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    translation_error, _ = tensors([0.0, 0.0, 0.0, 0.2, 0.0, 0.0])

    unnormalised_rotation = normalized_huber_loss(
        rotation_error, target, rotation_scale_deg=1.0, translation_scale_m=1.0
    )
    unnormalised_translation = normalized_huber_loss(
        translation_error, target, rotation_scale_deg=1.0, translation_scale_m=1.0
    )

    assert float(unnormalised_rotation) > 10.0 * float(unnormalised_translation)


def test_the_loss_averages_over_the_batch_and_over_the_six_degrees() -> None:
    """A sum would make the learning rate depend on the batch size."""

    from bevcalib.training.loss import normalized_huber_loss

    target = torch.zeros((3, 6), dtype=torch.float64)
    prediction = torch.zeros((3, 6), dtype=torch.float64)
    prediction[0, 0] = 2.0

    # One component of eighteen carries 0.5.
    assert float(normalized_huber_loss(prediction, target)) == pytest.approx(0.5 / 18.0)


def test_the_gradient_is_finite_everywhere_including_at_the_kink() -> None:
    """The kink is where a naive implementation produces a NaN and kills a run silently."""

    from bevcalib.training.loss import normalized_huber_loss

    for difference in ([0.0] * 6, [2.0] + [0.0] * 5, [4.0] + [0.0] * 5, [-2.0] + [0.0] * 5):
        prediction = torch.tensor([difference], dtype=torch.float64, requires_grad=True)
        target = torch.zeros((1, 6), dtype=torch.float64)

        normalized_huber_loss(prediction, target).backward()

        assert prediction.grad is not None
        assert bool(torch.isfinite(prediction.grad).all()), difference


def test_the_loss_is_symmetric_in_the_sign_of_the_error() -> None:
    """Over-prediction and under-prediction by the same amount are equally wrong."""

    from bevcalib.training.loss import normalized_huber_loss

    positive, target = tensors([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    negative, _ = tensors([-1.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    assert float(normalized_huber_loss(positive, target)) == pytest.approx(
        float(normalized_huber_loss(negative, target))
    )


@pytest.mark.parametrize("shape", [(1, 5), (1, 7), (6,), (2, 3, 6)])
def test_predictions_that_are_not_six_degrees_of_freedom_are_rejected(
    shape: tuple[int, ...],
) -> None:
    """Six, in the order roll, pitch, yaw, x, y, z, or the scales apply to the wrong axes."""

    from bevcalib.training.loss import normalized_huber_loss

    with pytest.raises(
        ValueError,
        match=r"^predictions must have shape \[N, 6\] in the order roll, pitch, yaw, x, y, z, got ",
    ):
        normalized_huber_loss(
            torch.zeros(shape, dtype=torch.float64), torch.zeros(shape, dtype=torch.float64)
        )


def test_a_prediction_and_a_target_of_different_sizes_are_rejected() -> None:
    """Broadcasting one against the other would silently score the wrong pairs."""

    from bevcalib.training.loss import normalized_huber_loss

    with pytest.raises(ValueError, match=r"^prediction .* and target .* must have the same shape$"):
        normalized_huber_loss(
            torch.zeros((2, 6), dtype=torch.float64), torch.zeros((3, 6), dtype=torch.float64)
        )


@pytest.mark.parametrize(
    ("rotation_scale", "translation_scale", "delta"),
    [(0.0, 0.2, 1.0), (2.0, 0.0, 1.0), (2.0, 0.2, 0.0), (-2.0, 0.2, 1.0)],
)
def test_scales_and_delta_must_be_positive(
    rotation_scale: float, translation_scale: float, delta: float
) -> None:
    """A zero scale divides by zero; a zero delta makes the loss identically linear."""

    from bevcalib.training.loss import normalized_huber_loss

    prediction, target = tensors([0.0] * 6)

    with pytest.raises(ValueError, match=r"^the scales and delta must be positive, got "):
        normalized_huber_loss(
            prediction,
            target,
            rotation_scale_deg=rotation_scale,
            translation_scale_m=translation_scale,
            delta=delta,
        )
