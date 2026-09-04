"""Contracts for the paired-scene bootstrap that puts an interval on an improvement.

The scene is the resampling unit, not the sample. Two keyframes from one scene are
the same road a second apart, so resampling samples would treat correlated
observations as independent and return an interval far narrower than the evidence
supports.
"""

from __future__ import annotations

import numpy as np
import pytest

DEFAULT_SEED = 20260831


def paired(**scenes: tuple[float, float]) -> dict[str, tuple[float, float]]:
    return dict(scenes)


def test_the_estimate_is_the_mean_improvement_per_scene() -> None:
    """Hand-computable: improvements of 1.0, 2.0 and 3.0 average to 2.0."""

    from bevcalib.metrics.bootstrap import paired_scene_bootstrap

    interval = paired_scene_bootstrap(
        paired(a=(5.0, 4.0), b=(5.0, 3.0), c=(5.0, 2.0)), resamples=200
    )

    assert interval.estimate == pytest.approx(2.0)


def test_the_interval_brackets_its_own_estimate() -> None:
    """An interval that excludes the point estimate would be evidence of a wiring bug."""

    from bevcalib.metrics.bootstrap import paired_scene_bootstrap

    interval = paired_scene_bootstrap(
        {f"scene-{index}": (10.0, float(index)) for index in range(30)}, resamples=500
    )

    assert interval.low <= interval.estimate <= interval.high


def test_the_same_seed_gives_exactly_the_same_interval() -> None:
    """A published interval that moves between runs is not a published interval."""

    from bevcalib.metrics.bootstrap import paired_scene_bootstrap

    data = {f"scene-{index}": (10.0, float(index)) for index in range(20)}

    assert paired_scene_bootstrap(data, resamples=300) == paired_scene_bootstrap(
        data, resamples=300
    )


def test_a_different_seed_gives_a_different_interval() -> None:
    """Proof the previous test is about determinism and not about a constant."""

    from bevcalib.metrics.bootstrap import paired_scene_bootstrap

    data = {f"scene-{index}": (10.0, float(index)) for index in range(20)}

    first = paired_scene_bootstrap(data, resamples=300, seed=1)
    second = paired_scene_bootstrap(data, resamples=300, seed=2)

    assert (first.low, first.high) != (second.low, second.high)
    assert first.estimate == pytest.approx(second.estimate)


def test_the_order_the_scenes_are_given_in_does_not_change_anything() -> None:
    """Dictionary order is an accident of construction and must not reach a result."""

    from bevcalib.metrics.bootstrap import paired_scene_bootstrap

    forward = {f"scene-{index}": (10.0, float(index)) for index in range(10)}
    backward = dict(reversed(list(forward.items())))

    assert paired_scene_bootstrap(forward, resamples=300) == paired_scene_bootstrap(
        backward, resamples=300
    )


def test_scenes_that_all_agree_give_an_interval_of_zero_width() -> None:
    """With no variation between scenes there is nothing for a resample to vary."""

    from bevcalib.metrics.bootstrap import paired_scene_bootstrap

    interval = paired_scene_bootstrap(
        {f"scene-{index}": (5.0, 3.0) for index in range(10)}, resamples=200
    )

    assert interval.low == pytest.approx(2.0)
    assert interval.high == pytest.approx(2.0)


def test_a_wider_confidence_gives_a_wider_interval() -> None:
    """The quantiles have to be wired to the confidence, not to a constant."""

    from bevcalib.metrics.bootstrap import paired_scene_bootstrap

    data = {f"scene-{index}": (10.0, float(index)) for index in range(30)}

    narrow = paired_scene_bootstrap(data, resamples=500, confidence=0.5)
    wide = paired_scene_bootstrap(data, resamples=500, confidence=0.99)

    assert (wide.high - wide.low) > (narrow.high - narrow.low)


def test_the_interval_records_what_produced_it() -> None:
    """An interval without its seed and resample count cannot be reproduced by anyone."""

    from bevcalib.metrics.bootstrap import paired_scene_bootstrap

    interval = paired_scene_bootstrap(paired(a=(1.0, 0.0), b=(2.0, 0.0)), resamples=123, seed=7)

    assert interval.resamples == 123
    assert interval.seed == 7
    assert interval.confidence == pytest.approx(0.95)


def test_the_protocol_defaults_are_the_ones_p1_used() -> None:
    """Five thousand resamples at seed 20260831, so the two projects are comparable."""

    from bevcalib.metrics.bootstrap import DEFAULT_CONFIDENCE, DEFAULT_RESAMPLES, DEFAULT_SEED

    assert (DEFAULT_RESAMPLES, DEFAULT_SEED, DEFAULT_CONFIDENCE) == (5000, 20260831, 0.95)


def test_the_full_default_bootstrap_runs_and_stays_deterministic() -> None:
    """The defaults are what the report will use, so they get exercised, not assumed."""

    from bevcalib.metrics.bootstrap import paired_scene_bootstrap

    data = {f"scene-{index}": (10.0, float(index % 7)) for index in range(30)}

    interval = paired_scene_bootstrap(data)

    assert interval.resamples == 5000
    assert interval.seed == DEFAULT_SEED
    assert interval == paired_scene_bootstrap(data)


def test_one_scene_gives_an_interval_with_no_width_and_says_so() -> None:
    """Resampling one scene can only ever draw that scene; the interval is honest, not narrow."""

    from bevcalib.metrics.bootstrap import paired_scene_bootstrap

    interval = paired_scene_bootstrap(paired(only=(5.0, 1.0)), resamples=100)

    assert interval.estimate == pytest.approx(4.0)
    assert interval.low == interval.high == pytest.approx(4.0)


def test_asking_for_an_interval_over_no_scenes_fails_closed() -> None:
    """There is no mean of an empty set, and reporting zero would be a claim."""

    from bevcalib.metrics.bootstrap import paired_scene_bootstrap

    with pytest.raises(ValueError, match=r"^cannot bootstrap over no scenes$"):
        paired_scene_bootstrap({})


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_a_scene_with_an_unmeasurable_value_fails_closed(value: float) -> None:
    """A NaN would spread to the estimate and to every resample that happened to draw it."""

    from bevcalib.metrics.bootstrap import paired_scene_bootstrap

    with pytest.raises(
        ValueError, match=r"^every scene must contribute a finite before and after value$"
    ):
        paired_scene_bootstrap(paired(a=(1.0, 0.0), b=(value, 0.0)), resamples=10)


@pytest.mark.parametrize(
    ("resamples", "confidence", "expected"),
    [
        (0, 0.95, r"^a bootstrap needs at least one resample, got "),
        (-1, 0.95, r"^a bootstrap needs at least one resample, got "),
        (10, 0.0, r"^the confidence must lie within \(0, 1\), got "),
        (10, 1.0, r"^the confidence must lie within \(0, 1\), got "),
    ],
)
def test_a_bootstrap_that_cannot_produce_an_interval_is_refused(
    resamples: int, confidence: float, expected: str
) -> None:
    """Zero resamples has no distribution, and a confidence of one has no quantiles.

    Those are two separate refusals and each row names its own, because the
    two are reported to different callers: a resample count is a knob the
    operator set, and a confidence of exactly one is a claim the statistics
    cannot support at any resample count.
    """

    from bevcalib.metrics.bootstrap import paired_scene_bootstrap

    with pytest.raises(ValueError, match=expected):
        paired_scene_bootstrap(paired(a=(1.0, 0.0)), resamples=resamples, confidence=confidence)


def test_an_interval_cannot_be_edited_after_the_fact() -> None:
    """It is the uncertainty attached to a published number."""

    import dataclasses

    from bevcalib.metrics.bootstrap import paired_scene_bootstrap

    with pytest.raises(dataclasses.FrozenInstanceError):
        paired_scene_bootstrap(paired(a=(1.0, 0.0)), resamples=10).low = 0.0  # type: ignore[misc]


SCENES = {"a": (1.0, 0.0), "b": (2.0, 0.5), "c": (3.0, 1.0), "d": (4.0, 3.0)}


def test_the_resampled_index_stream_is_the_rule_the_interval_records() -> None:
    """The draw is a documented recipe, so the indices it produces are a contract.

    `BootstrapInterval` records only the seed and the resample count, so a
    reader reproduces the interval by re-deriving the draw: `sha256("seed|n")`
    read as big-endian uint32 words, concatenated until the block is full,
    reduced modulo the scene count. This pins what that recipe yields.

    Every part of the loop is load-bearing and none of it is visible in the
    interval itself. Starting the fill or the counter at one, stepping the
    counter by two or backwards, or assigning the fill count instead of adding
    to it, all leave a well-formed array of indices in range — a different
    array, drawn from a different stream, producing a different published
    interval that still looks entirely reasonable.
    """

    from bevcalib.metrics.bootstrap import _resample_indices

    indices = _resample_indices(7, 3, 4)

    assert indices.tolist() == [[2, 3, 0, 3], [2, 3, 1, 1], [3, 0, 1, 1]]
    assert indices.dtype == np.int64


def test_a_single_resample_is_a_usable_bootstrap() -> None:
    """One resample is degenerate but well formed, and the guard says at least one.

    Written `<= 1` or `< 2` the validator would refuse the smallest bootstrap
    anybody can ask for, and the refusal would read as a malformed request
    rather than as an off-by-one bound. The interval it returns collapses to a
    point, which is the caller's problem and not the validator's.
    """

    from bevcalib.metrics.bootstrap import paired_scene_bootstrap

    interval = paired_scene_bootstrap(SCENES, resamples=1, seed=7)

    assert interval.resamples == 1
    assert interval.low == interval.high
    assert interval.estimate == pytest.approx(1.375)


def test_the_interval_cuts_an_equal_tail_from_each_end() -> None:
    """A 50% interval is the quartiles, and the two tails must be the same size.

    The bound comes from `(1 - confidence) / 2` at each end. Multiplying instead
    of halving sends the lower cut past the upper one; dividing by three leaves
    a wider interval than the confidence claims. Neither shows in the recorded
    `confidence` field, which is copied through untouched, so the number a
    reader trusts would not match the interval they are given.

    The expectation is recomputed here from the same draws with numpy's own
    quantile, so it tests the tail arithmetic rather than restating it.
    """

    from bevcalib.metrics.bootstrap import _resample_indices, paired_scene_bootstrap

    interval = paired_scene_bootstrap(SCENES, resamples=200, seed=7, confidence=0.5)

    differences = np.array([before - after for before, after in SCENES.values()])
    means = differences[_resample_indices(7, 200, differences.size)].mean(axis=1)
    expected_low, expected_high = np.quantile(means, [0.25, 0.75])

    assert interval.low == pytest.approx(float(expected_low))
    assert interval.high == pytest.approx(float(expected_high))
    assert interval.low < interval.high


@pytest.mark.parametrize(
    ("resamples", "confidence", "message"),
    [
        (0, 0.95, r"^a bootstrap needs at least one resample, got 0$"),
        (-3, 0.95, r"^a bootstrap needs at least one resample, got -3$"),
        (10, 0.0, r"^the confidence must lie within \(0, 1\), got 0\.0$"),
        (10, 1.0, r"^the confidence must lie within \(0, 1\), got 1\.0$"),
        (10, 1.5, r"^the confidence must lie within \(0, 1\), got 1\.5$"),
    ],
)
def test_a_malformed_request_names_the_value_it_refused(
    resamples: int,
    confidence: float,
    message: str,
) -> None:
    """The offending number is the diagnosis, and both bounds are exclusive.

    A caller sweeping a confidence level or a resample count needs to see which
    value was rejected, not merely that something was. Both messages interpolate
    it, and asserting them in full is also what makes them detectable when the
    wording changes.
    """

    from bevcalib.metrics.bootstrap import paired_scene_bootstrap

    with pytest.raises(ValueError, match=message):
        paired_scene_bootstrap(SCENES, resamples=resamples, seed=7, confidence=confidence)
