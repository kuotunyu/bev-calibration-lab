"""Contracts for the paired-scene bootstrap that puts an interval on an improvement.

The scene is the resampling unit, not the sample. Two keyframes from one scene are
the same road a second apart, so resampling samples would treat correlated
observations as independent and return an interval far narrower than the evidence
supports.
"""

from __future__ import annotations

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

    with pytest.raises(ValueError, match="no scenes"):
        paired_scene_bootstrap({})


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_a_scene_with_an_unmeasurable_value_fails_closed(value: float) -> None:
    """A NaN would spread to the estimate and to every resample that happened to draw it."""

    from bevcalib.metrics.bootstrap import paired_scene_bootstrap

    with pytest.raises(ValueError, match="finite"):
        paired_scene_bootstrap(paired(a=(1.0, 0.0), b=(value, 0.0)), resamples=10)


@pytest.mark.parametrize(("resamples", "confidence"), [(0, 0.95), (-1, 0.95), (10, 0.0), (10, 1.0)])
def test_a_bootstrap_that_cannot_produce_an_interval_is_refused(
    resamples: int, confidence: float
) -> None:
    """Zero resamples has no distribution, and a confidence of one has no quantiles."""

    from bevcalib.metrics.bootstrap import paired_scene_bootstrap

    with pytest.raises(ValueError):
        paired_scene_bootstrap(paired(a=(1.0, 0.0)), resamples=resamples, confidence=confidence)


def test_an_interval_cannot_be_edited_after_the_fact() -> None:
    """It is the uncertainty attached to a published number."""

    import dataclasses

    from bevcalib.metrics.bootstrap import paired_scene_bootstrap

    with pytest.raises(dataclasses.FrozenInstanceError):
        paired_scene_bootstrap(paired(a=(1.0, 0.0)), resamples=10).low = 0.0  # type: ignore[misc]
