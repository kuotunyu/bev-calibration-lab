"""Numerical support and confidence limits must be honest finite measurements."""

import pytest

from bevcalib.artifacts.statistics import Estimate, PairedEstimate, Support

SUPPORT = {
    "total_frames": 2,
    "frames": 1,
    "excluded_frames": 1,
    "total_scenes": 2,
    "scenes": 1,
    "exclusions": {"identity:operator_unavailable": 1},
}


@pytest.mark.parametrize(
    "change", [{"scenes": 0}, {"total_objects": 0, "objects": 0, "excluded_objects": 0}]
)
def test_positive_frame_support_requires_scene_and_object_denominators(change) -> None:
    with pytest.raises(ValueError, match="support"):
        Support.model_validate(SUPPORT | change)


@pytest.mark.parametrize("field,value", [("low", float("nan")), ("high", float("inf"))])
def test_intervals_refuse_nonfinite_endpoints(field: str, value: float) -> None:
    interval = {
        "estimate": 1.0,
        "low": 0.0,
        "high": 2.0,
        "confidence": 0.95,
        "resamples": 5000,
        "seed": 20260831,
    }
    with pytest.raises(ValueError, match="interval"):
        PairedEstimate(
            before=2,
            after=1,
            improvement=1,
            interval=interval | {field: value},
            support=SUPPORT,
            reason=None,
        )


def test_nulls_and_object_denominators_cannot_contradict_measured_support() -> None:
    with pytest.raises(ValueError, match="object support"):
        Support.model_validate(SUPPORT | {"objects": 1})
    with pytest.raises(ValueError, match="null estimate"):
        Estimate(value=None, support=SUPPORT, reason="missing")
    with pytest.raises(ValueError, match="paired null"):
        PairedEstimate(
            before=1, after=None, improvement=None, interval=None, support=SUPPORT, reason="missing"
        )
