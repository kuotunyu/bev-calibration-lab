"""A paired bootstrap over scenes, for putting an interval on an improvement."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

# The same constants P1 used, so an interval from either project means the same
# thing and the two are comparable side by side.
DEFAULT_RESAMPLES = 5000
DEFAULT_SEED = 20260831
DEFAULT_CONFIDENCE = 0.95


@dataclass(frozen=True)
class BootstrapInterval:
    """An estimate with its uncertainty, and everything needed to reproduce it."""

    estimate: float
    low: float
    high: float
    confidence: float
    resamples: int
    seed: int


def _resample_indices(seed: int, resamples: int, size: int) -> npt.NDArray[np.int64]:
    """Deterministic resample indices, derived from SHA-256 in counter mode.

    Not a library generator. A published interval has to reproduce years from now,
    on a different machine and a different numpy, and a digest is identical
    everywhere forever while a generator stream is only guaranteed while the
    library keeps its promise. This is the same reasoning as the keyed training
    faults in `perturbations.schedule`.

    Reducing a 32-bit value modulo the scene count introduces a bias below one
    part in ten million for any cohort this study will ever have, which is
    immaterial against 5000 resamples.
    """

    needed = resamples * size
    values = np.empty(needed, dtype=np.uint32)
    filled, counter = 0, 0
    while filled < needed:
        digest = hashlib.sha256(f"{seed}|{counter}".encode()).digest()
        chunk = np.frombuffer(digest, dtype=">u4")
        take = min(int(chunk.size), needed - filled)
        values[filled : filled + take] = chunk[:take]
        filled += take
        counter += 1
    return (values % size).astype(np.int64).reshape(resamples, size)


def paired_scene_bootstrap(
    before_after_by_scene: Mapping[str, tuple[float, float]],
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
    *,
    confidence: float = DEFAULT_CONFIDENCE,
) -> BootstrapInterval:
    """Interval for the mean per-scene improvement, resampling SCENES with replacement.

    The scene is the resampling unit, not the sample. Two keyframes from one scene
    are the same road a second apart, so resampling samples would treat correlated
    observations as independent and return an interval far narrower than the
    evidence supports. That is the failure mode this function exists to avoid.

    Paired, because each scene contributes a before and an after measured on the
    same data; the difference removes the scene's own difficulty from the estimate.
    """

    if resamples < 1:
        raise ValueError(f"a bootstrap needs at least one resample, got {resamples}")
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"the confidence must lie within (0, 1), got {confidence}")

    # Sorted, so the answer cannot depend on the order a caller happened to build
    # the mapping in.
    scenes = sorted(before_after_by_scene)
    if not scenes:
        raise ValueError("cannot bootstrap over no scenes")

    differences = np.array(
        [before_after_by_scene[scene][0] - before_after_by_scene[scene][1] for scene in scenes],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(differences)):
        raise ValueError("every scene must contribute a finite before and after value")

    means = differences[_resample_indices(seed, resamples, differences.size)].mean(axis=1)
    tail = (1.0 - confidence) / 2.0
    low, high = np.quantile(means, [tail, 1.0 - tail])
    return BootstrapInterval(
        estimate=float(differences.mean()),
        low=float(low),
        high=float(high),
        confidence=confidence,
        resamples=resamples,
        seed=seed,
    )
