"""The corrector that does nothing, and the record every corrector returns.

`CorrectionEstimate` lives here rather than beside the optimiser because this is
the module with no dependencies: naming a corrector's result should not require
importing a search algorithm.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class CorrectionEstimate:
    """What a corrector proposes, and what it cost to propose it."""

    rotation_rpy_deg: tuple[float, float, float]
    translation_xyz_m: tuple[float, float, float]
    objective_value: float
    evaluations: int
    converged: bool


def identity_correction() -> CorrectionEstimate:
    """Return the baseline correction: exactly nothing, and no claim about anything.

    Every recovery number in the study is measured against this, so the values are
    exact zeros rather than nearly zero; a baseline with a small bias would shift
    every comparison made against it.

    The objective value is `nan`, deliberately. The alignment score is a negated
    distance, so 0.0 is its ceiling, and a corrector that never measured anything
    reporting the best possible number would be the most misleading value
    available. Zero evaluations says why it has none.

    `converged` is true because there was no search to fail. A downstream filter
    on convergence must not silently drop the baseline that everything else is
    compared against.
    """

    return CorrectionEstimate(
        rotation_rpy_deg=(0.0, 0.0, 0.0),
        translation_xyz_m=(0.0, 0.0, 0.0),
        objective_value=math.nan,
        evaluations=0,
        converged=True,
    )
