"""Pre-outcome analysis policy: equal frames within scenes, equal scenes within study."""

from __future__ import annotations

from bevcalib.metrics.reprojection import RANGE_BINS

POLICY_ID = "bev-calibration-analysis/v2:frame-then-scene:fixed-three-seeds:bounded-gt-range"
# The geometry contract's absolute precision, not fitted to evaluation outcomes.
# Relative tolerance is zero; bin and 80 m validity decisions must agree exactly.
GT_RANGE_ABSOLUTE_TOLERANCE_M = 1e-9
METRIC_UNITS = {
    "rotation_geodesic_deg": "degree",
    **{
        f"rotation_{kind}_{axis}_deg": "degree"
        for kind in ("bias", "abs")
        for axis in ("roll", "pitch", "yaw")
    },
    "translation_norm_cm": "centimetre",
    **{
        f"translation_{kind}_{axis}_cm": "centimetre"
        for kind in ("bias", "abs")
        for axis in ("x", "y", "z")
    },
    "pixel_frame_p50_px": "pixel",
    "pixel_frame_p90_px": "pixel",
    "edge_score_px": "pixel",
    **{f"bev_frame_mean_m/{name}": "metre" for name in RANGE_BINS},
    "recovery_rate_pct": "percent",
}
HIGHER_BETTER = {"edge_score_px", "recovery_rate_pct"}
COMPARISON_METRICS = tuple(name for name in METRIC_UNITS if "_bias_" not in name)
LEARNED_RUNS = ("learned-17", "learned-42", "learned-73")
COMPARISONS = {
    "identity->classical": ("identity", ("classical",)),
    **{
        f"{before}->{after}": (before, (after,))
        for before in ("identity", "classical")
        for after in LEARNED_RUNS
    },
    **{
        f"{before}->learned-fixed-three-seed-mean": (before, LEARNED_RUNS)
        for before in ("identity", "classical")
    },
}
ESTIMAND_DESCRIPTION = {
    "policy_id": POLICY_ID,
    "sampling_unit": "scene; equal scene weights; equal matched-frame weights within scene",
    "pixel": "mean of per-frame P50/P90, then mean across scenes; each frame quantile uses its own within-row valid point errors; no cross-method point pairing",
    "bev": "exact sample and box identity; GT range span <= 1e-9 m, zero relative tolerance, identical range bin and inclusive 80 m cutoff classification; common valid objects averaged within frame, frames within scene; all five fixed GT range bins",
    "pose": "signed bias descriptors retained; absolute per-axis errors used for improvement; translation converted from metres to centimetres",
    "recovery": "joint geodesic <=0.25 degrees and translation norm <=5 cm; rate percent, improvement percentage points",
    "directions": "error before-minus-after; edge/recovery after-minus-before",
    "seeds": "report 17/42/73 separately; fixed-three-seed mean uses common eligible support across baseline and all three seeds; interval conditional on those fixed seeds, not training randomness",
    "eligibility": "operator validity, independent of global row validity; absent support is null with reason; missing expected source inventory is fatal",
    "timing": "identity stress only; no timing correction rows or correction denominators",
    "bootstrap": "5000 paired scene resamples; SHA256 counter indices; seed 20260831; confidence 0.95",
}
