"""Quantifying calibration recovery, reprojection error and their uncertainty."""

from .bootstrap import BootstrapInterval, paired_scene_bootstrap
from .calibration import (
    RECOVERY_ROTATION_THRESHOLD_DEG,
    RECOVERY_TRANSLATION_THRESHOLD_M,
    calibration_errors,
    recovered,
    rotation_geodesic_error_deg,
)
from .reprojection import (
    RANGE_BINS,
    RangeBin,
    ValiditySummary,
    pixel_error_percentiles,
    range_bin,
    summarize_validity,
)

__all__ = [
    "RANGE_BINS",
    "RECOVERY_ROTATION_THRESHOLD_DEG",
    "RECOVERY_TRANSLATION_THRESHOLD_M",
    "BootstrapInterval",
    "RangeBin",
    "ValiditySummary",
    "calibration_errors",
    "paired_scene_bootstrap",
    "pixel_error_percentiles",
    "range_bin",
    "recovered",
    "rotation_geodesic_error_deg",
    "summarize_validity",
]
