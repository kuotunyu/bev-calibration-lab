"""Controlled metadata calibration faults: the schedule, the algebra, the timing."""

from bevcalib.artifacts.results import CalibrationFaultModel

from .apply import apply_metadata_fault, fault_to_se3
from .schedule import (
    FaultAxis,
    PerturbationMatrixV1,
    formal_single_axis_faults,
    load_perturbation_matrix,
    sample_training_fault,
)
from .timing import learned_six_dof_target, select_camera_for_timing_fault

# The plan names this type `CalibrationFault`. It is the same concept as the
# validated artifact model, so it is one definition under two names rather than
# two definitions that would eventually disagree.
CalibrationFault = CalibrationFaultModel

__all__ = [
    "CalibrationFault",
    "CalibrationFaultModel",
    "FaultAxis",
    "PerturbationMatrixV1",
    "apply_metadata_fault",
    "fault_to_se3",
    "formal_single_axis_faults",
    "learned_six_dof_target",
    "load_perturbation_matrix",
    "sample_training_fault",
    "select_camera_for_timing_fault",
]
