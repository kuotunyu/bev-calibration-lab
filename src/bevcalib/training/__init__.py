"""Training the learned corrector: what it sees, how it is scored, how it is run."""

from bevcalib.cohort.manifest import CohortManifestV1, CohortManifestV2

from .dataset import TrainingExample, build_training_example, target_for_fault
from .engine import (
    CalibrationTrainingResult,
    CorrectorConfigV1,
    TrainingBackend,
    load_corrector_config,
    train_learned_corrector,
)
from .loss import normalized_huber_loss

__all__ = [
    "CalibrationTrainingResult",
    "CohortManifestV1",
    "CohortManifestV2",
    "CorrectorConfigV1",
    "TrainingBackend",
    "TrainingExample",
    "build_training_example",
    "load_corrector_config",
    "normalized_huber_loss",
    "target_for_fault",
    "train_learned_corrector",
]
