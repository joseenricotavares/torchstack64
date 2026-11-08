from importlib import metadata as _metadata
from importlib.metadata import version as _pkg_version

from dlkit.artifacts import (
    ArtifactBundle,
    BundleFile,
    BundleManifest,
    LabelSchema,
    LoadedModel,
    LocalPathSource,
    MLflowRunSource,
    ModelRegistry,
    ModelSource,
    UnknownFlavorError,
)
from dlkit.evaluation import AccuracyMetric, EvaluationResult, Evaluator, F1Metric, Metric
from dlkit.inference import Predictor, ProbabilisticPredictor
from dlkit.training import (
    OPTIMIZERS,
    BestCheckpoint,
    EpochResult,
    Monitor,
    NNTrainer,
    OptimizerType,
    SimpleTrainerConfig,
    TrainerCallback,
    TrainerConfig,
    TrainingHistory,
)

PackageNotFoundError = _metadata.PackageNotFoundError

try:
    __version__ = _pkg_version("dlkit")
except PackageNotFoundError:
    __version__ = "0.0.0"

__all__ = [
    "__version__",
    # artifacts
    "ArtifactBundle",
    "BundleFile",
    "BundleManifest",
    "LabelSchema",
    "ModelSource",
    "LocalPathSource",
    "MLflowRunSource",
    "ModelRegistry",
    "LoadedModel",
    "UnknownFlavorError",
    # inference
    "Predictor",
    "ProbabilisticPredictor",
    # evaluation
    "Metric",
    "Evaluator",
    "EvaluationResult",
    "AccuracyMetric",
    "F1Metric",
    # training
    "NNTrainer",
    "TrainerCallback",
    "TrainerConfig",
    "SimpleTrainerConfig",
    "Monitor",
    "TrainingHistory",
    "EpochResult",
    "BestCheckpoint",
    # optim
    "OptimizerType",
    "OPTIMIZERS",
]
