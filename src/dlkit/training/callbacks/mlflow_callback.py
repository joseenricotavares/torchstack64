from __future__ import annotations

import shutil
import time
from pathlib import Path

from dlkit.training.base import NNTrainer, TrainerCallback
from dlkit.training.history import EpochResult


class MLflowCallback(TrainerCallback):
    """Logs per-epoch metrics, training time, and the final model bundle to the active MLflow run.
    Requires `dlkit[callbacks-mlflow]`.
    """

    def __init__(self, artifact_dir: str = "model") -> None:
        self.artifact_dir = Path(artifact_dir)
        self.start_time: float | None = None

    def on_train_begin(self, trainer: NNTrainer) -> None:
        self.start_time = time.perf_counter()

    def on_epoch_end(self, trainer: NNTrainer, result: EpochResult) -> None:
        import mlflow

        mlflow.log_metrics(result.metrics, step=result.epoch)

    def on_train_end(self, trainer: NNTrainer) -> None:
        import mlflow

        if self.start_time is not None:
            mlflow.log_metric("train_time_seconds", time.perf_counter() - self.start_time)
        if trainer.best is None:
            return

        trainer.save(self.artifact_dir)
        try:
            mlflow.log_artifacts(local_dir=str(self.artifact_dir), artifact_path="model")
        finally:
            shutil.rmtree(self.artifact_dir, ignore_errors=True)
