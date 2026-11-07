from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from dlkit.artifacts.bundle import ArtifactBundle
from dlkit.training.config import TrainerConfig
from dlkit.training.history import BestCheckpoint, EpochResult, TrainingHistory


class TrainerCallback:
    """No-op base: override only the hooks a concrete callback needs."""

    def on_train_begin(self, trainer: NNTrainer) -> None:
        """Called before the first training epoch."""

    def on_epoch_end(self, trainer: NNTrainer, result: EpochResult) -> None:
        """Called after each training epoch."""

    def on_train_end(self, trainer: NNTrainer) -> None:
        """Called after training finishes (including on early stop)."""


class NNTrainer[ConfigT: TrainerConfig](ABC):
    """Epoch-based train / validate / early-stop / checkpoint loop.

    Subclasses implement `_train_epoch`, `predict`, and `save`; `fit()` is
    the one thing every epoch-based PyTorch trainer needs and none of them
    should have to rewrite.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: ConfigT,
        device: torch.device,
        callbacks: Sequence[TrainerCallback] | None = None,
    ) -> None:
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        self.callbacks = list(callbacks or [])
        self.history: TrainingHistory = TrainingHistory()
        self.best: BestCheckpoint | None = None
        self.monitor = config.monitor

    @abstractmethod
    def _train_epoch(self) -> float:
        """Runs one training epoch, returning the average training loss."""

    @abstractmethod
    def predict(self, loader: DataLoader) -> tuple[float, list, list]:
        """Runs inference over `loader`, returning (avg_loss, targets, predictions)."""

    @abstractmethod
    def save(self, output_dir: str | Path) -> ArtifactBundle:
        """Persists the trained model as an `ArtifactBundle` other code can `load()`."""

    def fit(self) -> TrainingHistory:
        """Trains until `config.epochs` or early stopping, returning per-epoch history."""
        for callback in self.callbacks:
            callback.on_train_begin(self)

        epochs_without_improvement = 0
        try:
            for epoch in range(1, self.config.epochs + 1):
                train_loss = self._train_epoch()
                val_loss, val_targets, val_preds = self.predict(self.val_loader)
                score = self.monitor.score(val_loss, val_targets, val_preds)

                epoch_metrics = {"train_loss": train_loss, "val_loss": val_loss}
                epoch_metrics[self.monitor.name] = score
                self.history.entries.append(epoch_metrics)

                improved = self.best is None or self.monitor.improved(score, self.best.score)
                if improved:
                    self.best = BestCheckpoint(
                        epoch=epoch,
                        score=score,
                        state=copy.deepcopy(self.model.state_dict()),
                        targets=val_targets,
                        preds=val_preds,
                    )
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1

                result = EpochResult(
                    epoch=epoch,
                    metrics=epoch_metrics,
                    improved=improved,
                    patience=epochs_without_improvement,
                    best=self.best,
                )
                for callback in self.callbacks:
                    callback.on_epoch_end(self, result)

                if epochs_without_improvement >= self.config.early_stopping_patience:
                    break

            if self.best is not None:
                self.model.load_state_dict(self.best.state)
        finally:
            for callback in self.callbacks:
                callback.on_train_end(self)

        return self.history
