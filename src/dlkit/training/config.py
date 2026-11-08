from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from dlkit.training.monitor import Monitor
from dlkit.training.optim import OptimizerType


@runtime_checkable
class TrainerConfig(Protocol):
    """The minimal shape `NNTrainer.fit()` needs from a config object."""

    epochs: int
    early_stopping_patience: int
    monitor: Monitor


@dataclass
class SimpleTrainerConfig:
    """Concrete `TrainerConfig` ready to use directly, or to copy as a starting point."""

    epochs: int = 30
    early_stopping_patience: int = 5
    monitor: Monitor = field(default_factory=lambda: Monitor(metric=None, mode="min"))
    optimizer: OptimizerType = OptimizerType.ADAMW
    learning_rate: float = 1e-3
    weight_decay: float = 1e-2
    grad_clip_norm: float | None = None
