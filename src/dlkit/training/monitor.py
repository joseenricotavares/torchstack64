from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from dlkit.evaluation.metrics.base import Metric


@dataclass(frozen=True, slots=True)
class Monitor:
    """What `NNTrainer.fit()` watches to pick the best checkpoint and decide when to early-stop. 
    `metric=None` means "monitor validation loss itself" 
    """

    metric: Metric | None
    mode: Literal["min", "max"]

    @property
    def name(self) -> str:
        return "val_loss" if self.metric is None else f"val_{self.metric.name}"

    def score(self, val_loss: float, y_true: Sequence, y_pred: Sequence) -> float:
        if self.metric is None:
            return val_loss
        return self.metric.compute(y_true, y_pred)

    def improved(self, current: float, best: float) -> bool:
        return current > best if self.mode == "max" else current < best
