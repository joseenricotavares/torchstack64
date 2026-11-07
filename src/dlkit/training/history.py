from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class TrainingHistory:
    """Per-epoch metric snapshots accumulated over a `fit()` run."""

    entries: list[dict[str, float]] = field(default_factory=list)


@dataclass(slots=True)
class BestCheckpoint:
    """The best model state seen so far, plus the validation predictions that earned it."""

    epoch: int
    score: float
    state: dict[str, Any]
    targets: list
    preds: list


@dataclass(slots=True)
class EpochResult:
    """Passed to callbacks after each epoch."""

    epoch: int
    metrics: dict[str, float]
    improved: bool
    patience: int
    best: BestCheckpoint | None
