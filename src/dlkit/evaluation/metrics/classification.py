from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

Average = Literal["macro", "micro", "weighted", "binary"]


@dataclass(slots=True)
class AccuracyMetric:
    """Plain accuracy - works for binary and multiclass alike."""

    name: str = "accuracy"

    def compute(self, y_true: Sequence, y_pred: Sequence) -> float:
        return float(accuracy_score(y_true, y_pred))


@dataclass(slots=True)
class F1Metric:
    """F1 score with a configurable averaging strategy."""

    average: Average = "macro"
    name: str = ""

    def __post_init__(self) -> None:
        if not self.name:
            self.name = f"f1_{self.average}"

    def compute(self, y_true: Sequence, y_pred: Sequence) -> float:
        return float(f1_score(y_true, y_pred, average=self.average, zero_division=0))


@dataclass(slots=True)
class PrecisionMetric:
    average: Average = "macro"
    name: str = ""

    def __post_init__(self) -> None:
        if not self.name:
            self.name = f"precision_{self.average}"

    def compute(self, y_true: Sequence, y_pred: Sequence) -> float:
        return float(precision_score(y_true, y_pred, average=self.average, zero_division=0))


@dataclass(slots=True)
class RecallMetric:
    average: Average = "macro"
    name: str = ""

    def __post_init__(self) -> None:
        if not self.name:
            self.name = f"recall_{self.average}"

    def compute(self, y_true: Sequence, y_pred: Sequence) -> float:
        return float(recall_score(y_true, y_pred, average=self.average, zero_division=0))
