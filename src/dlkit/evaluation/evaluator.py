from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from dlkit.evaluation.metrics.base import Metric


@dataclass(slots=True)
class EvaluationResult:
    """A flat, named-metric-value bag produced by an `Evaluator` run."""

    values: dict[str, float] = field(default_factory=dict)

    def __getitem__(self, key: str) -> float:
        return self.values[key]

    def __contains__(self, key: str) -> bool:
        return key in self.values

    def get(self, key: str, default: float | None = None) -> float | None:
        return self.values.get(key, default)


class Evaluator:
    """Runs a fixed list of `Metric`s over the same (y_true, y_pred) pair."""

    def __init__(self, metrics: Sequence[Metric]) -> None:
        self.metrics = list(metrics)

    def evaluate(self, y_true: Sequence, y_pred: Sequence) -> EvaluationResult:
        return EvaluationResult(
            values={metric.name: metric.compute(y_true, y_pred) for metric in self.metrics}
        )
