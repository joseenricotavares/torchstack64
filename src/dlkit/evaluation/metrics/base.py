from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable


@runtime_checkable
class Metric(Protocol):
    """A named, scalar evaluation metric."""

    name: str

    def compute(self, y_true: Sequence, y_pred: Sequence) -> float:
        """Returns a single scalar score for `y_true` vs. `y_pred`."""
        ...
