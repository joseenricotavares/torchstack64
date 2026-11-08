from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from dlkit.artifacts.bundle import ArtifactBundle, LabelSchema


@runtime_checkable
class Predictor[InputT](Protocol):
    """The inference interface every paradigm implements."""

    label_schema: LabelSchema

    def predict(self, inputs: Sequence[InputT]) -> list[int]:
        """Raw input in, predicted label out - including any cleaning/tokenization."""
        ...

    @classmethod
    def load(cls, bundle: ArtifactBundle, device: str = "cpu") -> Predictor[InputT]:
        """Reconstructs a ready-to-use predictor from a bundle written by `NNTrainer.save()`."""
        ...


@runtime_checkable
class ProbabilisticPredictor[InputT](Predictor[InputT], Protocol):
    """Optional extension for predictors that can also expose class probabilities."""

    def predict_proba(self, inputs: Sequence[InputT]) -> list[list[float]]: ...
