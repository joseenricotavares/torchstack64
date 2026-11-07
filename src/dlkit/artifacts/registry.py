from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from dlkit.artifacts.bundle import ArtifactBundle
from dlkit.inference.predictor import Predictor

PredictorFactory = Callable[[ArtifactBundle, str], Predictor[Any]]


@dataclass(frozen=True, slots=True)
class LoadedModel:
    predictor: Predictor[Any]
    bundle: ArtifactBundle


class UnknownFlavorError(LookupError):
    def __init__(self, flavor: str, known: list[str]) -> None:
        super().__init__(f"No predictor factory registered for flavor {flavor!r}; known: {known}")
        self.flavor = flavor
        self.known = known


@runtime_checkable
class ModelSource(Protocol):
    """Strategy: where a bundle's bytes come from."""

    def resolve(self) -> Path:
        """Returns a local directory containing `bundle.json` plus the bundle's files."""
        ...


class LocalPathSource:
    """Reads a bundle already sitting on the local filesystem."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def resolve(self) -> Path:
        return self.path


class MLflowRunSource:
    """Downloads a bundle from an MLflow run's artifacts."""

    def __init__(
        self, run_id: str, artifact_path: str = "model", tracking_uri: str | None = None
    ) -> None:
        self.run_id = run_id
        self.artifact_path = artifact_path
        self.tracking_uri = tracking_uri

    def resolve(self) -> Path:
        import mlflow

        if self.tracking_uri is not None:
            mlflow.set_tracking_uri(self.tracking_uri)
        local_dir = mlflow.artifacts.download_artifacts(
            run_id=self.run_id, artifact_path=self.artifact_path
        )
        return Path(local_dir)


class ModelRegistry:
    """Factory: resolves a bundle's `flavor` to a concrete `Predictor`.

    The single plug-in point for adding a new model/paradigm: implement a
    `Trainer.save()` that writes an `ArtifactBundle` with a new `flavor`,
    implement a `Predictor` for it, and add one entry to `factories`.
    """

    def __init__(
        self, source: ModelSource, factories: dict[str, PredictorFactory], device: str = "cpu"
    ) -> None:
        self.source = source
        self.factories = factories
        self.device = device

    def load_with_bundle(self) -> LoadedModel:
        bundle = ArtifactBundle.read(self.source.resolve())
        factory = self.factories.get(bundle.manifest.flavor)
        if factory is None:
            raise UnknownFlavorError(bundle.manifest.flavor, known=sorted(self.factories))
        return LoadedModel(predictor=factory(bundle, self.device), bundle=bundle)

    def load(self) -> Predictor[Any]:
        return self.load_with_bundle().predictor
