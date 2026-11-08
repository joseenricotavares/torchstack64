from pathlib import Path

import pytest

from dlkit.artifacts.bundle import ArtifactBundle, BundleFile, LabelSchema
from dlkit.artifacts.registry import LocalPathSource, MLflowRunSource, ModelRegistry, UnknownFlavorError

SCHEMA = LabelSchema(kind="binary", classes=(0, 1))


class StubPredictor:
    def __init__(self, bundle: ArtifactBundle, device: str) -> None:
        self.label_schema = bundle.manifest.label_schema
        self.device = device

    def predict(self, inputs):
        return [0 for _ in inputs]


def _write_bundle(root: Path, flavor: str = "stub") -> Path:
    ArtifactBundle.write(
        root,
        flavor=flavor,
        label_schema=SCHEMA,
        files={"weights": BundleFile("weights.bin", lambda p: p.write_bytes(b""))},
    )
    return root


def test_local_path_source_resolves_to_the_path_it_was_given(tmp_path):
    assert LocalPathSource(str(tmp_path)).resolve() == tmp_path
    assert LocalPathSource(tmp_path).resolve() == tmp_path


def test_registry_builds_the_predictor_registered_for_the_bundles_flavor(tmp_path):
    root = _write_bundle(tmp_path / "bundle", flavor="stub")
    registry = ModelRegistry(LocalPathSource(root), {"stub": StubPredictor}, device="cuda:0")

    loaded = registry.load_with_bundle()

    assert isinstance(loaded.predictor, StubPredictor)
    assert loaded.predictor.device == "cuda:0"
    assert loaded.bundle.manifest.flavor == "stub"
    assert loaded.bundle.manifest.label_schema == SCHEMA


def test_load_returns_just_the_predictor(tmp_path):
    root = _write_bundle(tmp_path / "bundle")
    predictor = ModelRegistry(LocalPathSource(root), {"stub": StubPredictor}).load()

    assert isinstance(predictor, StubPredictor)
    assert predictor.device == "cpu"
    assert predictor.predict(["a", "b"]) == [0, 0]


def test_unknown_flavor_names_the_flavor_and_the_known_ones(tmp_path):
    root = _write_bundle(tmp_path / "bundle", flavor="mystery")
    registry = ModelRegistry(LocalPathSource(root), {"zeta": StubPredictor, "alpha": StubPredictor})

    with pytest.raises(UnknownFlavorError) as excinfo:
        registry.load()

    assert excinfo.value.flavor == "mystery"
    assert excinfo.value.known == ["alpha", "zeta"]
    assert "mystery" in str(excinfo.value)


def test_mlflow_run_source_downloads_the_requested_artifact_path(fake_mlflow, tmp_path):
    fake_mlflow.download_dir = tmp_path / "downloaded"
    source = MLflowRunSource("run-1", artifact_path="bundle", tracking_uri="file:///tracking")

    assert source.resolve() == tmp_path / "downloaded"
    assert fake_mlflow.calls == [
        ("set_tracking_uri", "file:///tracking"),
        ("download_artifacts", "run-1", "bundle"),
    ]


def test_mlflow_run_source_leaves_the_tracking_uri_alone_unless_given_one(fake_mlflow):
    MLflowRunSource("run-1").resolve()

    assert fake_mlflow.calls == [("download_artifacts", "run-1", "model")]


def test_registry_can_load_a_bundle_downloaded_from_an_mlflow_run(fake_mlflow, tmp_path):
    fake_mlflow.download_dir = _write_bundle(tmp_path / "downloaded")

    predictor = ModelRegistry(MLflowRunSource("run-1"), {"stub": StubPredictor}).load()

    assert isinstance(predictor, StubPredictor)
