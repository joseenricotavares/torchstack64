import json

import pytest
from dlkit.artifacts.bundle import ArtifactBundle, BundleFile, LabelSchema


def test_write_then_read_roundtrips_manifest_and_files(tmp_path):
    root = tmp_path / "bundle"
    schema = LabelSchema(kind="categorical", classes=(1, 2, 3))

    written = ArtifactBundle.write(
        root,
        flavor="fake",
        label_schema=schema,
        files={
            "weights": BundleFile("weights.bin", lambda p: p.write_bytes(b"\x00\x01")),
            "config": BundleFile("config.json", lambda p: p.write_text(json.dumps({"a": 1}))),
        },
        metrics={"val_accuracy": 0.9},
    )

    assert written.path_for("weights").read_bytes() == b"\x00\x01"

    read_back = ArtifactBundle.read(root)
    assert read_back.manifest.flavor == "fake"
    assert read_back.manifest.label_schema == schema
    assert read_back.manifest.metrics == {"val_accuracy": 0.9}
    assert read_back.path_for("weights").read_bytes() == b"\x00\x01"
    assert json.loads(read_back.path_for("config").read_text()) == {"a": 1}


def test_path_for_unknown_logical_name_raises_key_error(tmp_path):
    bundle = ArtifactBundle.write(
        tmp_path / "bundle",
        flavor="fake",
        label_schema=LabelSchema(kind="binary", classes=(0, 1)),
        files={"weights": BundleFile("weights.bin", lambda p: p.write_bytes(b""))},
    )
    with pytest.raises(KeyError):
        bundle.path_for("does-not-exist")


def test_read_missing_manifest_raises_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        ArtifactBundle.read(tmp_path / "does-not-exist")


def test_read_missing_artifact_file_raises_file_not_found(tmp_path):
    root = tmp_path / "bundle"
    ArtifactBundle.write(
        root,
        flavor="fake",
        label_schema=LabelSchema(kind="binary", classes=(0, 1)),
        files={"weights": BundleFile("weights.bin", lambda p: p.write_bytes(b""))},
    )
    (root / "weights.bin").unlink()

    with pytest.raises(FileNotFoundError):
        ArtifactBundle.read(root)
