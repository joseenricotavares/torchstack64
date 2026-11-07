from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, NamedTuple


@dataclass(frozen=True, slots=True)
class LabelSchema:
    """Describes the target space a bundle's model was trained on."""

    kind: Literal["binary", "categorical", "ordinal"]
    classes: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class BundleManifest:
    flavor: str
    label_schema: LabelSchema
    artifacts: dict[str, str]
    metrics: dict[str, float] = field(default_factory=dict)
    bundle_version: str = "1"

    def to_json(self) -> dict[str, Any]:
        return {
            "bundle_version": self.bundle_version,
            "flavor": self.flavor,
            "label_schema": {
                "kind": self.label_schema.kind,
                "classes": list(self.label_schema.classes),
            },
            "artifacts": self.artifacts,
            "metrics": self.metrics,
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> BundleManifest:
        schema = data["label_schema"]
        return cls(
            flavor=data["flavor"],
            label_schema=LabelSchema(kind=schema["kind"], classes=tuple(schema["classes"])),
            artifacts=dict(data["artifacts"]),
            metrics=dict(data.get("metrics", {})),
            bundle_version=data.get("bundle_version", "1"),
        )


class BundleFile(NamedTuple):
    """One file to materialize inside a bundle: its relative path, and how to write it."""

    relative_path: str
    write: Callable[[Path], None]


class ArtifactBundle:
    """A directory holding a `bundle.json` manifest plus the files it describes.

    This is the one contract every `NNTrainer.save()` writes and every
    `Predictor.load()` reads - it is what lets training and serving be
    designed together instead of serving being bolted on after the fact.
    """

    MANIFEST_FILENAME = "bundle.json"

    def __init__(self, root: Path, manifest: BundleManifest) -> None:
        self.root = root
        self.manifest = manifest

    def path_for(self, logical_name: str) -> Path:
        """Resolves a logical artifact name (e.g. "weights") to its file on disk."""
        try:
            relative_path = self.manifest.artifacts[logical_name]
        except KeyError:
            raise KeyError(
                f"Bundle at {self.root} has no artifact named {logical_name!r}; "
                f"known artifacts: {sorted(self.manifest.artifacts)}"
            ) from None
        return self.root / relative_path

    @classmethod
    def write(cls, root: str | Path, *, flavor: str, label_schema: LabelSchema,
        files: dict[str, BundleFile], metrics: dict[str, float] | None = None,
    ) -> ArtifactBundle:
        """Materializes every file in `files`, then writes the manifest describing them.

        `files` maps a logical artifact name (used later via `path_for()`)
        to a `BundleFile(relative_path, write)` - decoupling how callers
        refer to an artifact from the actual filename on disk.
        """
        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)

        artifacts: dict[str, str] = {}
        for logical_name, bundle_file in files.items():
            path = root / bundle_file.relative_path
            path.parent.mkdir(parents=True, exist_ok=True)
            bundle_file.write(path)
            artifacts[logical_name] = bundle_file.relative_path

        manifest = BundleManifest(
            flavor=flavor,
            label_schema=label_schema,
            artifacts=artifacts,
            metrics=dict(metrics or {}),
        )
        (root / cls.MANIFEST_FILENAME).write_text(
            json.dumps(manifest.to_json(), indent=2), encoding="utf-8"
        )
        return cls(root=root, manifest=manifest)

    @classmethod
    def read(cls, root: str | Path) -> ArtifactBundle:
        root = Path(root)
        manifest_path = root / cls.MANIFEST_FILENAME
        if not manifest_path.exists():
            raise FileNotFoundError(f"No {cls.MANIFEST_FILENAME} found at {root}")

        manifest = BundleManifest.from_json(json.loads(manifest_path.read_text(encoding="utf-8")))
        bundle = cls(root=root, manifest=manifest)

        missing = [name for name in manifest.artifacts if not bundle.path_for(name).exists()]
        if missing:
            raise FileNotFoundError(f"Bundle at {root} is missing artifact files: {missing}")
        return bundle
