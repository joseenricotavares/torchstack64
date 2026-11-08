"""Stand-ins for the optional extras (mlflow, rich).

CI installs only `.[dev]`, so the code that talks to an extra has to be testable without it.
These fixtures put a small fake into `sys.modules`, which is where the lazy `import mlflow` /
`from rich.console import Console` inside dlkit look first - so the tests behave the same
whether or not the real package happens to be installed.
"""

from __future__ import annotations

import sys
import types
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest


class FakeMlflow(types.ModuleType):
    """Records what dlkit logs, so tests need neither the mlflow extra nor a tracking server."""

    def __init__(self) -> None:
        super().__init__("mlflow")
        self.calls: list[tuple] = []
        self.run_active = False
        self.download_dir = Path("downloaded")
        self.artifacts = types.SimpleNamespace(download_artifacts=self._download_artifacts)

    def active_run(self) -> object | None:
        return object() if self.run_active else None

    def set_tracking_uri(self, uri: str) -> None:
        self.calls.append(("set_tracking_uri", uri))

    @contextmanager
    def start_run(self, run_id: str | None = None) -> Iterator[None]:
        self.calls.append(("start_run", run_id))
        was_active, self.run_active = self.run_active, True
        try:
            yield
        finally:
            self.run_active = was_active
            self.calls.append(("end_run", run_id))

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        self.calls.append(("log_metrics", dict(metrics), step))

    def log_metric(self, key: str, value: float) -> None:
        self.calls.append(("log_metric", key, value))

    def log_artifacts(self, local_dir: str, artifact_path: str | None = None) -> None:
        # Snapshot the directory now: that is what a real upload would see.
        names = sorted(path.name for path in Path(local_dir).iterdir())
        self.calls.append(("log_artifacts", artifact_path, names))

    def _download_artifacts(self, run_id: str, artifact_path: str) -> str:
        self.calls.append(("download_artifacts", run_id, artifact_path))
        return str(self.download_dir)


@pytest.fixture
def fake_mlflow(monkeypatch: pytest.MonkeyPatch) -> FakeMlflow:
    module = FakeMlflow()
    monkeypatch.setitem(sys.modules, "mlflow", module)
    return module


class FakeTable:
    def __init__(self) -> None:
        self.rows: list[tuple] = []

    @classmethod
    def grid(cls, padding: object = 0) -> FakeTable:
        return cls()

    def add_column(self, **kwargs: object) -> None:
        pass

    def add_row(self, *cells: str) -> None:
        self.rows.append(cells)


class FakePanel:
    def __init__(self, renderable: FakeTable, title: str | None = None) -> None:
        self.renderable = renderable
        self.title = title


class FakeConsole:
    def __init__(self) -> None:
        self.printed: list[FakePanel] = []

    def print(self, renderable: FakePanel) -> None:
        self.printed.append(renderable)


class FakeLive:
    def __init__(self, console: FakeConsole, refresh_per_second: int) -> None:
        self.console = console
        self.events: list[tuple] = []

    def start(self) -> None:
        self.events.append(("start",))

    def update(self, renderable: FakePanel) -> None:
        self.events.append(("update", renderable))

    def stop(self) -> None:
        self.events.append(("stop",))


@pytest.fixture
def fake_rich(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replaces the few `rich` classes `RichProgressCallback` uses with recording stand-ins."""
    parts = {
        "rich": {},
        "rich.console": {"Console": FakeConsole},
        "rich.live": {"Live": FakeLive},
        "rich.panel": {"Panel": FakePanel},
        "rich.table": {"Table": FakeTable},
    }
    for name, attributes in parts.items():
        module = types.ModuleType(name)
        vars(module).update(attributes)
        monkeypatch.setitem(sys.modules, name, module)
