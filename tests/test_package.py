import importlib
from importlib import metadata

import dlkit


def test_the_package_knows_its_version() -> None:
    assert dlkit.__version__


def test_the_package_uses_a_fallback_when_uninstalled(monkeypatch) -> None:
    def missing_version(_: str) -> str:
        raise dlkit.PackageNotFoundError("dlkit")

    with monkeypatch.context() as patch:
        patch.setattr(metadata, "version", missing_version)
        importlib.reload(dlkit)
        assert dlkit.__version__ == "0.0.0"

    importlib.reload(dlkit)
