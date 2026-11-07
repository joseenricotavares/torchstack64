"""torchstack64: a small PyTorch training loop with callbacks and plots, to one person's taste,
plus the contract that lets what it trains be packed and served."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("dlkit")
except PackageNotFoundError:  # a checkout that was never installed
    __version__ = "0.0.0"
