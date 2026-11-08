import sys
import types
from types import SimpleNamespace

import pytest

from dlkit.training.callbacks.optuna_pruning import OptunaPruningCallback
from dlkit.training.history import EpochResult
from dlkit.training.monitor import Monitor


class FakeTrial:
    def __init__(self, prune: bool = False) -> None:
        self.prune = prune
        self.reports: list[tuple[float, int]] = []

    def report(self, value: float, step: int) -> None:
        self.reports.append((value, step))

    def should_prune(self) -> bool:
        return self.prune


@pytest.fixture
def fake_optuna(monkeypatch):
    module = types.ModuleType("optuna")
    module.TrialPruned = type("TrialPruned", (Exception,), {})
    monkeypatch.setitem(sys.modules, "optuna", module)
    return module


TRAINER = SimpleNamespace(monitor=Monitor(metric=None, mode="min"))


def _result(epoch: int) -> EpochResult:
    return EpochResult(
        epoch=epoch, metrics={"train_loss": 0.9, "val_loss": 0.7}, improved=True, patience=0, best=None
    )


def test_the_monitored_metric_is_reported_to_the_trial_at_each_epoch(fake_optuna):
    trial = FakeTrial()
    callback = OptunaPruningCallback(trial)

    callback.on_epoch_end(TRAINER, _result(epoch=1))
    callback.on_epoch_end(TRAINER, _result(epoch=2))

    assert trial.reports == [(0.7, 1), (0.7, 2)]


def test_the_trial_is_pruned_when_optuna_says_so(fake_optuna):
    callback = OptunaPruningCallback(FakeTrial(prune=True))

    with pytest.raises(fake_optuna.TrialPruned):
        callback.on_epoch_end(TRAINER, _result(epoch=1))
