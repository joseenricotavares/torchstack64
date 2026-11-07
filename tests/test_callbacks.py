from dataclasses import dataclass

import torch
from dlkit.training.base import NNTrainer
from dlkit.training.callbacks import HistoryPlotCallback, build
from dlkit.training.monitor import Monitor
from torch import nn


@dataclass
class FakeConfig:
    epochs: int
    early_stopping_patience: int
    monitor: Monitor


class OneEpochTrainer(NNTrainer):
    def __init__(self, model, config, callbacks):
        super().__init__(
            model, train_loader=None, val_loader=None, config=config,
            device=torch.device("cpu"), callbacks=callbacks,
        )

    def _train_epoch(self) -> float:
        return 0.5

    def predict(self, loader):
        return 0.4, [0, 1], [0, 1]

    def save(self, output_dir):
        raise NotImplementedError("not exercised in this test")


def _make_model() -> nn.Linear:
    return nn.Linear(1, 1, bias=False)


def test_build_with_mlflow_disabled_runs_without_an_active_mlflow_run(tmp_path, monkeypatch):
    # Regression guard: build(enable_mlflow=False) used to still include a
    # HistoryPlotCallback that unconditionally called mlflow.log_artifact(),
    # which raised because no MLflow run was active.
    monkeypatch.chdir(tmp_path)
    config = FakeConfig(epochs=1, early_stopping_patience=1, monitor=Monitor(metric=None, mode="min"))
    callbacks = build(enable_mlflow=False, progress=False, plot=True)

    trainer = OneEpochTrainer(_make_model(), config, callbacks)
    trainer.fit()  # must not raise


def test_history_plot_callback_skips_mlflow_logging_when_disabled(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = FakeConfig(epochs=1, early_stopping_patience=1, monitor=Monitor(metric=None, mode="min"))
    callback = HistoryPlotCallback(log_to_mlflow=False)
    trainer = OneEpochTrainer(_make_model(), config, callbacks=[callback])

    trainer.fit()  # would raise if mlflow.log_artifact were called without an active run


def test_history_plot_callback_can_keep_local_files_without_mlflow(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = FakeConfig(epochs=1, early_stopping_patience=1, monitor=Monitor(metric=None, mode="min"))
    callback = HistoryPlotCallback(keep_local=True, log_to_mlflow=False)
    trainer = OneEpochTrainer(_make_model(), config, callbacks=[callback])

    trainer.fit()

    assert callback.filename.exists()
    assert callback.filename.with_suffix(".csv").exists()
