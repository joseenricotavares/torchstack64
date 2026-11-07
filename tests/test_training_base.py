from dataclasses import dataclass

import torch
from dlkit.training.base import NNTrainer, TrainerCallback
from dlkit.training.monitor import Monitor
from torch import nn


@dataclass
class FakeConfig:
    epochs: int
    early_stopping_patience: int
    monitor: Monitor


class RecordingCallback(TrainerCallback):
    def __init__(self) -> None:
        self.events: list = []

    def on_train_begin(self, trainer) -> None:
        self.events.append("begin")

    def on_epoch_end(self, trainer, result) -> None:
        self.events.append(("epoch", result.epoch, result.improved, result.patience))

    def on_train_end(self, trainer) -> None:
        self.events.append("end")


class ScriptedTrainer(NNTrainer):
    """Deterministic trainer: each epoch bumps `model.weight` by 1.0 and
    reports a scripted val_loss, so `fit()`'s control flow can be verified
    without any real data or gradients."""

    def __init__(self, model, config, val_losses, callbacks):
        super().__init__(model, train_loader=None, val_loader=None, config=config,
                          device=torch.device("cpu"), callbacks=callbacks)
        self._val_losses = iter(val_losses)

    def _train_epoch(self) -> float:
        with torch.no_grad():
            self.model.weight.add_(1.0)
        return 0.0

    def predict(self, loader):
        return next(self._val_losses), [0, 1], [0, 1]

    def save(self, output_dir):
        raise NotImplementedError("not exercised in this test")


def _make_model() -> nn.Linear:
    model = nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.zero_()
    return model


# epoch1=0.5 (best) epoch2=0.4 (best) epoch3=0.45 (no improve, patience=1)
# epoch4=0.46 (no improve, patience=2 -> stop before epoch5)
VAL_LOSSES = [0.5, 0.4, 0.45, 0.46, 0.99]


def test_fit_tracks_best_checkpoint_and_early_stops_on_patience():
    config = FakeConfig(epochs=5, early_stopping_patience=2, monitor=Monitor(metric=None, mode="min"))
    callback = RecordingCallback()
    trainer = ScriptedTrainer(_make_model(), config, val_losses=VAL_LOSSES, callbacks=[callback])

    history = trainer.fit()

    assert len(history.entries) == 4  # stopped early, epoch 5 never ran
    assert trainer.best is not None
    assert trainer.best.epoch == 2
    assert trainer.best.score == 0.4


def test_fit_restores_the_best_checkpoint_weights_not_the_last_epoch():
    config = FakeConfig(epochs=5, early_stopping_patience=2, monitor=Monitor(metric=None, mode="min"))
    trainer = ScriptedTrainer(_make_model(), config, val_losses=VAL_LOSSES, callbacks=[])

    trainer.fit()

    # best was captured after epoch 2 (weight bumped twice -> 2.0), even
    # though training continued to epoch 4 (weight would be 4.0) before stopping.
    assert trainer.model.weight.item() == 2.0


def test_fit_fires_callbacks_in_order_with_correct_improved_flags():
    config = FakeConfig(epochs=5, early_stopping_patience=2, monitor=Monitor(metric=None, mode="min"))
    callback = RecordingCallback()
    trainer = ScriptedTrainer(_make_model(), config, val_losses=VAL_LOSSES, callbacks=[callback])

    trainer.fit()

    assert callback.events[0] == "begin"
    assert callback.events[-1] == "end"
    epoch_events = [e for e in callback.events if e[0] == "epoch"]
    assert [e[1] for e in epoch_events] == [1, 2, 3, 4]
    assert [e[2] for e in epoch_events] == [True, True, False, False]
    assert [e[3] for e in epoch_events] == [0, 0, 1, 2]


def test_fit_runs_the_full_epoch_budget_when_never_early_stopped():
    config = FakeConfig(epochs=3, early_stopping_patience=10, monitor=Monitor(metric=None, mode="min"))
    trainer = ScriptedTrainer(_make_model(), config, val_losses=[0.5, 0.4, 0.3], callbacks=[])

    history = trainer.fit()

    assert len(history.entries) == 3
    assert trainer.best.epoch == 3
