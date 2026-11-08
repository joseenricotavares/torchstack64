from __future__ import annotations

from typing import TYPE_CHECKING

from dlkit.training.base import NNTrainer, TrainerCallback
from dlkit.training.history import EpochResult

if TYPE_CHECKING:
    import optuna


class OptunaPruningCallback(TrainerCallback):
    """Reports the monitored metric to an Optuna trial and prunes if it says to.
    Requires `dlkit[callbacks-optuna]`.
    """

    def __init__(self, trial: optuna.Trial) -> None:
        self.trial = trial

    def on_epoch_end(self, trainer: NNTrainer, result: EpochResult) -> None:
        import optuna

        self.trial.report(result.metrics[trainer.monitor.name], step=result.epoch)
        if self.trial.should_prune():
            raise optuna.TrialPruned()
