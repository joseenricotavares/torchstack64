from __future__ import annotations

from typing import TYPE_CHECKING

from dlkit.training.base import TrainerCallback

if TYPE_CHECKING:
    import optuna


def build(*, enable_mlflow: bool = True, progress: bool = True, plot: bool = True,
    optuna_trial: optuna.Trial | None = None, keep_plot: bool = False,
) -> list[TrainerCallback]:
    """Assembles the common callback stack from a few toggles.

    `optuna_trial` takes the `optuna.Trial` itself
    """
    callbacks: list[TrainerCallback] = []

    if enable_mlflow:
        from dlkit.training.callbacks.mlflow_callback import MLflowCallback
        callbacks.append(MLflowCallback())
        
    if progress:
        from dlkit.training.callbacks.rich_progress import RichProgressCallback
        callbacks.append(RichProgressCallback())

    if plot:
        from dlkit.training.callbacks.history_plot import HistoryPlotCallback
        callbacks.append(HistoryPlotCallback(keep_plot, log_to_mlflow=enable_mlflow))

    if optuna_trial is not None:
        from dlkit.training.callbacks.optuna_pruning import OptunaPruningCallback
        callbacks.append(OptunaPruningCallback(optuna_trial))

    return callbacks
