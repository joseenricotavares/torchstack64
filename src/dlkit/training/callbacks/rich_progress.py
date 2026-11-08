from __future__ import annotations

from typing import TYPE_CHECKING

from dlkit.training.base import NNTrainer, TrainerCallback
from dlkit.training.history import EpochResult

if TYPE_CHECKING:
    from rich.live import Live
    from rich.panel import Panel


class RichProgressCallback(TrainerCallback):
    """Displays training progress using a live Rich table. Requires `dlkit[callbacks-rich]`."""

    def __init__(self) -> None:
        from rich.console import Console

        self.console = Console()
        self.live: Live | None = None
        self.last_table: Panel | None = None

    def on_train_begin(self, trainer: NNTrainer) -> None:
        from rich.live import Live

        self.live = Live(console=self.console, refresh_per_second=4)
        self.live.start()

    def on_epoch_end(self, trainer: NNTrainer, result: EpochResult) -> None:
        assert self.live is not None
        self.last_table = self._build_table(trainer, result)
        self.live.update(self.last_table)

    def on_train_end(self, trainer: NNTrainer) -> None:
        if self.live is not None:
            self.live.stop()
        if self.last_table is not None:
            self.console.print(self.last_table)

    @staticmethod
    def _format_value(value: object) -> str:
        try:
            return f"{float(value):.4f}"  # type: ignore[arg-type]  # deliberately duck-typed, falls back below
        except (TypeError, ValueError):
            return str(value)

    def _build_table(self, trainer: NNTrainer, result: EpochResult) -> Panel:
        from rich.panel import Panel
        from rich.table import Table

        table = Table.grid(padding=(0, 2))
        metric_name = trainer.monitor.name
        best_score = self._format_value(result.best.score) if result.best is not None else "n/a"
        if result.improved:
            best_score += " ✓"

        table.add_column(style="cyan")
        table.add_column(justify="right")
        table.add_row("Epoch", f"{result.epoch}/{trainer.config.epochs}")
        table.add_row("Train Loss", self._format_value(result.metrics["train_loss"]))
        table.add_row("Val Loss", self._format_value(result.metrics["val_loss"]))
        table.add_row(metric_name, self._format_value(result.metrics[metric_name]))
        table.add_row(f"Best {metric_name}", best_score)
        table.add_row("Best Epoch", str(result.best.epoch) if result.best is not None else "n/a")
        table.add_row("Patience", f"{result.patience}/{trainer.config.early_stopping_patience}")
        return Panel(table, title="Training Progress")
