from __future__ import annotations

import csv
import random
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from dlkit.training.base import NNTrainer, TrainerCallback


def build_training_curve_figure(
    entries: Sequence[dict[str, float]],
    metric: str,
    *,
    title: str = "Training History",
):
    """Returns a matplotlib `Figure` plotting train/val loss and `metric` per epoch.

    `entries` is the raw list of per-epoch metric dicts (e.g.
    `TrainingHistory.entries`), each expected to contain `train_loss`,
    `val_loss`, and `metric`.
    """
    import matplotlib

    matplotlib.use("Agg")  # headless: this module never renders interactively
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    epochs = list(range(1, len(entries) + 1))
    train_loss = [e["train_loss"] for e in entries]
    val_loss = [e["val_loss"] for e in entries]
    metric_values = [e[metric] for e in entries]

    width = min(max(8, len(entries) * 0.25), 20)
    fig, ax_loss = plt.subplots(figsize=(width, 4))
    ax_metric = ax_loss.twinx()

    line_kwargs = {"linewidth": 2, "marker": "o", "markersize": 3}
    ax_loss.plot(epochs, train_loss, label="Train Loss", color="tab:blue", **line_kwargs)
    ax_loss.plot(epochs, val_loss, label="Validation Loss", color="tab:cyan", **line_kwargs)
    ax_metric.plot(epochs, metric_values, label=metric, color="tab:orange", **line_kwargs)

    ax_loss.set(title=title, xlabel="Epoch", ylabel="Loss")
    ax_metric.set_ylabel(metric)
    ax_loss.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax_loss.grid(True)

    lines = ax_loss.get_lines() + ax_metric.get_lines()
    ax_loss.legend(lines, [line.get_label() for line in lines], loc="best")
    fig.tight_layout()
    return fig


class HistoryPlotCallback(TrainerCallback):
    """Plots the training history and, optionally, logs it as an MLflow artifact."""

    def __init__(
        self,
        keep_local: bool = False,
        on_figure: Callable[[Any], None] | None = None,
        log_to_mlflow: bool = True,
    ) -> None:
        suffix = random.randint(1000, 9999)
        self.filename = Path(f"training_history_{suffix}.png")
        self.keep_local = keep_local
        self.on_figure = on_figure
        self.log_to_mlflow = log_to_mlflow

    def on_train_end(self, trainer: NNTrainer) -> None:
        import matplotlib

        matplotlib.use("Agg") 
        import matplotlib.pyplot as plt

        entries = trainer.history.entries
        if not entries:
            return

        fig = build_training_curve_figure(entries, trainer.monitor.name)
        if self.on_figure is not None:
            self.on_figure(fig)

        if not self.log_to_mlflow and not self.keep_local:
            plt.close(fig)
            return

        self.filename.parent.mkdir(parents=True, exist_ok=True)
        csv_path = self.filename.with_suffix(".csv")
        try:
            fig.savefig(self.filename, dpi=300, bbox_inches="tight")
            self._write_history_csv(csv_path, entries)
            if self.log_to_mlflow:
                import mlflow

                mlflow.log_artifact(str(self.filename))
                mlflow.log_artifact(str(csv_path))
        finally:
            plt.close(fig)
            if not self.keep_local:
                for path in (self.filename, csv_path):
                    if path.exists():
                        path.unlink()

    @staticmethod
    def _write_history_csv(csv_path: Path, entries: list[dict[str, float]]) -> None:
        fieldnames = ["epoch", *entries[0].keys()]
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for epoch, entry in enumerate(entries, start=1):
                writer.writerow({"epoch": epoch, **entry})
