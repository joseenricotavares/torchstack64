from types import SimpleNamespace

import pytest

from dlkit.training.callbacks.rich_progress import RichProgressCallback
from dlkit.training.history import BestCheckpoint, EpochResult
from dlkit.training.monitor import Monitor

pytestmark = pytest.mark.usefixtures("fake_rich")

TRAINER = SimpleNamespace(
    monitor=Monitor(metric=None, mode="min"),
    config=SimpleNamespace(epochs=10, early_stopping_patience=3),
)


def _best(epoch: int = 2, score: float = 0.3123) -> BestCheckpoint:
    return BestCheckpoint(epoch=epoch, score=score, state={}, targets=[], preds=[])


def test_the_table_shows_the_epoch_the_losses_and_the_best_so_far():
    callback = RichProgressCallback()
    result = EpochResult(
        epoch=2,
        metrics={"train_loss": 0.5, "val_loss": 0.3123},
        improved=False,
        patience=1,
        best=_best(epoch=1, score=0.2),
    )

    callback.on_train_begin(TRAINER)
    callback.on_epoch_end(TRAINER, result)

    assert callback.last_table is not None
    assert callback.last_table.title == "Training Progress"
    assert callback.last_table.renderable.rows == [
        ("Epoch", "2/10"),
        ("Train Loss", "0.5000"),
        ("Val Loss", "0.3123"),
        ("val_loss", "0.3123"),
        ("Best val_loss", "0.2000"),
        ("Best Epoch", "1"),
        ("Patience", "1/3"),
    ]


def test_an_improving_epoch_marks_the_best_score():
    callback = RichProgressCallback()
    result = EpochResult(
        epoch=2, metrics={"train_loss": 0.5, "val_loss": 0.3123}, improved=True, patience=0, best=_best()
    )

    callback.on_train_begin(TRAINER)
    callback.on_epoch_end(TRAINER, result)

    assert ("Best val_loss", "0.3123 ✓") in callback.last_table.renderable.rows


def test_values_that_are_not_numbers_are_shown_as_they_are():
    callback = RichProgressCallback()
    result = EpochResult(
        epoch=1, metrics={"train_loss": "skipped", "val_loss": 0.4}, improved=False, patience=0, best=None
    )

    callback.on_train_begin(TRAINER)
    callback.on_epoch_end(TRAINER, result)

    rows = callback.last_table.renderable.rows
    assert ("Train Loss", "skipped") in rows
    assert ("Best val_loss", "n/a") in rows
    assert ("Best Epoch", "n/a") in rows


def test_the_live_display_runs_for_the_whole_training_and_the_last_table_is_printed_at_the_end():
    callback = RichProgressCallback()
    result = EpochResult(
        epoch=1, metrics={"train_loss": 0.5, "val_loss": 0.4}, improved=True, patience=0, best=_best(1, 0.4)
    )

    callback.on_train_begin(TRAINER)
    callback.on_epoch_end(TRAINER, result)
    callback.on_train_end(TRAINER)

    assert callback.live is not None
    assert callback.live.events == [("start",), ("update", callback.last_table), ("stop",)]
    assert callback.console.printed == [callback.last_table]


def test_nothing_is_printed_when_no_epoch_ran():
    callback = RichProgressCallback()

    callback.on_train_end(TRAINER)  # on_train_begin never ran either

    assert callback.console.printed == []
