import sys
from pathlib import Path

import pytest

from dlkit.training.callbacks.mlflow_callback import MLflowCallback
from dlkit.training.history import BestCheckpoint, EpochResult

BEST = BestCheckpoint(epoch=1, score=0.4, state={}, targets=[], preds=[])


def _result(epoch: int = 1) -> EpochResult:
    return EpochResult(
        epoch=epoch, metrics={"train_loss": 0.5, "val_loss": 0.4}, improved=True, patience=0, best=BEST
    )


class StubTrainer:
    """Only what `MLflowCallback` touches: `best` and `save()`."""

    def __init__(self, best: BestCheckpoint | None = BEST) -> None:
        self.best = best
        self.saved_to: Path | None = None

    def save(self, output_dir) -> None:
        self.saved_to = Path(output_dir)
        self.saved_to.mkdir(parents=True)
        (self.saved_to / "bundle.json").write_text("{}")


def test_epoch_metrics_are_logged_with_their_epoch_as_the_step(fake_mlflow):
    fake_mlflow.run_active = True

    MLflowCallback().on_epoch_end(StubTrainer(), _result(epoch=3))

    assert fake_mlflow.calls == [("log_metrics", {"train_loss": 0.5, "val_loss": 0.4}, 3)]


def test_train_end_logs_the_training_time_and_uploads_the_bundle(fake_mlflow, tmp_path):
    fake_mlflow.run_active = True
    callback = MLflowCallback(artifact_dir=str(tmp_path / "model"))
    trainer = StubTrainer()

    callback.on_train_begin(trainer)
    callback.on_train_end(trainer)

    (time_call, upload_call) = fake_mlflow.calls
    assert time_call[:2] == ("log_metric", "train_time_seconds")
    assert time_call[2] >= 0
    assert upload_call == ("log_artifacts", "model", ["bundle.json"])


def test_the_local_bundle_directory_is_removed_after_the_upload(fake_mlflow, tmp_path):
    fake_mlflow.run_active = True
    callback = MLflowCallback(artifact_dir=str(tmp_path / "model"))
    trainer = StubTrainer()

    callback.on_train_end(trainer)

    assert trainer.saved_to is not None
    assert not trainer.saved_to.exists()


def test_the_local_bundle_directory_is_removed_even_when_the_upload_fails(fake_mlflow, tmp_path, monkeypatch):
    def failing_upload(local_dir, artifact_path=None):
        raise RuntimeError("tracking server unreachable")

    fake_mlflow.run_active = True
    monkeypatch.setattr(fake_mlflow, "log_artifacts", failing_upload)
    callback = MLflowCallback(artifact_dir=str(tmp_path / "model"))
    trainer = StubTrainer()

    with pytest.raises(RuntimeError, match="unreachable"):
        callback.on_train_end(trainer)

    assert trainer.saved_to is not None
    assert not trainer.saved_to.exists()


def test_without_a_best_checkpoint_only_the_training_time_is_logged(fake_mlflow, tmp_path):
    fake_mlflow.run_active = True
    callback = MLflowCallback(artifact_dir=str(tmp_path / "model"))
    trainer = StubTrainer(best=None)

    callback.on_train_begin(trainer)
    callback.on_train_end(trainer)

    assert [call[0] for call in fake_mlflow.calls] == ["log_metric"]
    assert trainer.saved_to is None


def test_nothing_is_logged_or_saved_without_an_active_run(fake_mlflow, tmp_path):
    callback = MLflowCallback(artifact_dir=str(tmp_path / "model"))
    trainer = StubTrainer()

    callback.on_train_begin(trainer)
    callback.on_epoch_end(trainer, _result())
    callback.on_train_end(trainer)

    assert fake_mlflow.calls == []
    assert trainer.saved_to is None


def test_the_callback_is_a_no_op_when_mlflow_is_not_installed(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "mlflow", None)  # makes `import mlflow` raise ModuleNotFoundError
    callback = MLflowCallback(artifact_dir=str(tmp_path / "model"))
    trainer = StubTrainer()

    callback.on_train_begin(trainer)
    callback.on_epoch_end(trainer, _result())
    callback.on_train_end(trainer)  # must not raise

    assert trainer.saved_to is None


def test_a_failing_run_lookup_counts_as_no_active_run(fake_mlflow, tmp_path, monkeypatch):
    def broken_lookup():
        raise RuntimeError("tracking store unavailable")

    monkeypatch.setattr(fake_mlflow, "active_run", broken_lookup)
    trainer = StubTrainer()

    MLflowCallback(artifact_dir=str(tmp_path / "model")).on_train_end(trainer)

    assert fake_mlflow.calls == []
    assert trainer.saved_to is None
