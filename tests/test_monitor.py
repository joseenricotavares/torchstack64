from dlkit.evaluation import AccuracyMetric
from dlkit.training.monitor import Monitor


def test_monitor_with_no_metric_tracks_val_loss():
    monitor = Monitor(metric=None, mode="min")
    assert monitor.name == "val_loss"
    assert monitor.score(0.42, y_true=[1, 0], y_pred=[1, 1]) == 0.42
    assert monitor.improved(current=0.3, best=0.5) is True
    assert monitor.improved(current=0.6, best=0.5) is False


def test_monitor_with_a_metric_computes_and_names_from_it():
    monitor = Monitor(metric=AccuracyMetric(), mode="max")
    assert monitor.name == "val_accuracy"
    score = monitor.score(val_loss=1.0, y_true=[1, 0, 1, 1], y_pred=[1, 0, 0, 1])
    assert score == 0.75
    assert monitor.improved(current=0.8, best=0.7) is True
    assert monitor.improved(current=0.6, best=0.7) is False


def test_monitor_accepts_any_metric_implementation_not_just_dlkit_ones():
    class CustomMetric:
        name = "custom"

        def compute(self, y_true, y_pred):
            return 123.0

    monitor = Monitor(metric=CustomMetric(), mode="max")
    assert monitor.name == "val_custom"
    assert monitor.score(val_loss=0.0, y_true=[], y_pred=[]) == 123.0
