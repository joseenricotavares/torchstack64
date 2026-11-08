from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from dlkit.evaluation import (
    AccuracyMetric,
    EvaluationResult,
    Evaluator,
    F1Metric,
    PrecisionMetric,
    RecallMetric,
)

Y_TRUE = [0, 1, 1, 0, 2, 2, 1, 0]
Y_PRED = [0, 1, 0, 0, 2, 1, 1, 1]


def test_accuracy_metric_matches_sklearn():
    metric = AccuracyMetric()
    assert metric.name == "accuracy"
    assert metric.compute(Y_TRUE, Y_PRED) == accuracy_score(Y_TRUE, Y_PRED)


def test_f1_metric_matches_sklearn_for_each_average():
    for average in ("macro", "micro", "weighted"):
        metric = F1Metric(average=average)
        assert metric.name == f"f1_{average}"
        expected = f1_score(Y_TRUE, Y_PRED, average=average, zero_division=0)
        assert metric.compute(Y_TRUE, Y_PRED) == expected


def test_f1_metric_accepts_custom_name():
    metric = F1Metric(average="macro", name="my_f1")
    assert metric.name == "my_f1"


def test_precision_and_recall_metrics_match_sklearn_for_each_average():
    for average in ("macro", "micro", "weighted"):
        precision = PrecisionMetric(average=average)
        recall = RecallMetric(average=average)
        assert precision.name == f"precision_{average}"
        assert recall.name == f"recall_{average}"
        expected_precision = precision_score(Y_TRUE, Y_PRED, average=average, zero_division=0)
        expected_recall = recall_score(Y_TRUE, Y_PRED, average=average, zero_division=0)
        assert precision.compute(Y_TRUE, Y_PRED) == expected_precision
        assert recall.compute(Y_TRUE, Y_PRED) == expected_recall


def test_precision_and_recall_metrics_accept_custom_names():
    assert PrecisionMetric(name="my_precision").name == "my_precision"
    assert RecallMetric(name="my_recall").name == "my_recall"


def test_evaluator_aggregates_all_metrics_into_a_flat_dict():
    evaluator = Evaluator([AccuracyMetric(), F1Metric("macro"), F1Metric("weighted")])
    result = evaluator.evaluate(Y_TRUE, Y_PRED)

    assert isinstance(result, EvaluationResult)
    assert set(result.values) == {"accuracy", "f1_macro", "f1_weighted"}
    assert result["accuracy"] == accuracy_score(Y_TRUE, Y_PRED)


def test_evaluator_has_no_io_side_effects(tmp_path, monkeypatch):
    # Evaluating must not touch the filesystem or any external service -
    # regression guard for the "god method" the original evaluator had.
    monkeypatch.chdir(tmp_path)
    Evaluator([AccuracyMetric()]).evaluate(Y_TRUE, Y_PRED)
    assert list(tmp_path.iterdir()) == []
