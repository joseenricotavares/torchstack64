from dlkit.evaluation import EvaluationResult
from dlkit.mlflow_logging import log_evaluation

RESULT = EvaluationResult(values={"accuracy": 0.9, "f1_macro": 0.8})


def test_logs_every_metric_to_the_active_run(fake_mlflow):
    log_evaluation(RESULT)

    assert fake_mlflow.calls == [("log_metrics", {"accuracy": 0.9, "f1_macro": 0.8}, None)]


def test_phase_prefixes_the_metric_names(fake_mlflow):
    log_evaluation(RESULT, phase="test")

    assert fake_mlflow.calls == [("log_metrics", {"test_accuracy": 0.9, "test_f1_macro": 0.8}, None)]


def test_run_id_logs_inside_that_run_and_closes_it(fake_mlflow):
    log_evaluation(RESULT, run_id="abc")

    assert fake_mlflow.calls == [
        ("start_run", "abc"),
        ("log_metrics", {"accuracy": 0.9, "f1_macro": 0.8}, None),
        ("end_run", "abc"),
    ]
