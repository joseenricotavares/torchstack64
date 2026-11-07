from __future__ import annotations

from dlkit.evaluation.evaluator import EvaluationResult


def log_evaluation(
    result: EvaluationResult, *, phase: str | None = None, run_id: str | None = None
) -> None:
    """Logs every metric in `result.values` to the active or given MLflow run."""
    import mlflow

    prefix = f"{phase}_" if phase else ""
    metrics = {f"{prefix}{name}": value for name, value in result.values.items()}

    if run_id is not None:
        with mlflow.start_run(run_id=run_id):
            mlflow.log_metrics(metrics)
    else:
        mlflow.log_metrics(metrics)
