"""Optional experiment tracking with MLflow. Nothing else in the project depends on it.

    pip install mlflow
    python -m delivery_risk.pipeline --mlflow
    mlflow ui --backend-store-uri sqlite:///mlflow.db
"""

from __future__ import annotations

from . import config, model


def log_results(results: dict) -> None:
    """Record one MLflow run per stage: settings, backtest metrics, and the report files."""
    import mlflow  # imported here so the rest of the project runs without it

    mlflow.set_tracking_uri(f"sqlite:///{config.ROOT / 'mlflow.db'}")
    mlflow.set_experiment("late-delivery-risk")
    for name, stage in config.STAGES.items():
        with mlflow.start_run(run_name=name):
            mlflow.log_params(
                {
                    "stage": name,
                    "candidate": model.SELECTED[name],
                    **model.CANDIDATES[model.SELECTED[name]],
                    "n_features": len(stage.features),
                    "evaluation_months": f"{config.EVAL_FOLDS[0]}..{config.EVAL_FOLDS[-1]}",
                    "alert_budget": config.ALERT_BUDGET,
                }
            )
            for who in ("model", "rule", "logistic"):
                for metric, value in results["stages"][name][who].items():
                    mlflow.log_metric(f"{who}_{metric}", value)
            mlflow.log_artifact(str(config.REPORT_DIR / "metrics.json"))
            mlflow.log_artifacts(str(config.FIGURE_DIR), artifact_path="figures")
