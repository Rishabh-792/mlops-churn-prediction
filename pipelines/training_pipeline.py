"""
training_pipeline.py

Stage 3: feature matrices -> trained per-segment ensemble + metrics artifact.

One CatBoost classifier per (segment, feature view). Every model is early-stopped
on a validation split and scored on a test split it has never seen; only those
test metrics are written to the metrics artifact, which is what the README
quotes. MLflow tracking is file-based by default, so a full run needs no server.

Usage:
    python -m pipelines.training_pipeline
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import mlflow

from utils.core_utils import ensure_dir, get_logger
from utils.model_training_utils import feature_importances, train_segment_model
from utils.settings_manager import PipelineSettings, SettingsManager

logger = get_logger("training")


class TrainingPipeline:
    """Fits one model per segment and feature view, and records the results."""

    def __init__(self, settings: PipelineSettings) -> None:
        self.settings = settings

    def run(self, segment_views: dict) -> dict:
        mlflow.set_tracking_uri(self.settings.tracking.tracking_uri)
        mlflow.set_experiment(self.settings.tracking.experiment_name)

        cb = self.settings.training.catboost
        report: dict = {
            "project": self.settings.project_name,
            "trained_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "dataset": {
                "source_url": self.settings.data.source_url,
                "sha256": self.settings.data.sha256,
            },
            "params": cb.as_params(),
            "segments": {},
        }

        with mlflow.start_run(run_name="churn-ensemble"):
            mlflow.log_params(cb.as_params())

            for segment, views in segment_views.items():
                report["segments"][segment] = {}
                for view, (features, target) in views.items():
                    run_name = f"{segment}_{view}"
                    model, result = train_segment_model(
                        features=features,
                        target=target,
                        params=cb.as_params(),
                        run_name=run_name,
                        test_size=self.settings.training.test_size,
                        val_size=self.settings.training.val_size,
                        seed=self.settings.training.random_seed,
                        early_stopping_rounds=cb.early_stopping_rounds,
                        model_dir=self.settings.outputs.model_dir,
                    )
                    report["segments"][segment][view] = {
                        **result.to_dict(),
                        "top_features": feature_importances(model, features),
                    }

            report["summary"] = self._summarise(report["segments"])
            mlflow.log_metrics(
                {f"mean_{k}": v for k, v in report["summary"].items() if isinstance(v, float)}
            )

        self._write(report)
        return report

    def _summarise(self, segments: dict) -> dict:
        """Row-count-weighted averages, so a small segment cannot flatter the headline."""
        rows = 0
        weighted = {"roc_auc": 0.0, "pr_auc": 0.0, "f1": 0.0}
        model_count = 0
        for views in segments.values():
            for metrics in views.values():
                n = metrics["n_test"]
                rows += n
                model_count += 1
                for key in weighted:
                    weighted[key] += metrics[key] * n
        summary = {k: round(v / rows, 6) for k, v in weighted.items()} if rows else {}
        summary["models_trained"] = model_count
        summary["total_test_rows"] = rows
        return summary

    def _write(self, report: dict) -> None:
        path = Path(self.settings.outputs.metrics_path)
        ensure_dir(str(path.parent))
        path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        logger.info("Wrote metrics to %s", path)

        summary = report["summary"]
        logger.info(
            "Ensemble: %d models | weighted ROC-AUC %.4f | PR-AUC %.4f | F1 %.4f",
            summary["models_trained"],
            summary["roc_auc"],
            summary["pr_auc"],
            summary["f1"],
        )


def main() -> int:
    from pipelines.feature_pipeline import FeaturePipeline

    settings = SettingsManager().load()
    segment_views = FeaturePipeline(settings).run()
    TrainingPipeline(settings).run(segment_views)
    return 0


if __name__ == "__main__":
    sys.exit(main())
