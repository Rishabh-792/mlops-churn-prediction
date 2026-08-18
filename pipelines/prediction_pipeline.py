"""
prediction_pipeline.py

Stage 4: trained ensemble + customer frame -> scored risk table.

Each customer is routed to their own segment's models, scored, and assigned a
risk band and a recommended action. Output is a CSV keyed by customer id.

Usage:
    python -m pipelines.prediction_pipeline
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from utils.core_utils import ensure_dir, get_logger
from utils.feature_builders import split_by_segment
from utils.pipeline_errors import SchemaValidationFault
from utils.prediction_utils import EnsembleModels, band_for
from utils.settings_manager import PipelineSettings, SettingsManager

logger = get_logger("prediction")


class PredictionPipeline:
    """Scores a customer population with the trained ensemble."""

    def __init__(self, settings: PipelineSettings) -> None:
        self.settings = settings

    def run(self, df: pd.DataFrame | None = None) -> pd.DataFrame:
        if df is None:
            df = self._load()

        ensemble = EnsembleModels.load_from_directory(self.settings.outputs.model_dir)
        segments = split_by_segment(
            df,
            column=self.settings.segmentation.column,
            thresholds=self.settings.segmentation.segments,
        )

        id_col = self.settings.data.id_column
        results = []
        skipped = 0

        for name, frame in segments.items():
            if name not in ensemble.models:
                # A segment with no trained model is reported, never guessed at.
                logger.warning("No model for segment %s; %d customers unscored", name, len(frame))
                skipped += len(frame)
                continue

            scores = ensemble.score(frame, name)
            bands = [band_for(s) for s in scores]
            results.append(
                pd.DataFrame(
                    {
                        id_col: frame[id_col].to_numpy(),
                        "segment": name,
                        "churn_risk_score": scores.round(6).to_numpy(),
                        "risk_band": [b for b, _ in bands],
                        "recommended_action": [a for _, a in bands],
                    }
                )
            )

        if not results:
            raise SchemaValidationFault("No customers could be scored", code="SYS-302")

        out = pd.concat(results, ignore_index=True).sort_values(
            "churn_risk_score", ascending=False
        )

        path = Path(self.settings.outputs.predictions_path)
        ensure_dir(str(path.parent))
        out.to_csv(path, index=False)

        logger.info(
            "Scored %d customers (%d unscored) | high-risk %d | wrote %s",
            len(out),
            skipped,
            int((out["risk_band"] == "high").sum()),
            path,
        )
        return out

    def _load(self) -> pd.DataFrame:
        path = Path(self.settings.data.processed_path)
        if not path.exists():
            raise SchemaValidationFault(
                f"Processed dataset not found at {path}. "
                "Run `python -m pipelines.preprocessing_pipeline` first.",
                code="SYS-103",
            )
        return pd.read_parquet(path)


def main() -> int:
    settings = SettingsManager().load()
    PredictionPipeline(settings).run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
