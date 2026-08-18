"""
feature_pipeline.py

Stage 2: cleaned frame -> per-segment, per-view feature matrices.

Output shape is ``{segment: {view: (features, target)}}`` — one entry per model
the training stage will fit.

Usage:
    python -m pipelines.feature_pipeline
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from utils.core_utils import get_logger
from utils.feature_builders import FEATURE_VIEWS, build_view, split_by_segment
from utils.pipeline_errors import SchemaValidationFault
from utils.settings_manager import PipelineSettings, SettingsManager

logger = get_logger("features")

SegmentViews = dict[str, dict[str, tuple[pd.DataFrame, pd.Series]]]


class FeaturePipeline:
    """Splits the population into segments and materialises each feature view."""

    def __init__(self, settings: PipelineSettings) -> None:
        self.settings = settings

    def run(self, df: pd.DataFrame | None = None) -> SegmentViews:
        if df is None:
            df = self._load()

        target_col = self.settings.schema.target_variable
        segments = split_by_segment(
            df,
            column=self.settings.segmentation.column,
            thresholds=self.settings.segmentation.segments,
        )

        out: SegmentViews = {}
        minimum = self.settings.training.min_segment_rows

        for name, frame in segments.items():
            if len(frame) < minimum:
                # Too few rows to split three ways and still trust the metrics.
                logger.warning(
                    "Skipping segment %s: %d rows < min_segment_rows=%d", name, len(frame), minimum
                )
                continue

            target = frame[target_col]
            out[name] = {
                view: (build_view(frame, view), target) for view in FEATURE_VIEWS
            }
            logger.info(
                "Segment %-11s rows=%-6d churn=%.4f views=%s",
                name,
                len(frame),
                target.mean(),
                ",".join(FEATURE_VIEWS),
            )

        if not out:
            raise SchemaValidationFault("No segment met min_segment_rows", code="SYS-401")

        logger.info("Prepared %d models' worth of features", sum(len(v) for v in out.values()))
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
    FeaturePipeline(settings).run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
