"""
preprocessing_pipeline.py

Stage 1: raw CSV -> validated, typed, cleaned parquet.

Everything downstream assumes this stage has run, so it is deliberately strict:
a schema violation raises rather than being coerced away silently.

Usage:
    python -m pipelines.preprocessing_pipeline
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from utils.core_utils import ensure_dir, get_logger, require_columns
from utils.feature_builders import engineer_features
from utils.pipeline_errors import SchemaValidationFault
from utils.settings_manager import PipelineSettings, SettingsManager

logger = get_logger("preprocessing")


class PreprocessingPipeline:
    """Loads raw data, validates it against the configured schema, cleans it."""

    def __init__(self, settings: PipelineSettings) -> None:
        self.settings = settings

    def run(self) -> pd.DataFrame:
        raw_path = Path(self.settings.data.raw_path)
        if not raw_path.exists():
            raise SchemaValidationFault(
                f"Raw dataset not found at {raw_path}. "
                "Run `python -m scripts.download_data` first.",
                code="SYS-103",
            )

        df = pd.read_csv(raw_path)
        logger.info("Loaded %d rows x %d columns from %s", len(df), len(df.columns), raw_path)

        require_columns(df, self.settings.schema.mandatory_features)
        require_columns(df, [self.settings.schema.target_variable])

        df = self._clean(df)
        df = engineer_features(df)
        df = self._encode_target(df)

        out_path = Path(self.settings.data.processed_path)
        ensure_dir(str(out_path.parent))
        df.to_parquet(out_path, index=False)
        logger.info("Wrote %d rows to %s", len(df), out_path)
        return df

    def _clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handles the dataset's known quality defects."""
        before = len(df)

        # TotalCharges arrives as text and carries blanks for accounts whose
        # first bill has not been issued. Those rows are genuine (tenure == 0),
        # so impute rather than drop: a brand-new account has been charged
        # nothing yet.
        charges = pd.to_numeric(df["TotalCharges"].astype(str).str.strip(), errors="coerce")
        blanks = int(charges.isna().sum())
        if blanks:
            logger.info("Imputing %d blank TotalCharges values as 0.0", blanks)
        df["TotalCharges"] = charges.fillna(0.0)

        df = df.drop_duplicates(subset=[self.settings.data.id_column])
        if len(df) != before:
            logger.info("Dropped %d duplicate rows", before - len(df))

        return df

    def _encode_target(self, df: pd.DataFrame) -> pd.DataFrame:
        """Maps the Yes/No label onto 0/1."""
        target = self.settings.schema.target_variable
        if df[target].dtype == "object":
            mapping = {"Yes": 1, "No": 0}
            unknown = set(df[target].unique()) - set(mapping)
            if unknown:
                raise SchemaValidationFault(
                    f"Unexpected values in target {target!r}: {sorted(unknown)}", code="SYS-102"
                )
            df[target] = df[target].map(mapping)
        df[target] = df[target].astype(int)
        logger.info("Target %s positive rate: %.4f", target, df[target].mean())
        return df


def main() -> int:
    settings = SettingsManager().load()
    PreprocessingPipeline(settings).run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
