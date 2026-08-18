"""
Centralized settings management for the ML pipeline.
Parses a unified JSON configuration into strictly typed dataclasses.

Every stage of the pipeline reads its configuration from here rather than
from module-level constants, so a run is fully described by one JSON file.
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .pipeline_enums import OptimizationGoal
from .pipeline_errors import ConfigurationFault

logger = logging.getLogger(__name__)

# =========================================================================
# SETTINGS DATACLASSES
# =========================================================================


@dataclass
class DataSettings:
    raw_path: str
    processed_path: str
    source_url: str
    sha256: str
    id_column: str


@dataclass
class SchemaSettings:
    mandatory_features: list[str]
    categorical_features: list[str]
    target_variable: str
    numeric_features: list[str] = field(default_factory=list)


@dataclass
class SegmentSettings:
    min_activity_threshold: int
    max_activity_threshold: int | None = None


@dataclass
class SegmentationSettings:
    column: str
    segments: dict[str, SegmentSettings]
    description: str = ""


@dataclass
class CatBoostSettings:
    iterations: int = 500
    learning_rate: float = 0.05
    depth: int = 6
    eval_metric: str = "AUC"
    early_stopping_rounds: int = 50
    random_seed: int = 42

    def as_params(self) -> dict[str, Any]:
        """CatBoost constructor kwargs (early stopping is a fit() argument)."""
        return {
            "iterations": self.iterations,
            "learning_rate": self.learning_rate,
            "depth": self.depth,
            "eval_metric": self.eval_metric,
            "random_seed": self.random_seed,
        }


@dataclass
class TrainingSettings:
    test_size: float = 0.2
    val_size: float = 0.2
    random_seed: int = 42
    min_segment_rows: int = 200
    catboost: CatBoostSettings = field(default_factory=CatBoostSettings)


@dataclass
class TuningSettings:
    enabled: bool = False
    n_trials: int = 30
    min_precision: float = 0.5
    min_recall: float = 0.5


@dataclass
class TrackingSettings:
    tracking_uri: str = "sqlite:///mlflow.db"
    experiment_name: str = "churn-prediction"


@dataclass
class OutputSettings:
    model_dir: str = "artifacts/models"
    metrics_path: str = "artifacts/metrics.json"
    predictions_path: str = "artifacts/predictions.csv"


@dataclass
class PipelineSettings:
    """Master typed object representing the entire application state."""

    project_name: str
    data: DataSettings
    schema: SchemaSettings
    segmentation: SegmentationSettings
    training: TrainingSettings
    tuning: TuningSettings
    tracking: TrackingSettings
    outputs: OutputSettings

    def segment_of(self, value: float) -> str:
        """Maps a segmentation-column value onto its segment name."""
        for name, seg in self.segmentation.segments.items():
            lo = seg.min_activity_threshold
            hi = seg.max_activity_threshold
            if value >= lo and (hi is None or value <= hi):
                return name
        return "guest"


# =========================================================================
# SETTINGS MANAGER
# =========================================================================


class SettingsManager:
    """Handles the ingestion and materialization of pipeline settings."""

    def __init__(
        self,
        settings_path: str = "configs/config.json",
        goal: OptimizationGoal = OptimizationGoal.BALANCED,
    ) -> None:
        self.settings_path = Path(settings_path)
        self.goal = goal
        logger.info("SettingsManager initialized | Goal: %s", self.goal.value)

    def load(self) -> PipelineSettings:
        """Loads and parses the JSON configuration."""
        raw_data = self._read_json(self.settings_path)
        try:
            return self._build_settings_object(raw_data)
        except ConfigurationFault:
            raise
        except Exception as exc:
            raise ConfigurationFault(f"Failed to parse settings: {exc}") from exc

    def _read_json(self, path: Path) -> dict[str, Any]:
        if not path.exists():
            raise ConfigurationFault(f"Settings file missing at {path}", code="SYS-201")
        with open(path, encoding="utf-8") as file:
            return json.load(file)

    def _build_settings_object(self, data: dict[str, Any]) -> PipelineSettings:
        seg_block = data.get("segmentation", {})
        segments = {
            name: SegmentSettings(**values)
            for name, values in seg_block.get("segments", {}).items()
        }
        if not segments:
            raise ConfigurationFault("No segments configured", code="SYS-201")

        training_block = dict(data.get("training", {}))
        catboost_block = training_block.pop("catboost", {})

        return PipelineSettings(
            project_name=data.get("project_name", "Unknown"),
            data=DataSettings(**data["data"]),
            schema=SchemaSettings(**data["schema"]),
            segmentation=SegmentationSettings(
                column=seg_block.get("column", "tenure"),
                description=seg_block.get("description", ""),
                segments=segments,
            ),
            training=TrainingSettings(
                catboost=CatBoostSettings(**catboost_block), **training_block
            ),
            tuning=TuningSettings(**data.get("tuning", {})),
            tracking=TrackingSettings(**data.get("tracking", {})),
            outputs=OutputSettings(**data.get("outputs", {})),
        )
