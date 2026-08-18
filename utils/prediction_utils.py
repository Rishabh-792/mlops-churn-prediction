"""
prediction_utils.py

Loading and scoring for the trained per-segment ensemble.

A customer is routed to the models for their own segment, scored by each
feature view, and the view probabilities are averaged. Routing is explicit:
a customer whose segment has no trained model is reported rather than
silently scored by the wrong one.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
from catboost import CatBoostClassifier, Pool

from .feature_builders import build_view
from .pipeline_errors import MLSystemFault

logger = logging.getLogger(__name__)

# Risk bands, and the action each one implies. Thresholds are business
# choices, not model outputs, which is why they live here and not in the model.
RISK_BANDS = [
    (0.70, "high", "priority_retention_outreach"),
    (0.40, "medium", "targeted_offer"),
    (0.00, "low", "monitor"),
]


@dataclass
class EnsembleModels:
    """Trained models keyed by segment, then by feature view."""

    models: dict[str, dict[str, CatBoostClassifier]] = field(default_factory=dict)

    @classmethod
    def load_from_directory(cls, base_dir: str) -> EnsembleModels:
        """Loads every ``<segment>_<view>/model.cb`` under a directory."""
        root = Path(base_dir)
        if not root.exists():
            raise MLSystemFault("SYS-301", f"Model directory not found: {root}")

        models: dict[str, dict[str, CatBoostClassifier]] = {}
        for artifact in sorted(root.glob("*/model.cb")):
            name = artifact.parent.name
            if "_" not in name:
                logger.warning("Skipping unrecognised model directory: %s", name)
                continue
            segment, view = name.rsplit("_", 1)
            model = CatBoostClassifier()
            model.load_model(str(artifact))
            models.setdefault(segment, {})[view] = model

        if not models:
            raise MLSystemFault("SYS-301", f"No model artifacts found under {root}")

        total = sum(len(v) for v in models.values())
        logger.info("Loaded %d models across %d segments", total, len(models))
        return cls(models=models)

    @property
    def segments(self) -> list[str]:
        return sorted(self.models)

    def score(self, frame: pd.DataFrame, segment: str) -> pd.Series:
        """Averages the per-view churn probabilities for one segment's rows."""
        if segment not in self.models:
            raise MLSystemFault("SYS-401", f"No trained model for segment {segment!r}")

        probabilities = []
        for view, model in self.models[segment].items():
            features = build_view(frame, view)
            cat_features = [c for c in features.columns if features[c].dtype == "object"]
            pool = Pool(features, cat_features=cat_features)
            probabilities.append(model.predict_proba(pool)[:, 1])

        stacked = pd.DataFrame(probabilities).T
        return stacked.mean(axis=1).set_axis(frame.index)


def band_for(score: float) -> tuple[str, str]:
    """Maps a probability onto its (risk band, recommended action)."""
    for threshold, band, action in RISK_BANDS:
        if score >= threshold:
            return band, action
    return "low", "monitor"
