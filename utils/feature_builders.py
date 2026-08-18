"""
feature_builders.py

Feature engineering and segmentation.

Two ideas drive this module:

1. **Segmentation by lifecycle stage.** Churn drivers are not the same for a
   four-month-old account and a four-year-old one, and the base rates differ
   by a factor of four. Each segment therefore gets its own model rather than
   one global model with a tenure feature.

2. **Two feature views per segment.** An *activity* view (spend and contract
   behaviour) and a *profile* view (demographics and subscribed services).
   Training one model per view and averaging their probabilities gives a
   small, cheap ensemble whose members make different kinds of mistakes.
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)

# Feature views. Columns absent from the frame are skipped at build time, so
# these lists stay declarative rather than defensive.
ACTIVITY_FEATURES: list[str] = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "avg_monthly_spend",
    "services_subscribed",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]

PROFILE_FEATURES: list[str] = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]

FEATURE_VIEWS: dict[str, list[str]] = {
    "activity": ACTIVITY_FEATURES,
    "profile": PROFILE_FEATURES,
}

# Columns counted by engineer_features to derive services_subscribed.
_SERVICE_COLUMNS = [
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adds derived columns used by the activity view.

    - ``avg_monthly_spend``: lifetime charges amortised over tenure. Separates
      a long-tenured low-spender from a new high-spender, which raw
      TotalCharges conflates.
    - ``services_subscribed``: breadth of product adoption, a proxy for
      switching cost.
    """
    out = df.copy()

    tenure = out["tenure"].clip(lower=1)
    out["avg_monthly_spend"] = (out["TotalCharges"] / tenure).round(4)

    present = [c for c in _SERVICE_COLUMNS if c in out.columns]
    subscribed = pd.DataFrame(index=out.index)
    for col in present:
        values = out[col].astype(str)
        subscribed[col] = (~values.isin(["No", "No phone service", "No internet service"])).astype(int)
    out["services_subscribed"] = subscribed.sum(axis=1)

    logger.info("Engineered %d derived features", 2)
    return out


def split_by_segment(df: pd.DataFrame, column: str, thresholds: dict[str, dict]) -> dict[str, pd.DataFrame]:
    """Partitions the frame into named segments on a numeric column.

    Bounds come straight from configuration; a segment with no upper bound is
    open-ended. Returns only non-empty segments.
    """
    frames: dict[str, pd.DataFrame] = {}
    for name, bounds in thresholds.items():
        low = bounds.min_activity_threshold
        high = bounds.max_activity_threshold
        mask = df[column] >= low
        if high is not None:
            mask &= df[column] <= high
        segment = df[mask].copy()
        if segment.empty:
            logger.warning("Segment %s matched no rows", name)
            continue
        frames[name] = segment
        logger.info("Segment %-11s rows=%-6d", name, len(segment))
    return frames


def build_view(df: pd.DataFrame, view: str) -> pd.DataFrame:
    """Selects the columns belonging to a named feature view."""
    if view not in FEATURE_VIEWS:
        raise KeyError(f"unknown feature view: {view!r} (expected {sorted(FEATURE_VIEWS)})")
    columns = [c for c in FEATURE_VIEWS[view] if c in df.columns]
    if not columns:
        raise ValueError(f"no columns of view {view!r} present in frame")
    return df[columns].copy()


def categorical_columns(df: pd.DataFrame) -> list[str]:
    """CatBoost needs categorical columns declared by name."""
    return [c for c in df.columns if df[c].dtype == "object"]
