"""
inference.py

SageMaker entry point for the CatBoost ensemble container.

This is the same routing and averaging logic the batch prediction pipeline
uses, expressed through SageMaker's handler contract. It reuses
``EnsembleModels`` rather than reimplementing loading, so an online prediction
and an offline one cannot drift apart.

Reference implementation: see the deployment section of the README for what
has and has not been provisioned.
"""

from __future__ import annotations

import json
import os
from typing import Any

import pandas as pd

from utils.feature_builders import engineer_features
from utils.prediction_utils import EnsembleModels, band_for


def model_fn(model_dir: str) -> EnsembleModels:
    """Loads every model in the artifact directory. Runs once at container start.

    SageMaker unpacks model.tar.gz into model_dir; the training pipeline writes
    its artifacts under a "model/" prefix inside that archive.
    """
    base_dir = os.path.join(model_dir, "model")
    if not os.path.isdir(base_dir):
        base_dir = model_dir
    return EnsembleModels.load_from_directory(base_dir)


def input_fn(request_body: str, content_type: str = "application/json") -> dict[str, Any]:
    """Parses the request payload."""
    if content_type != "application/json":
        raise ValueError(f"unsupported content type: {content_type}")
    return json.loads(request_body)


def predict_fn(payload: dict[str, Any], models: EnsembleModels) -> dict[str, Any]:
    """Scores a batch of customers belonging to one segment.

    The segment must be supplied and must have a trained model. Defaulting an
    unknown segment to some other segment's model would return a confident
    number computed by the wrong model, so it raises instead.
    """
    segment = payload.get("segment")
    if segment is None:
        raise ValueError("payload must specify 'segment'")
    if segment not in models.models:
        raise ValueError(
            f"no model for segment {segment!r}; available: {models.segments}"
        )

    frame = engineer_features(pd.DataFrame(payload["features"]))
    scores = models.score(frame, segment)
    bands = [band_for(float(s)) for s in scores]

    return {
        "segment": segment,
        "risk_scores": [round(float(s), 6) for s in scores],
        "risk_bands": [b for b, _ in bands],
        "recommended_actions": [a for _, a in bands],
    }


def output_fn(prediction: dict[str, Any], accept: str = "application/json") -> str:
    """Serialises the response."""
    return json.dumps(prediction)
