"""Training discipline and end-to-end scoring."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pipelines.feature_pipeline import FeaturePipeline
from pipelines.prediction_pipeline import PredictionPipeline
from pipelines.preprocessing_pipeline import PreprocessingPipeline
from pipelines.training_pipeline import TrainingPipeline
from utils.model_training_utils import stratified_three_way_split
from utils.pipeline_errors import MLSystemFault
from utils.prediction_utils import EnsembleModels, band_for


def test_three_way_split_is_disjoint_and_stratified():
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"a": rng.normal(size=600)})
    y = pd.Series((rng.random(600) < 0.25).astype(int))

    X_train, X_val, X_test, y_train, y_val, y_test = stratified_three_way_split(
        X, y, test_size=0.2, val_size=0.2, seed=42
    )

    # No row appears in two splits.
    indices = [set(part.index) for part in (X_train, X_val, X_test)]
    assert indices[0].isdisjoint(indices[1])
    assert indices[0].isdisjoint(indices[2])
    assert indices[1].isdisjoint(indices[2])
    assert sum(len(i) for i in indices) == len(X)

    # Proportions honour the requested sizes.
    assert len(X_test) == pytest.approx(len(X) * 0.2, abs=2)
    assert len(X_val) == pytest.approx(len(X) * 0.2, abs=2)

    # Stratification preserves the base rate in every split.
    for part in (y_train, y_val, y_test):
        assert part.mean() == pytest.approx(y.mean(), abs=0.05)


def test_test_split_is_never_used_for_early_stopping():
    """Guards the property the metrics depend on: val and test are separate.

    If these ever became the same rows, every reported score would be a
    training score and the metrics artifact would be meaningless.
    """
    X = pd.DataFrame({"a": range(500)})
    y = pd.Series([0, 1] * 250)
    _, X_val, X_test, _, _, _ = stratified_three_way_split(X, y, 0.2, 0.2, 7)
    assert set(X_val.index).isdisjoint(set(X_test.index))


@pytest.mark.parametrize(
    ("score", "band"),
    [(0.95, "high"), (0.70, "high"), (0.69, "medium"), (0.40, "medium"), (0.39, "low"), (0.0, "low")],
)
def test_risk_bands_are_contiguous(score, band):
    assert band_for(score)[0] == band


def test_missing_model_directory_raises_registered_code(tmp_path):
    with pytest.raises(MLSystemFault) as excinfo:
        EnsembleModels.load_from_directory(str(tmp_path / "absent"))
    assert excinfo.value.code == "SYS-301"


@pytest.mark.slow
def test_full_pipeline_trains_scores_and_reports(local_settings):
    """Preprocess -> features -> train -> predict, on synthetic data."""
    df = PreprocessingPipeline(local_settings).run()
    views = FeaturePipeline(local_settings).run(df)
    report = TrainingPipeline(local_settings).run(views)

    # Metrics artifact is written and internally consistent.
    metrics_path = Path(local_settings.outputs.metrics_path)
    assert metrics_path.exists()
    saved = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert saved["summary"]["models_trained"] == sum(len(v) for v in views.values())
    assert saved["dataset"]["sha256"] == local_settings.data.sha256

    for per_view in report["segments"].values():
        for metrics in per_view.values():
            assert 0.0 <= metrics["roc_auc"] <= 1.0
            assert 0.0 <= metrics["f1"] <= 1.0
            assert metrics["n_test"] > 0
            assert metrics["top_features"]

    # Every model was persisted where the prediction stage expects it.
    model_dir = Path(local_settings.outputs.model_dir)
    saved_models = list(model_dir.glob("*/model.cb"))
    assert len(saved_models) == saved["summary"]["models_trained"]

    # Scoring runs and produces a well-formed risk table.
    scored = PredictionPipeline(local_settings).run(df)
    assert scored["churn_risk_score"].between(0, 1).all()
    assert set(scored["risk_band"]) <= {"low", "medium", "high"}
    assert scored["customerID"].duplicated().sum() == 0

    # Customers are scored only by a model trained on their own segment.
    # A segment too small to train is left unscored rather than routed to
    # another segment's model, so the scored set is exactly the trained set.
    trained_segments = set(report["segments"])
    assert set(scored["segment"]) == trained_segments
    expected = int(df[local_settings.segmentation.column].map(local_settings.segment_of).isin(trained_segments).sum())
    assert len(scored) == expected
