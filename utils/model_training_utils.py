"""
model_training_utils.py

Training and evaluation for the per-segment CatBoost classifiers.

Evaluation discipline this module enforces:

- Three-way split. The model early-stops on the validation split and is scored
  on a test split it has never seen. Reporting validation scores as headline
  numbers overstates performance, so the test split is the only thing that
  reaches the metrics artifact.
- Stratified splits, because segment churn rates range from 14% to 55%.
- Class weighting rather than resampling, so the probability calibration used
  by the risk score stays interpretable.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass

import mlflow
import mlflow.catboost
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from mlflow.models.signature import infer_signature
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Test-set metrics for one trained model."""

    roc_auc: float
    pr_auc: float
    f1: float
    precision: float
    recall: float
    brier: float
    n_train: int
    n_val: int
    n_test: int
    positive_rate: float

    def to_dict(self) -> dict[str, float]:
        return {k: (round(v, 6) if isinstance(v, float) else v) for k, v in asdict(self).items()}


def stratified_three_way_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float,
    val_size: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Splits into train/val/test, stratified on the label."""
    X_rest, X_test, y_rest, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    # val_size is expressed as a fraction of the whole, so rescale it.
    relative_val = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_rest, y_rest, test_size=relative_val, random_state=seed, stratify=y_rest
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def evaluate(model: CatBoostClassifier, X_test: pd.DataFrame, y_test: pd.Series,
             cat_features: list[str], sizes: tuple[int, int, int]) -> EvaluationResult:
    """Scores a fitted model on a held-out split."""
    pool = Pool(X_test, cat_features=cat_features)
    proba = model.predict_proba(pool)[:, 1]
    pred = (proba >= 0.5).astype(int)

    n_train, n_val, n_test = sizes
    return EvaluationResult(
        roc_auc=float(roc_auc_score(y_test, proba)),
        pr_auc=float(average_precision_score(y_test, proba)),
        f1=float(f1_score(y_test, pred, zero_division=0)),
        precision=float(precision_score(y_test, pred, zero_division=0)),
        recall=float(recall_score(y_test, pred, zero_division=0)),
        brier=float(brier_score_loss(y_test, proba)),
        n_train=n_train,
        n_val=n_val,
        n_test=n_test,
        positive_rate=float(y_test.mean()),
    )


def train_segment_model(
    features: pd.DataFrame,
    target: pd.Series,
    params: dict,
    run_name: str,
    test_size: float = 0.2,
    val_size: float = 0.2,
    seed: int = 42,
    early_stopping_rounds: int = 50,
    model_dir: str | None = None,
) -> tuple[CatBoostClassifier, EvaluationResult]:
    """Trains one CatBoost classifier and logs it to MLflow.

    Returns the fitted model and its held-out test metrics.
    """
    cat_features = [c for c in features.columns if features[c].dtype == "object"]
    X_train, X_val, X_test, y_train, y_val, y_test = stratified_three_way_split(
        features, target, test_size, val_size, seed
    )

    # Counter the class imbalance without resampling.
    positives = max(int(y_train.sum()), 1)
    negatives = max(len(y_train) - positives, 1)
    scale_pos_weight = negatives / positives

    with mlflow.start_run(run_name=run_name, nested=True):
        model = CatBoostClassifier(
            **params,
            cat_features=cat_features,
            scale_pos_weight=scale_pos_weight,
            verbose=False,
            allow_writing_files=False,
        )
        model.fit(
            X_train,
            y_train,
            eval_set=(X_val, y_val),
            early_stopping_rounds=early_stopping_rounds,
            use_best_model=True,
        )

        result = evaluate(
            model, X_test, y_test, cat_features, (len(X_train), len(X_val), len(X_test))
        )

        mlflow.log_params({**params, "scale_pos_weight": round(scale_pos_weight, 4)})
        mlflow.log_metrics(result.to_dict())
        mlflow.log_param("best_iteration", model.get_best_iteration())

        signature = infer_signature(X_test, model.predict_proba(X_test)[:, 1])
        mlflow.catboost.log_model(model, name="model", signature=signature)

        if model_dir:
            from pathlib import Path

            out = Path(model_dir) / run_name
            out.mkdir(parents=True, exist_ok=True)
            model.save_model(str(out / "model.cb"))

    logger.info(
        "%-26s ROC-AUC %.4f | PR-AUC %.4f | F1 %.4f | test n=%d",
        run_name,
        result.roc_auc,
        result.pr_auc,
        result.f1,
        result.n_test,
    )
    return model, result


def feature_importances(model: CatBoostClassifier, features: pd.DataFrame, top_n: int = 5) -> dict[str, float]:
    """Top-N feature importances, for the model card."""
    scores = model.get_feature_importance()
    order = np.argsort(scores)[::-1][:top_n]
    return {str(features.columns[i]): round(float(scores[i]), 4) for i in order}
