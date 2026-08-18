"""
model_tune_utils.py

Hyperparameter tuning utilities for CatBoost models using Optuna.
Optimizes F1 score subject to strict precision/recall business constraints.
"""

import logging

import optuna
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

def suggest_catboost_params(trial: optuna.Trial) -> dict:
    """Defines the hyperparameter search space."""
    return {
        "iterations": trial.suggest_int("iterations", 100, 1000),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),
        "depth": trial.suggest_int("depth", 4, 10),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 10),
        "random_strength": trial.suggest_float("random_strength", 1e-9, 10, log=True),
        "eval_metric": "Logloss",
        "verbose": False
    }

def tune_single_model(
    X: pd.DataFrame,
    y: pd.Series,
    cat_features: list[str],
    n_trials: int = 50,
    study_name: str = "churn_tuning",
    min_precision: float = 0.70,
    min_recall: float = 0.70
) -> tuple[dict, optuna.Study]:
    """Optuna study orchestration with business constraints."""

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    def objective(trial: optuna.Trial) -> float:
        params = suggest_catboost_params(trial)

        model = CatBoostClassifier(**params, cat_features=cat_features)
        model.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=50)

        y_pred = model.predict(X_val)

        precision = precision_score(y_val, y_pred, zero_division=0)
        recall = recall_score(y_val, y_pred, zero_division=0)
        f1 = f1_score(y_val, y_pred, zero_division=0)

        # Log metrics to Optuna for tracking
        trial.set_user_attr("precision", precision)
        trial.set_user_attr("recall", recall)

        # Apply business constraints (Penalize trial if constraints not met)
        if precision < min_precision or recall < min_recall:
            raise optuna.TrialPruned(f"Constraints not met. P:{precision:.2f}, R:{recall:.2f}")

        return f1

    study = optuna.create_study(direction="maximize", study_name=study_name)
    logger.info(f"Starting tuning study: {study_name} | Trials: {n_trials}")
    study.optimize(objective, n_trials=n_trials)

    logger.info(f"Best Trial F1: {study.best_value}")
    return study.best_params, study
