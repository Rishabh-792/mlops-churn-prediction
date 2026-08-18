"""Shared fixtures.

Tests run entirely on a synthetic frame with the same schema as the real
dataset, so the suite needs no network, no download, and no trained model.
The one test that touches real data is marked ``integration`` and skips
itself when the dataset is absent.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from utils.settings_manager import SettingsManager

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session")
def settings():
    return SettingsManager(str(REPO_ROOT / "configs" / "config.json")).load()


@pytest.fixture
def raw_frame() -> pd.DataFrame:
    """A synthetic frame matching the raw dataset's schema and quirks.

    Deliberately includes the two defects the real file has: TotalCharges as
    text, and blank strings for accounts that have not been billed yet.
    """
    rng = np.random.default_rng(42)
    n = 400
    tenure = rng.integers(0, 72, n)
    monthly = rng.uniform(20, 120, n).round(2)
    total = (tenure * monthly).round(2).astype(object)
    total[tenure == 0] = " "  # unbilled accounts arrive blank

    def pick(options):
        return rng.choice(options, n)

    return pd.DataFrame(
        {
            "customerID": [f"{i:04d}-TEST" for i in range(n)],
            "gender": pick(["Male", "Female"]),
            "SeniorCitizen": rng.integers(0, 2, n),
            "Partner": pick(["Yes", "No"]),
            "Dependents": pick(["Yes", "No"]),
            "tenure": tenure,
            "PhoneService": pick(["Yes", "No"]),
            "MultipleLines": pick(["Yes", "No", "No phone service"]),
            "InternetService": pick(["DSL", "Fiber optic", "No"]),
            "OnlineSecurity": pick(["Yes", "No", "No internet service"]),
            "OnlineBackup": pick(["Yes", "No", "No internet service"]),
            "DeviceProtection": pick(["Yes", "No", "No internet service"]),
            "TechSupport": pick(["Yes", "No", "No internet service"]),
            "StreamingTV": pick(["Yes", "No", "No internet service"]),
            "StreamingMovies": pick(["Yes", "No", "No internet service"]),
            "Contract": pick(["Month-to-month", "One year", "Two year"]),
            "PaperlessBilling": pick(["Yes", "No"]),
            "PaymentMethod": pick(["Electronic check", "Mailed check", "Bank transfer (automatic)"]),
            "MonthlyCharges": monthly,
            "TotalCharges": total,
            "Churn": pick(["Yes", "No"]),
        }
    )


@pytest.fixture
def local_settings(tmp_path, settings, raw_frame):
    """Settings pointed at a temp directory, with the synthetic frame written out."""
    raw = tmp_path / "raw.csv"
    raw_frame.to_csv(raw, index=False)

    config = json.loads((REPO_ROOT / "configs" / "config.json").read_text(encoding="utf-8"))
    config["data"]["raw_path"] = str(raw)
    config["data"]["processed_path"] = str(tmp_path / "clean.parquet")
    config["training"]["min_segment_rows"] = 50
    config["training"]["catboost"]["iterations"] = 30
    config["tracking"]["tracking_uri"] = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    config["outputs"]["model_dir"] = str(tmp_path / "models")
    config["outputs"]["metrics_path"] = str(tmp_path / "metrics.json")
    config["outputs"]["predictions_path"] = str(tmp_path / "predictions.csv")

    path = tmp_path / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    return SettingsManager(str(path)).load()
