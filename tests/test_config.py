"""Configuration loading and segment routing."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from utils.pipeline_errors import ConfigurationFault, MLSystemFault
from utils.settings_manager import SettingsManager


def test_loads_typed_settings(settings):
    assert settings.project_name
    assert settings.schema.target_variable == "Churn"
    assert settings.data.sha256, "dataset checksum must be pinned"
    assert set(settings.segmentation.segments) == {"power_user", "casual", "guest"}


def test_catboost_params_exclude_fit_only_arguments(settings):
    params = settings.training.catboost.as_params()
    # early_stopping_rounds belongs to fit(), not the constructor; passing it
    # to CatBoostClassifier raises.
    assert "early_stopping_rounds" not in params
    assert params["iterations"] > 0


@pytest.mark.parametrize(
    ("tenure", "expected"),
    [(0, "guest"), (4, "guest"), (5, "casual"), (24, "casual"), (25, "power_user"), (72, "power_user")],
)
def test_segment_boundaries(settings, tenure, expected):
    assert settings.segment_of(tenure) == expected


def test_missing_config_raises_registered_code(tmp_path):
    with pytest.raises(ConfigurationFault) as excinfo:
        SettingsManager(str(tmp_path / "nope.json")).load()
    assert excinfo.value.code == "SYS-201"


def test_empty_segments_rejected(tmp_path, settings):
    config = json.loads(Path("configs/config.json").read_text(encoding="utf-8"))
    config["segmentation"]["segments"] = {}
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(ConfigurationFault):
        SettingsManager(str(path)).load()


def test_errors_serialize_for_structured_logs():
    fault = MLSystemFault("SYS-301", "missing artifact")
    payload = fault.serialize()
    assert payload["error_code"] == "SYS-301"
    assert payload["context"] == "missing artifact"
    assert "registry" in payload["message"].lower()
