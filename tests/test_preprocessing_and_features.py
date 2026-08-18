"""Preprocessing correctness and feature construction."""

from __future__ import annotations

import pandas as pd
import pytest

from pipelines.feature_pipeline import FeaturePipeline
from pipelines.preprocessing_pipeline import PreprocessingPipeline
from utils.feature_builders import FEATURE_VIEWS, build_view, engineer_features, split_by_segment
from utils.pipeline_errors import SchemaValidationFault


def test_blank_total_charges_imputed_not_dropped(local_settings, raw_frame):
    unbilled = int((raw_frame["tenure"] == 0).sum())
    assert unbilled > 0, "fixture should contain unbilled accounts"

    out = PreprocessingPipeline(local_settings).run()

    # Every row survives; the blanks become 0.0 rather than NaN.
    assert len(out) == len(raw_frame)
    assert out["TotalCharges"].isna().sum() == 0
    assert (out.loc[out["tenure"] == 0, "TotalCharges"] == 0.0).all()


def test_target_encoded_to_binary(local_settings):
    out = PreprocessingPipeline(local_settings).run()
    assert set(out["Churn"].unique()) <= {0, 1}
    assert out["Churn"].dtype.kind in "iu"


def test_unexpected_target_value_rejected(local_settings, raw_frame, tmp_path):
    raw_frame.loc[0, "Churn"] = "Maybe"
    raw_frame.to_csv(local_settings.data.raw_path, index=False)

    with pytest.raises(SchemaValidationFault) as excinfo:
        PreprocessingPipeline(local_settings).run()
    assert excinfo.value.code == "SYS-102"


def test_missing_mandatory_column_rejected(local_settings, raw_frame):
    raw_frame.drop(columns=["MonthlyCharges"]).to_csv(local_settings.data.raw_path, index=False)
    with pytest.raises(SchemaValidationFault):
        PreprocessingPipeline(local_settings).run()


def test_missing_raw_file_names_the_fix(local_settings):
    import os

    os.remove(local_settings.data.raw_path)
    with pytest.raises(SchemaValidationFault) as excinfo:
        PreprocessingPipeline(local_settings).run()
    assert "download_data" in str(excinfo.value)


def test_avg_monthly_spend_never_divides_by_zero():
    df = pd.DataFrame(
        {
            "tenure": [0, 1, 10],
            "TotalCharges": [0.0, 50.0, 500.0],
            "PhoneService": ["Yes", "No", "Yes"],
        }
    )
    out = engineer_features(df)
    assert out["avg_monthly_spend"].notna().all()
    assert (out["avg_monthly_spend"] >= 0).all()
    # tenure is clipped to a floor of 1, so a zero-tenure account is not inf.
    assert out.loc[0, "avg_monthly_spend"] == 0.0


def test_services_subscribed_ignores_negative_sentinels():
    df = pd.DataFrame(
        {
            "tenure": [1, 1],
            "TotalCharges": [10.0, 10.0],
            "PhoneService": ["Yes", "No"],
            "OnlineSecurity": ["Yes", "No internet service"],
            "StreamingTV": ["Yes", "No"],
        }
    )
    out = engineer_features(df)
    assert out.loc[0, "services_subscribed"] == 3
    # "No internet service" must count as absence, not as a distinct service.
    assert out.loc[1, "services_subscribed"] == 0


def test_segments_partition_without_overlap(local_settings):
    df = PreprocessingPipeline(local_settings).run()
    segments = split_by_segment(
        df, local_settings.segmentation.column, local_settings.segmentation.segments
    )
    total = sum(len(f) for f in segments.values())
    assert total == len(df), "segments must cover every row exactly once"

    ids = pd.concat([f["customerID"] for f in segments.values()])
    assert ids.duplicated().sum() == 0


def test_feature_views_are_disjoint_and_nonempty(local_settings):
    df = PreprocessingPipeline(local_settings).run()
    activity = set(build_view(df, "activity").columns)
    profile = set(build_view(df, "profile").columns)
    assert activity and profile
    assert activity.isdisjoint(profile)


def test_unknown_view_rejected(local_settings):
    df = PreprocessingPipeline(local_settings).run()
    with pytest.raises(KeyError):
        build_view(df, "nonsense")


def test_feature_pipeline_emits_one_entry_per_model(local_settings):
    df = PreprocessingPipeline(local_settings).run()
    views = FeaturePipeline(local_settings).run(df)

    assert views, "expected at least one trainable segment"
    for per_view in views.values():
        assert set(per_view) == set(FEATURE_VIEWS)
        for features, target in per_view.values():
            assert len(features) == len(target)
            assert not features.empty


def test_undersized_segments_are_skipped_not_trained(local_settings):
    df = PreprocessingPipeline(local_settings).run()
    local_settings.training.min_segment_rows = 10_000
    with pytest.raises(SchemaValidationFault):
        FeaturePipeline(local_settings).run(df)
