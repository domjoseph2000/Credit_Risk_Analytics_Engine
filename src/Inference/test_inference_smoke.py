"""
Smoke tests that don't require a real MLflow model or real Home Credit
data -- just checks the plumbing (joins, feature contract enforcement,
PDO math) works. Run with: pytest tests/
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config  # noqa: E402
import pdo_scorecard  # noqa: E402
from features.pipeline import assemble_features  # noqa: E402


@pytest.fixture
def tiny_application_df():
    # Columns match features/application.py::REQUIRED_COLUMNS (the v4-era
    # feature set) -- not just whatever build_features happens to touch, so
    # this stays valid as long as REQUIRED_COLUMNS does.
    return pd.DataFrame(
        {
            "SK_ID_CURR": [1, 2, 3],
            "DAYS_BIRTH": [-12000, -15000, -9000],
            "DAYS_EMPLOYED": [-2000, -3000, 365243],  # 365243 is the sentinel -> NaN
            "DAYS_ID_PUBLISH": [-3000, -4000, -2000],
            "EXT_SOURCE_1": [0.5, 0.6, 0.7],
            "EXT_SOURCE_2": [0.4, 0.5, 0.6],
            "EXT_SOURCE_3": [0.3, 0.4, 0.5],
            "FLOORSMAX_MEDI": [0.1, 0.2, 0.3],
            "OWN_CAR_AGE": [5, 10, 15],
            "AMT_CREDIT": [300000, 400000, 250000],
            "AMT_GOODS_PRICE": [270000, 380000, 240000],
            "OCCUPATION_TYPE": ["Laborers", "Core staff", "Drivers"],
            "ORGANIZATION_TYPE": ["Business Entity Type 3", "School", "XNA"],
        }
    )


def _write_feature_list(tmp_path, features):
    path = tmp_path / "selected_features.xlsx"
    pd.DataFrame({"feature": features}).to_excel(path, index=False)
    return path


def test_assemble_features_raises_on_missing_expected_columns(tmp_path, tiny_application_df):
    # AGE is a real column features/application.py produces; the second
    # name is not -- assemble_features must complain about that one, not AGE.
    feature_list_path = _write_feature_list(
        tmp_path, ["AGE", "TOTALLY_MADE_UP_FEATURE"]
    )
    config.SELECTED_FEATURES_PATH = feature_list_path

    with pytest.raises(ValueError, match="TOTALLY_MADE_UP_FEATURE"):
        assemble_features({"application": tiny_application_df})


def test_assemble_features_selects_only_expected_columns(tmp_path, tiny_application_df):
    # Both are real columns features/application.py::build_features produces
    # (see its final_attributes list) -- AGE is a passthrough-derived ratio
    # off DAYS_BIRTH, GOODSPRICE_TO_AMT_CREDIT off AMT_GOODS_PRICE/AMT_CREDIT.
    feature_list_path = _write_feature_list(
        tmp_path, ["AGE", "GOODSPRICE_TO_AMT_CREDIT"]
    )
    config.SELECTED_FEATURES_PATH = feature_list_path

    result = assemble_features({"application": tiny_application_df})

    assert list(result.columns) == ["SK_ID_CURR", "AGE", "GOODSPRICE_TO_AMT_CREDIT"]
    assert len(result) == 3


def test_pdo_transform_monotonic_decreasing_in_default_proba():
    proba = np.array([0.01, 0.05, 0.20, 0.50])
    score = pdo_scorecard.proba_to_score(proba)
    assert (np.diff(score) < 0).all()  # higher default risk -> lower score


def test_pdo_transform_hits_base_score_at_base_odds():
    base_proba = 1 / (1 + config.BASE_ODDS)  # P(default) implied by base odds
    score = pdo_scorecard.proba_to_score(np.array([base_proba]))
    assert score[0] == config.BASE_SCORE


def test_pdo_transform_clips_to_min_max():
    score = pdo_scorecard.proba_to_score(np.array([0.9999999, 0.0000001]))
    assert score[0] >= config.MIN_SCORE
    assert score[1] <= config.MAX_SCORE


def test_risk_bands_match_scorecard_notebook_edges():
    # Scores chosen to land one per band per Credit_Scoring.ipynb section 5
    scores = np.array([550, 570, 600, 630, 660])
    bands = pdo_scorecard.score_to_risk_band(scores)
    assert list(bands) == [
        "E — Very High Risk",
        "D — High Risk",
        "C — Medium Risk",
        "B — Low Risk",
        "A — Very Low Risk",
    ]


def test_sanitize_optional_fields_gives_numeric_dtype_not_object():
    # A single-row DataFrame built straight from a dict with a None value
    # gets dtype "object" for that column (nothing else to infer a numeric
    # dtype from), which LightGBM's predict() rejects -- realtime_api.py hit
    # this for any request leaving an optional field like OWN_CAR_AGE unset.
    import realtime_api

    raw = {"SK_ID_CURR": 1, "OWN_CAR_AGE": None, "AMT_CREDIT": 1000.0}
    sanitized = realtime_api._sanitize_optional_fields(raw)
    df = pd.DataFrame([sanitized])
    assert df["OWN_CAR_AGE"].dtype == np.float64
    assert np.isnan(df["OWN_CAR_AGE"].iloc[0])
