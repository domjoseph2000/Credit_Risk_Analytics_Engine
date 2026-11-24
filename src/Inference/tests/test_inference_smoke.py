"""
Quick tests of the inference plumbing that need no MLflow model and no real
data: the feature-list check, PD-to-score maths, risk band cut-offs, and
the real-time API's handling of empty fields.

Run from src/Inference with the project environment active (needs pytest):
    pytest tests
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# The modules under test live one folder up, in src/Inference
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils import config  # noqa: E402
from utils import pdo_scorecard  # noqa: E402
from features.pipeline import assemble_features  # noqa: E402


@pytest.fixture
def tiny_application_df():
    """Three made-up applicants with every column in features/application.py's REQUIRED_COLUMNS."""
    return pd.DataFrame(
        {
            "SK_ID_CURR": [1, 2, 3],
            "DAYS_BIRTH": [-12000, -15000, -9000],
            "DAYS_EMPLOYED": [-2000, -3000, 365243],  # 365243 is the sentinel -> NaN
            "DAYS_ID_PUBLISH": [-3000, -4000, -2000],
            "DAYS_LAST_PHONE_CHANGE": [-1134, -828, 0],
            "EXT_SOURCE_1": [0.5, 0.6, 0.7],
            "EXT_SOURCE_2": [0.4, 0.5, 0.6],
            "EXT_SOURCE_3": [0.3, 0.4, 0.5],
            "FLOORSMAX_MEDI": [0.1, 0.2, 0.3],
            "OWN_CAR_AGE": [5, 10, 15],
            "AMT_CREDIT": [300000, 400000, 250000],
            "AMT_GOODS_PRICE": [270000, 380000, 240000],
            "OCCUPATION_TYPE": ["Laborers", "Core staff", "Drivers"],
            "ORGANIZATION_TYPE": ["Business Entity Type 3", "School", "XNA"],
            "CODE_GENDER": ["M", "F", "XNA"],
            "NAME_EDUCATION_TYPE": ["Secondary / secondary special", "Higher education", "Incomplete higher"],
        }
    )


def _write_feature_list(tmp_path, features):
    """Write a throwaway selected_features.xlsx listing `features`; returns its path."""
    path = tmp_path / "selected_features.xlsx"
    pd.DataFrame({"feature": features}).to_excel(path, index=False)
    return path


def test_assemble_features_raises_on_missing_expected_columns(tmp_path, tiny_application_df):
    # A feature list with one real feature (AGE) and one that nothing makes:
    # assembling features must fail, naming the made-up one.
    feature_list_path = _write_feature_list(
        tmp_path, ["AGE", "TOTALLY_MADE_UP_FEATURE"]
    )
    config.SELECTED_FEATURES_PATH = feature_list_path

    with pytest.raises(ValueError, match="TOTALLY_MADE_UP_FEATURE"):
        assemble_features({"application": tiny_application_df})


def test_assemble_features_selects_only_expected_columns(tmp_path, tiny_application_df):
    # The application code makes more columns than these two; only the listed
    # features (plus SK_ID_CURR) should come out, in the listed order.
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
    # Odds of 50:1 should score exactly BASE_SCORE (600)
    base_proba = 1 / (1 + config.BASE_ODDS)  # the PD that gives those odds
    score = pdo_scorecard.proba_to_score(np.array([base_proba]))
    assert score[0] == config.BASE_SCORE


def test_pdo_transform_clips_to_min_max():
    # Extreme PDs must still give scores within 300–850
    score = pdo_scorecard.proba_to_score(np.array([0.9999999, 0.0000001]))
    assert score[0] >= config.MIN_SCORE
    assert score[1] <= config.MAX_SCORE


def test_risk_bands_match_scorecard_notebook_edges():
    # One score inside each band (cut-offs as in Credit_Scoring.ipynb section 5)
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
    # An empty optional field (None) must become a numeric blank (NaN). Left
    # as None, the column's type would be 'object', which the model rejects
    # -- this used to break any real-time request without e.g. OWN_CAR_AGE.
    # (Importing realtime_api needs fastapi, but no model or data.)
    import realtime_api

    raw = {"SK_ID_CURR": 1, "OWN_CAR_AGE": None, "AMT_CREDIT": 1000.0}
    sanitized = realtime_api._sanitize_optional_fields(raw)
    df = pd.DataFrame([sanitized])
    assert df["OWN_CAR_AGE"].dtype == np.float64
    assert np.isnan(df["OWN_CAR_AGE"].iloc[0])
