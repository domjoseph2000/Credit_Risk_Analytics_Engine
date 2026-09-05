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
    return pd.DataFrame(
        {
            "SK_ID_CURR": [1, 2, 3],
            "AMT_INCOME_TOTAL": [100000, 200000, 150000],
            "AMT_CREDIT": [300000, 400000, 250000],
            "AMT_ANNUITY": [15000, 20000, 12500],
            "DAYS_BIRTH": [-12000, -15000, -9000],
            "DAYS_EMPLOYED": [-2000, -3000, 365243],
        }
    )


def _write_feature_list(tmp_path, features):
    path = tmp_path / "selected_features.xlsx"
    pd.DataFrame({"feature": features}).to_excel(path, index=False)
    return path


def test_assemble_features_raises_on_missing_expected_columns(tmp_path, tiny_application_df):
    feature_list_path = _write_feature_list(
        tmp_path, ["CREDIT_INCOME_RATIO", "TOTALLY_MADE_UP_FEATURE"]
    )
    config.SELECTED_FEATURES_PATH = feature_list_path

    with pytest.raises(ValueError, match="TOTALLY_MADE_UP_FEATURE"):
        assemble_features({"application": tiny_application_df})


def test_assemble_features_selects_only_expected_columns(tmp_path, tiny_application_df):
    feature_list_path = _write_feature_list(
        tmp_path, ["CREDIT_INCOME_RATIO", "ANNUITY_INCOME_RATIO"]
    )
    config.SELECTED_FEATURES_PATH = feature_list_path

    result = assemble_features({"application": tiny_application_df})

    assert list(result.columns) == ["SK_ID_CURR", "CREDIT_INCOME_RATIO", "ANNUITY_INCOME_RATIO"]
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
