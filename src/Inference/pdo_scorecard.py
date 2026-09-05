"""
PDO (Points-to-Double-the-Odds) transform -- this is a direct port of
`pd_to_score` from Credit_Scoring.ipynb (section 3) plus the risk-band
assignment from section 5, so batch/API inference produces scores and
bands on the exact same scale as your scorecard analysis notebook.
"""
import numpy as np
import pandas as pd

import config


def proba_to_score(prob_default) -> np.ndarray:
    """
    Convert predicted probability of default to an integer credit score.
    Mirrors Credit_Scoring.ipynb's pd_to_score exactly, including the
    1e-10 epsilon and int rounding/clipping -- so a given PD always maps
    to the same score whether it went through this pipeline or the
    scorecard notebook.
    """
    factor = config.PDO / np.log(2)
    offset = config.BASE_SCORE - factor * np.log(config.BASE_ODDS)
    prob_default = np.asarray(prob_default, dtype=float)
    odds = (1 - prob_default) / (prob_default + 1e-10)
    score = offset + factor * np.log(odds)
    return np.clip(np.round(score).astype(int), config.MIN_SCORE, config.MAX_SCORE)


def score_to_risk_band(score) -> pd.Categorical:
    """A-E bands with fixed thresholds, matching Credit_Scoring.ipynb section 5."""
    return pd.cut(
        score,
        bins=config.BAND_BINS,
        labels=config.BAND_LABELS,
        right=False,
    )
