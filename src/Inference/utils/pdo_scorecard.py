"""
Turns a probability of default (PD) into a credit score and a risk band.

The same formula and band cut-offs as Credit_Scoring.ipynb (pd_to_score in
section 3, bands in section 5), so batch and real-time scores sit on exactly
the same scale as the scorecard analysis. The settings are in config.py.
"""
import numpy as np
import pandas as pd

from utils import config


def proba_to_score(prob_default) -> np.ndarray:
    """
    PD -> whole-number credit score between 300 and 850 (higher = safer).

        odds   = (1 - PD) / PD               (good:bad odds)
        factor = PDO / ln(2)                 (points per doubling of the odds)
        offset = BASE_SCORE - factor × ln(BASE_ODDS)
        score  = offset + factor × ln(odds)

    With the current settings, odds of 50:1 score 600 and every doubling of
    the odds adds 20 points. Matches Credit_Scoring.ipynb's pd_to_score
    exactly, including the tiny 1e-10 that avoids dividing by zero and the
    rounding and capping, so a given PD always gets the same score.
    """
    factor = config.PDO / np.log(2)
    offset = config.BASE_SCORE - factor * np.log(config.BASE_ODDS)
    prob_default = np.asarray(prob_default, dtype=float)
    odds = (1 - prob_default) / (prob_default + 1e-10)
    score = offset + factor * np.log(odds)
    return np.clip(np.round(score).astype(int), config.MIN_SCORE, config.MAX_SCORE)


def score_to_risk_band(score) -> pd.Categorical:
    """Score -> risk band A (very low risk) to E (very high risk), using the fixed cut-offs in config.BAND_BINS."""
    return pd.cut(
        score,
        bins=config.BAND_BINS,
        labels=config.BAND_LABELS,
        right=False,   # each band includes its lower cut-off, e.g. 620 is band B
    )
