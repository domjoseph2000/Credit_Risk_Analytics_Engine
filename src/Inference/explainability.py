"""
Per-attribute points contributions for a scored applicant.

This is the same additive decomposition validated against
pdo_scorecard.py's PDO transform: proba_to_score() is a LINEAR function of
the model's raw margin (score = offset - factor * raw_margin), and SHAP
values are an EXACT additive decomposition of that same raw margin
(base_value + sum(shap_values) == raw_margin, to floating-point precision).
Substituting one into the other:

    score(x) = (offset - factor * base_value) + sum(-factor * shap_i(x))
             = reference_score                + sum(points_i(x))

So each attribute's points contribution is just `-factor * shap_i(x)`, and
they sum EXACTLY (verified empirically, ~1e-7 float noise) to that
applicant's score minus a constant reference_score (the score of a
"perfectly average" applicant -- same for everyone).

This module only reads from the already-trained model to produce an
explanation after the fact -- it never feeds back into score_applicants()
or proba_to_score(), so it cannot change anyone's actual score or band.
"""
import functools
import logging

import numpy as np
import pandas as pd
import shap

import config
from model import load_model

logger = logging.getLogger(__name__)


@functools.lru_cache(maxsize=1)
def load_explainer():
    """
    shap.TreeExplainer needs the actual tree ensemble, not the pyfunc
    wrapper's predict() callable -- unwrap_python_model() (an MLflow API,
    not a workaround) returns the LightGBMProductionWrapper instance
    logged in 02_training_LightGBM_v4.ipynb section 8, from which .trained_model
    is the raw sklearn LGBMClassifier and .categorical_features is the
    same list the wrapper itself uses to recast dtypes before predict().
    """
    pyfunc_model = load_model()
    wrapper = pyfunc_model.unwrap_python_model()
    explainer = shap.TreeExplainer(wrapper.trained_model)
    logger.info("Built TreeExplainer from the loaded model.")
    return explainer, wrapper.categorical_features


def explain_contributions(feature_df: pd.DataFrame) -> pd.DataFrame:
    """
    feature_df: same input as model.predict_proba() -- output of
    features.pipeline.assemble_features, including SK_ID_CURR.

    Returns one row per applicant: SK_ID_CURR, REFERENCE_SCORE (constant --
    see module docstring), and one POINTS_<feature> column per model
    feature. REFERENCE_SCORE + sum of the POINTS_ columns reconstructs
    proba_to_score()'s output exactly (before its final round/clip).
    """
    explainer, categorical_features = load_explainer()

    ids = feature_df["SK_ID_CURR"]
    X = feature_df.drop(columns=["SK_ID_CURR"]).copy()
    for col in categorical_features:
        if col in X.columns:
            X[col] = X[col].astype("category")

    explanation = explainer(X)

    factor = config.PDO / np.log(2)
    offset = config.BASE_SCORE - factor * np.log(config.BASE_ODDS)

    points = -factor * explanation.values
    reference_score = offset - factor * explanation.base_values

    points_df = pd.DataFrame(points, columns=[f"POINTS_{c}" for c in X.columns], index=X.index)
    points_df.insert(0, "REFERENCE_SCORE", reference_score)
    points_df.insert(0, "SK_ID_CURR", ids.values)
    return points_df.reset_index(drop=True)


def top_reason_codes(points_row: pd.Series, n: int = 5) -> list[dict]:
    """
    Reduce one applicant's full POINTS_ breakdown (a row of
    explain_contributions()'s output) to the top-n features by |points| --
    for the real-time API response, where returning all ~24 columns as
    JSON would be noise. Batch output keeps the full breakdown since it's
    a file, not a payload -- see score_model_input.py/batch_feature_tasks.py.
    """
    points = {
        col[len("POINTS_"):]: value
        for col, value in points_row.items()
        if col.startswith("POINTS_")
    }
    ranked = sorted(points.items(), key=lambda kv: -abs(kv[1]))[:n]
    return [{"feature": feat, "points": round(float(pts), 2)} for feat, pts in ranked]
