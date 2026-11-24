"""
Reason codes: how many credit-score points each feature added or removed for
an applicant.

How it works
------------
SHAP splits the model's raw output for one applicant into one contribution
per feature, and those contributions add up exactly:

    raw output = base value + sum of the feature contributions

The credit score (pdo_scorecard.proba_to_score) is a straight-line function
of that same raw output: score = offset - factor × raw output. Putting the
two together:

    score = (offset - factor × base value) + sum(-factor × contribution)
          =        REFERENCE_SCORE         + sum(points per feature)

So each feature's points are just -factor × its SHAP value, and they add up
(to within ~1e-7) to the applicant's score minus REFERENCE_SCORE -- the score
of a "perfectly average" applicant, the same for everyone. Positive points
raised the score (lower risk); negative points lowered it.

This only reads the trained model after scoring; it never changes anyone's
score or band.

Full explanation with a worked example: src/Inference/README.md, section
"Reason codes — how a feature's points are derived".
"""
import functools
import logging

import numpy as np
import pandas as pd
import shap

from utils import config
from utils.model import load_model

logger = logging.getLogger(__name__)


@functools.lru_cache(maxsize=1)
def load_explainer():
    """
    Build the SHAP explainer once and reuse it.

    SHAP needs the LightGBM model itself, not MLflow's wrapper around it, so
    this unwraps the registered LightGBMProductionWrapper (see section 8 of
    02_training_LightGBM_v4.ipynb): .trained_model is the LightGBM classifier
    and .categorical_features lists its text columns.
    """
    pyfunc_model = load_model()
    wrapper = pyfunc_model.unwrap_python_model()
    explainer = shap.TreeExplainer(wrapper.trained_model)
    logger.info("Built TreeExplainer from the loaded model.")
    return explainer, wrapper.categorical_features


def explain_contributions(feature_df: pd.DataFrame) -> pd.DataFrame:
    """
    Points per feature for every applicant.

    feature_df: the same input as model.predict_proba() -- the output of
    features.pipeline.assemble_features, including SK_ID_CURR.

    Returns one row per applicant with SK_ID_CURR, REFERENCE_SCORE (the same
    for everyone) and one POINTS_<feature> column per model feature.
    REFERENCE_SCORE + the sum of the POINTS_ columns equals the applicant's
    score before it's rounded and capped to 300–850.
    """
    explainer, categorical_features = load_explainer()

    ids = feature_df["SK_ID_CURR"]
    X = feature_df.drop(columns=["SK_ID_CURR"]).copy()
    # The model needs its text columns as the 'category' type, as the wrapper does before predicting
    for col in categorical_features:
        if col in X.columns:
            X[col] = X[col].astype("category")

    explanation = explainer(X)

    # Same factor and offset as pdo_scorecard.proba_to_score
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
    The n features that moved one applicant's score the most (up or down).

    points_row: one row of explain_contributions()'s output. Used by the
    real-time API, which returns just the top few; batch scoring keeps every
    feature's points in reason_codes.parquet.
    Returns e.g. [{"feature": "EXT_SOURCE_3", "points": -24.1}, ...].
    """
    points = {
        col[len("POINTS_"):]: value
        for col, value in points_row.items()
        if col.startswith("POINTS_")
    }
    ranked = sorted(points.items(), key=lambda kv: -abs(kv[1]))[:n]
    return [{"feature": feat, "points": round(float(pts), 2)} for feat, pts in ranked]
