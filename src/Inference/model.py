"""
Loads the registered LightGBM pyfunc model (LightGBMProductionWrapper,
see 02_training_LightGBM_v3.ipynb section 8) once and reuses it.

Note on categorical columns: the wrapper re-casts its stored
`categorical_features` list to pandas 'category' dtype internally on
every call, so this module doesn't need to do that -- just make sure
the feature DataFrame has the right column names with plain values
(strings are fine) and the wrapper handles the rest.
"""
import functools
import logging
import mlflow
import pandas as pd

import config

logger = logging.getLogger(__name__)


@functools.lru_cache(maxsize=1)
def load_model():
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    # Alias-based, not stage-based: "prod" is assigned manually in the
    # MLflow UI after reviewing a run -- see the notebook's section 8 note.
    model_uri = f"models:/{config.MODEL_NAME}@{config.MODEL_ALIAS}"
    logger.info("Loading model from %s", model_uri)
    return mlflow.pyfunc.load_model(model_uri)


def predict_proba(feature_df: pd.DataFrame) -> pd.Series:
    """
    feature_df: output of features.pipeline.assemble_features, INCLUDING
    SK_ID_CURR (dropped here, not passed to the model).

    LightGBMProductionWrapper.predict() always returns
    `probabilities[:, 1].tolist()` -- a plain list of P(default), one per
    row, in the same order as the input rows. No column-count branching
    needed; that's the actual, known contract, not a guess.
    """
    model = load_model()
    ids = feature_df["SK_ID_CURR"]
    X = feature_df.drop(columns=["SK_ID_CURR"])

    preds = model.predict(X)
    proba = pd.Series(preds, index=ids.values, name="PROBA_DEFAULT")
    return proba
