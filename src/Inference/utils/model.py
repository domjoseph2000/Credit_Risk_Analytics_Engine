"""
Loads the @prod model from MLflow and predicts probability of default (PD).

The registered model is a small wrapper around the trained LightGBM model
(LightGBMProductionWrapper, defined in section 8 of
02_training_LightGBM_v4.ipynb). The wrapper turns text columns such as
ORGANIZATION_TYPE into the 'category' type the model expects, so callers can
pass plain values. The model is loaded once and then reused.
"""
import functools
import logging
import mlflow
import pandas as pd

from utils import config

logger = logging.getLogger(__name__)

# MLflow's names for column types -> the matching pandas types. The model
# records each column's type from training, and MLflow refuses to convert a
# column to a type it can't prove is safe (e.g. int16 -> float32).
_SCHEMA_DTYPES = {"integer": "int32", "long": "int64", "float": "float32", "double": "float64"}


@functools.lru_cache(maxsize=1)
def load_model():
    """Load models:/<MODEL_NAME>@prod once; later calls reuse the same object."""
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    # Loaded by alias, so promoting a new version in the MLflow UI is all it
    # takes to switch models (restart the real-time API to pick it up)
    model_uri = f"models:/{config.MODEL_NAME}@{config.MODEL_ALIAS}"
    logger.info("Loading model from %s", model_uri)
    return mlflow.pyfunc.load_model(model_uri)


def match_input_schema(X: pd.DataFrame, model) -> pd.DataFrame:
    """
    Convert each numeric column to the type the model was trained with.

    Why: the feature code shrinks each column to the smallest type that fits
    the values in *this* batch. A column with no blanks in this batch (e.g.
    one real-time applicant) can come out as a whole-number type where
    training had decimals, and MLflow then rejects the whole request with
    "Can not safely convert int16 to float32".

    Whole-number columns that contain blanks are left alone, so a genuinely
    missing required value still fails loudly instead of being made up here.
    """
    schema = model.metadata.get_input_schema()
    if schema is None or not schema.has_input_names():
        return X
    X = X.copy()
    for spec in schema.inputs:
        target = _SCHEMA_DTYPES.get(getattr(spec.type, "name", ""))
        if target is None or spec.name not in X.columns or X[spec.name].dtype == target:
            continue
        if target.startswith("int") and X[spec.name].isna().any():
            continue
        X[spec.name] = X[spec.name].astype(target)
    return X


def predict_proba(feature_df: pd.DataFrame) -> pd.Series:
    """
    Predict PD for every row.

    feature_df: the output of features.pipeline.assemble_features, including
    the SK_ID_CURR column (it's removed before predicting).

    Returns a Series of probabilities indexed by SK_ID_CURR, in the same order
    as the input. (The wrapper's predict() returns a plain list of P(default),
    one per row.)
    """
    model = load_model()
    ids = feature_df["SK_ID_CURR"]
    X = match_input_schema(feature_df.drop(columns=["SK_ID_CURR"]), model)

    preds = model.predict(X)
    proba = pd.Series(preds, index=ids.values, name="PROBA_DEFAULT")
    return proba
