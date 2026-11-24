"""
Builds the feature table the model expects, from the raw data tables.

This is the one place where raw data becomes model input: batch scoring
(both the Airflow DAG and score_model_input.py), the real-time API and
Batch_Inference.ipynb all go through it. That's what keeps them consistent.

If inference ever calculated a feature differently from training, the model
would quietly give wrong scores rather than an error. So merge_feature_frames()
stops with an error if any feature the model expects is missing, and
verify_feature_parity.py checks the values against the training data.

The feature list is the @prod model's (MLflow version 23, see
data/Model-Training/HyperParameters/LightGBM_v4/selected_features.json).
The feature modules also still make a few columns for the previous @prod
(version 11); only the current model's columns are kept.
"""
import logging
from pathlib import Path
import pandas as pd

from utils import config
from . import (
    application,
    bureau,
    credit_card,
    installments,
    pos_cash,
    previous_application,
)

logger = logging.getLogger(__name__)

# The raw columns each table's feature code reads. Code that loads the CSVs
# can pass usecols=RAW_COLUMN_REQUIREMENTS[table] to read only these columns
# -- most of the multi-million-row history files' columns are never used.
RAW_COLUMN_REQUIREMENTS: dict[str, list[str]] = {
    "application": application.REQUIRED_COLUMNS,
    "bureau": bureau.REQUIRED_COLUMNS,
    "previous_application": previous_application.REQUIRED_COLUMNS,
    "pos_cash": pos_cash.REQUIRED_COLUMNS,
    "installments": installments.REQUIRED_COLUMNS,
    "credit_card": credit_card.REQUIRED_COLUMNS,
}


def load_expected_feature_list(path: Path | None = None) -> list[str]:
    """
    The @prod model's features, in training order, from the 'feature' column
    of selected_features.xlsx (config.SELECTED_FEATURES_PATH). When you
    promote a new model, replace that list with the new version's
    HyperParameters/<model_folder>/selected_features.json.
    """
    path = path or config.SELECTED_FEATURES_PATH
    features_df = pd.read_excel(path)
    return features_df["feature"].tolist()


def merge_feature_frames(app_df: pd.DataFrame, feature_frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Join the per-table features into the model's input table.

    app_df: output of application.build_features() -- one row per applicant.
    feature_frames: some or all of "bureau", "previous_application",
        "pos_cash", "installments", "credit_card", each the output of that
        table's build_features() (one row per applicant).

    Joins everything on SK_ID_CURR (applicants with no history in a table get
    blanks), keeps exactly the model's features in training order, and
    converts number types to match training. Returns SK_ID_CURR + the features.

    Both assemble_features() below and the Airflow DAG's join task (which
    builds each table in its own task) end here, so they can't produce
    different feature tables.
    """
    merged = app_df
    for key in ("bureau", "previous_application", "pos_cash", "installments", "credit_card"):
        if key in feature_frames:
            merged = merged.merge(feature_frames[key], on="SK_ID_CURR", how="left")

    # Stop with a clear error if any feature the model needs wasn't produced
    expected = load_expected_feature_list()
    missing = [c for c in expected if c not in merged.columns]
    if missing:
        raise ValueError(
            f"{len(missing)} features the model expects are missing after "
            f"feature assembly (e.g. {missing[:10]}). This means either a raw "
            f"table wasn't supplied, or a feature module's logic hasn't been "
            f"updated to match the currently-promoted model's feature list."
        )

    # Keep only the model's features, in the exact order used in training
    id_col = merged[["SK_ID_CURR"]]
    feature_matrix = merged[expected].copy()

    # Shrink number types the way model_dataset_creation.ipynb did before
    # training (decimals to float32, whole numbers to the smallest type that
    # fits). The registered model records the training types, and MLflow
    # rejects input whose types don't match. (model.match_input_schema then
    # fixes any column whose type depends on this batch's values.)
    float_cols = feature_matrix.select_dtypes(include="float64").columns
    int_cols = feature_matrix.select_dtypes(include="int64").columns
    if len(float_cols) > 0:
        feature_matrix[float_cols] = feature_matrix[float_cols].astype("float32")
    if len(int_cols) > 0:
        feature_matrix[int_cols] = feature_matrix[int_cols].apply(pd.to_numeric, downcast="integer")

    return pd.concat([id_col, feature_matrix], axis=1)


def assemble_features(raw_tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Raw tables in, model input out, in one call. Used by the real-time API,
    score_model_input.py and Batch_Inference.ipynb.

    raw_tables: the raw data, keyed by table name -- "application" (required)
        plus any of "bureau", "previous_application", "pos_cash",
        "installments", "credit_card".

    A missing history table is fine: those features are just left blank, and
    LightGBM handles blanks itself, as it did in training. (pos_cash also
    needs "previous_application".)
    """
    app_df = application.build_features(raw_tables["application"])

    # The history tables cover everyone Home Credit has ever seen, so each
    # feature function first cuts its table down to the applicants being
    # scored. Taken from the raw application table, where SK_ID_CURR is still
    # a number (app_df has already turned it into text).
    valid_ids = set(raw_tables["application"]["SK_ID_CURR"])

    feature_frames: dict[str, pd.DataFrame] = {}
    if "bureau" in raw_tables:
        feature_frames["bureau"] = bureau.build_features(raw_tables["bureau"], valid_ids)

    if "previous_application" in raw_tables:
        feature_frames["previous_application"] = previous_application.build_features(
            raw_tables["previous_application"], valid_ids
        )

    if "pos_cash" in raw_tables and "previous_application" in raw_tables:
        feature_frames["pos_cash"] = pos_cash.build_features(
            raw_tables["pos_cash"], raw_tables["previous_application"], valid_ids
        )

    if "installments" in raw_tables:
        feature_frames["installments"] = installments.build_features(
            raw_tables["installments"], valid_ids
        )

    if "credit_card" in raw_tables:
        feature_frames["credit_card"] = credit_card.build_features(raw_tables["credit_card"], valid_ids)

    return merge_feature_frames(app_df, feature_frames)
