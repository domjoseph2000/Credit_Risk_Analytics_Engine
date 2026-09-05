"""
Single entrypoint that builds the exact feature matrix the model expects.

This is deliberately the ONE place where raw tables become model input,
called identically by batch scoring, the real-time API, and (should you
choose) your training notebooks. If this drifts from what training used,
you get silent skew, not an error -- so `assemble_features` fails loudly
if the VIF-reduced feature list doesn't match what's available.
"""
import logging
from pathlib import Path
import pandas as pd

import config
from . import (
    application,
    bureau,
    installments,
    pos_cash,
    previous_application,
)

logger = logging.getLogger(__name__)

# Which raw columns each table actually needs, keyed the same way as
# raw_tables/assemble_features. Lets any CSV loader (batch script, notebook,
# Airflow task) pass usecols=RAW_COLUMN_REQUIREMENTS[key] instead of reading
# full tables -- bureau/previous_application/installments are multi-million
# row files where most columns are never touched by feature engineering.
# credit_card and bureau_balance are omitted: neither is currently wired
# into assemble_features below (credit_card.py is a TODO stub).

RAW_COLUMN_REQUIREMENTS: dict[str, list[str]] = {
    "application": application.REQUIRED_COLUMNS,
    "bureau": bureau.REQUIRED_COLUMNS,
    "previous_application": previous_application.REQUIRED_COLUMNS,
    "pos_cash": pos_cash.REQUIRED_COLUMNS,
    "installments": installments.REQUIRED_COLUMNS,
}


def load_expected_feature_list(path: Path | None = None) -> list[str]:
    """
    Reads the same file Model_LightGBM_v3.ipynb reads: an xlsx with a
    'feature' column, already produced by your upstream feature-selection
    step. This is not a separate export you need to create -- it already
    exists once that notebook has been run.
    """
    path = path or config.SELECTED_FEATURES_PATH
    features_df = pd.read_excel(path)
    return features_df["feature"].tolist()


def merge_feature_frames(app_df: pd.DataFrame, feature_frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    app_df: output of application.build_features().
    feature_frames: dict with keys among "bureau", "previous_application",
        "pos_cash", "installments", each the output of that table's own
        build_features() (already SK_ID_CURR-deduplicated, one row per
        applicant).

    Joins everything on SK_ID_CURR, validates the result against the
    model's expected feature list, and downcasts dtypes to match training.
    This is the one place assemble_features() (single raw_tables dict,
    single process) and the Airflow DAG's per-table tasks (each table
    built in its own task, handed here as already-built parquets) both
    funnel through -- so a task-per-table pipeline can't silently compute
    a different feature matrix than the single-process path.
    """
    merged = app_df
    for key in ("bureau", "previous_application", "pos_cash", "installments"):
        if key in feature_frames:
            merged = merged.merge(feature_frames[key], on="SK_ID_CURR", how="left")

    expected = load_expected_feature_list()
    missing = [c for c in expected if c not in merged.columns]
    if missing:
        raise ValueError(
            f"{len(missing)} features the model expects are missing after "
            f"feature assembly (e.g. {missing[:10]}). This means either a raw "
            f"table wasn't supplied, or a feature module's TODO hasn't been "
            f"filled in yet with the notebook logic that produces it."
        )

    # Reindex to the exact training column order -- LightGBM is positional,
    # not name-aware, when fed a plain numpy array, and even with a DataFrame
    # mismatched dtypes/order across categoricals can bite you.
    id_col = merged[["SK_ID_CURR"]]
    feature_matrix = merged[expected].copy()

    # model_dataset_creation.ipynb downcasts float64->float32 and
    # int64->smallest-safe-int before training (see downcast_numeric there),
    # so the registered model's MLflow signature expects those narrower
    # dtypes. mlflow.pyfunc enforces the signature strictly and refuses to
    # silently narrow float64->float32 itself, so this has to match on the
    # way in or every prediction call raises SCHEMA_ENFORCEMENT_FAILED.
    float_cols = feature_matrix.select_dtypes(include="float64").columns
    int_cols = feature_matrix.select_dtypes(include="int64").columns
    if len(float_cols) > 0:
        feature_matrix[float_cols] = feature_matrix[float_cols].astype("float32")
    if len(int_cols) > 0:
        feature_matrix[int_cols] = feature_matrix[int_cols].apply(pd.to_numeric, downcast="integer")

    return pd.concat([id_col, feature_matrix], axis=1)


def assemble_features(raw_tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Input is raw CSVs. It calls each table's own build_features()
    itself. Used by realtime_api.py, Batch_Inference.ipynb

    raw_tables: dict with keys among
        "application", "bureau", "bureau_balance", "previous_application",
        "pos_cash", "credit_card", "installments"

    Missing optional tables are fine -- their aggregates just won't be joined,
    and any resulting NaNs are handled the same way LightGBM handled them in
    training (it natively supports missing values, so no imputation needed
    unless your notebook did something custom).
    """
    app_df = application.build_features(raw_tables["application"])

    # Every other raw table is Home Credit's full history for every
    # applicant it's ever seen, not just this batch (bureau.csv/
    # previous_application.csv/installments_payments.csv are multi-million
    # row files) -- filter each to this batch's SK_ID_CURR values before
    # aggregating, or feature engineering burns time/memory on IDs the
    # merge in merge_feature_frames would drop anyway. Taken from the raw
    # application table (still its original int64 dtype), not app_df,
    # since app_df's SK_ID_CURR has already been cast to string.
    valid_ids = set(raw_tables["application"]["SK_ID_CURR"])

    feature_frames: dict[str, pd.DataFrame] = {}
    if "bureau" in raw_tables:
        feature_frames["bureau"] = bureau.build_features(raw_tables["bureau"], valid_ids)

    if "previous_application" in raw_tables:
        feature_frames["previous_application"] = previous_application.build_features(
            raw_tables["previous_application"], valid_ids
        )

    if "pos_cash" in raw_tables:
        feature_frames["pos_cash"] = pos_cash.build_features(raw_tables["pos_cash"], valid_ids)

    # if "credit_card" in raw_tables:
    #     feature_frames["credit_card"] = credit_card.build_features(raw_tables["credit_card"], valid_ids)

    if "installments" in raw_tables:
        feature_frames["installments"] = installments.build_features(
            raw_tables["installments"], raw_tables["previous_application"], valid_ids
        )

    return merge_feature_frames(app_df, feature_frames)
