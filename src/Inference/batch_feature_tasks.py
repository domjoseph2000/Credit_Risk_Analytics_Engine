"""
Per-table task callables for the Airflow batch-inference DAG
(dags/inference_batch_dag.py).

Splits score_model_input.py's single sequential "read everything, build
everything, score" run into one task per raw table -- application,
bureau, previous_application, pos_cash, installments, credit_card -- so
they run in parallel and fail/retry independently in the Airflow UI,
mirroring 01_preprocessing_dag.py's per-table task pattern. join_features
then merges all six, and score_features scores the result -- kept as
two separate tasks (not one) so a scoring failure can be retried without
redoing the join, and so the assembled feature matrix is itself an
inspectable artifact.

None of the six build_*_features functions depend on each other: each
reads the application table's SK_ID_CURR column itself (a cheap,
single-column read) to filter its own table to this batch, instead of
waiting on a shared "application" task to hand off an ID list. Only
join_features_and_score depends on all six having finished.

Intermediate per-table feature parquets are written to
DATA_ROOT/Inference/BatchFeatures/<application_date>/ -- DataFrames don't
belong in Airflow's XCom (meant for small, JSON-serializable values), so
disk is the hand-off between tasks here, the same way the training
pipeline's preprocessing notebooks stage everything through
DATA_ROOT/Model-Training/Staging/.
"""
import logging
import os
import sys
from pathlib import Path

import pandas as pd

SRC_ROOT = os.environ.get("HOMECREDIT_SRC_DIR", "/src")
INFERENCE_PKG_SRC = os.environ.get("HOME_CREDIT_INFERENCE_SRC", os.path.join(SRC_ROOT, "Inference"))
if INFERENCE_PKG_SRC not in sys.path:
    sys.path.insert(0, INFERENCE_PKG_SRC)

import config  # noqa: E402
from explainability import explain_contributions  # noqa: E402
from features import application, bureau, credit_card, installments, pos_cash, previous_application  # noqa: E402
from features.pipeline import RAW_COLUMN_REQUIREMENTS, merge_feature_frames  # noqa: E402
from score_model_input import build_band_summary, score_applicants  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

RAW_FILES = {
    "application": "application.csv",
    "bureau": "bureau.csv",
    "previous_application": "previous_application.csv",
    "pos_cash": "POS_CASH_balance.csv",
    "installments": "installments_payments.csv",
    "credit_card": "credit_card_balance.csv",
}

# this function grabs the application_date from the Airflow context, either from the DAG run's params or from the execution date (ds)
def _application_date(**context) -> str:
    params = context.get("params") or {}
    return params.get("application_date") or context["ds"]

# helper functions to construct paths for input and output data
def _app_data_dir(application_date: str) -> Path:
    return Path(config.DATA_ROOT) / "Inference" / "BatchInput" / application_date


def _raw_data_dir() -> Path:
    return Path(config.DATA_ROOT) / "Input"


def _features_dir(application_date: str) -> Path:
    out_dir = Path(config.DATA_ROOT) / "Inference" / "BatchFeatures" / application_date
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir

# helper function to read application IDs from the batch input directory
def _read_application_ids(application_date: str) -> set:
    path = _app_data_dir(application_date) / RAW_FILES["application"]
    return set(pd.read_csv(path, usecols=["SK_ID_CURR"])["SK_ID_CURR"])


def build_application_features(**context) -> None:
    application_date = _application_date(**context)
    path = _app_data_dir(application_date) / RAW_FILES["application"]
    app_df = pd.read_csv(path, usecols=RAW_COLUMN_REQUIREMENTS["application"])
    feats = application.build_features(app_df)

    out_path = _features_dir(application_date) / "application.parquet"
    feats.to_parquet(out_path, index=False)
    logger.info("application features: %s -> %s", feats.shape, out_path)


def build_bureau_features(**context) -> None:
    application_date = _application_date(**context)
    valid_ids = _read_application_ids(application_date)
    raw_df = pd.read_csv(_raw_data_dir() / RAW_FILES["bureau"], usecols=RAW_COLUMN_REQUIREMENTS["bureau"])
    feats = bureau.build_features(raw_df, valid_ids)

    out_path = _features_dir(application_date) / "bureau.parquet"
    feats.to_parquet(out_path, index=False)
    logger.info("bureau features: %s -> %s", feats.shape, out_path)


def build_previous_application_features(**context) -> None:
    application_date = _application_date(**context)
    valid_ids = _read_application_ids(application_date)
    raw_df = pd.read_csv(
        _raw_data_dir() / RAW_FILES["previous_application"],
        usecols=RAW_COLUMN_REQUIREMENTS["previous_application"],
    )
    feats = previous_application.build_features(raw_df, valid_ids)

    out_path = _features_dir(application_date) / "previous_application.parquet"
    feats.to_parquet(out_path, index=False)
    logger.info("previous_application features: %s -> %s", feats.shape, out_path)


def build_pos_cash_features(**context) -> None:
    application_date = _application_date(**context)
    valid_ids = _read_application_ids(application_date)
    raw_dir = _raw_data_dir()
    raw_df = pd.read_csv(raw_dir / RAW_FILES["pos_cash"], usecols=RAW_COLUMN_REQUIREMENTS["pos_cash"])
    # Needed for POS_AVG_CREDIT_TO_APPL (previous-loan enrichment) --
    # see features/pos_cash.py.
    prev_df = pd.read_csv(
        raw_dir / RAW_FILES["previous_application"], usecols=RAW_COLUMN_REQUIREMENTS["previous_application"]
    )
    feats = pos_cash.build_features(raw_df, prev_df, valid_ids)

    out_path = _features_dir(application_date) / "pos_cash.parquet"
    feats.to_parquet(out_path, index=False)
    logger.info("pos_cash features: %s -> %s", feats.shape, out_path)


def build_installments_features(**context) -> None:
    application_date = _application_date(**context)
    valid_ids = _read_application_ids(application_date)
    raw_dir = _raw_data_dir()
    inst_df = pd.read_csv(raw_dir / RAW_FILES["installments"], usecols=RAW_COLUMN_REQUIREMENTS["installments"])
    feats = installments.build_features(inst_df, valid_ids)

    out_path = _features_dir(application_date) / "installments.parquet"
    feats.to_parquet(out_path, index=False)
    logger.info("installments features: %s -> %s", feats.shape, out_path)


def build_credit_card_features(**context) -> None:
    application_date = _application_date(**context)
    valid_ids = _read_application_ids(application_date)
    raw_df = pd.read_csv(
        _raw_data_dir() / RAW_FILES["credit_card"], usecols=RAW_COLUMN_REQUIREMENTS["credit_card"]
    )
    feats = credit_card.build_features(raw_df, valid_ids)

    out_path = _features_dir(application_date) / "credit_card.parquet"
    feats.to_parquet(out_path, index=False)
    logger.info("credit_card features: %s -> %s", feats.shape, out_path)


def join_features(**context) -> None:
    """
    Merges the six per-table feature parquets into the single matrix the
    model expects, and writes it back to disk rather than handing it
    straight to scoring -- so a scoring failure (e.g. the MLflow model
    alias isn't set) can be retried on its own without rebuilding or
    rejoining any of the per-table features, and the assembled matrix is
    itself an inspectable artifact (e.g. for verify_feature_parity.py).
    """
    application_date = _application_date(**context)
    features_dir = _features_dir(application_date)

    app_df = pd.read_parquet(features_dir / "application.parquet")

    feature_frames: dict[str, pd.DataFrame] = {}
    for key in ("bureau", "previous_application", "pos_cash", "installments", "credit_card"):
        path = features_dir / f"{key}.parquet"
        if path.exists():
            feature_frames[key] = pd.read_parquet(path)
        else:
            logger.warning("Feature file for %s not found at %s, skipping", key, path)

    feature_df = merge_feature_frames(app_df, feature_frames)

    out_path = features_dir / "feature_matrix.parquet"
    feature_df.to_parquet(out_path, index=False)
    logger.info("Joined feature matrix: %s -> %s", feature_df.shape, out_path)


def score_features(**context) -> None:
    """
    Scores the feature matrix join_features() wrote and saves results.
    Kept independent of join_features() so it can be retried alone.
    """
    application_date = _application_date(**context)
    features_dir = _features_dir(application_date)

    feature_df = pd.read_parquet(features_dir / "feature_matrix.parquet")

    scored_df = score_applicants(feature_df)
    logger.info(
        "Scored %d applicants | PD range [%.4f, %.4f] | score range [%d, %d]",
        len(scored_df),
        scored_df["PROBA_DEFAULT"].min(),
        scored_df["PROBA_DEFAULT"].max(),
        scored_df["score"].min(),
        scored_df["score"].max(),
    )

    band_summary = build_band_summary(scored_df)
    logger.info("Risk band distribution:\n%s", band_summary.to_string(index=False, float_format="{:.4f}".format))

    # Per-attribute points breakdown -- reason codes for every applicant in
    # this batch. Computed from the same feature_df/model already used to
    # score above; failing here never changes scored_df, since
    # explain_contributions() only reads from the trained model.
    reason_codes_df = explain_contributions(feature_df)
    logger.info("Reason codes shape: %s", reason_codes_df.shape)

    out_dir = Path(config.DATA_ROOT) / "Inference" / "BatchOutput" / application_date
    out_dir.mkdir(parents=True, exist_ok=True)
    scored_df.to_parquet(out_dir / "scored_applicants.parquet", index=False)
    band_summary.to_csv(out_dir / "band_summary.csv", index=False)
    reason_codes_df.to_parquet(out_dir / "reason_codes.parquet", index=False)
    logger.info("Saved scored applicants, band summary, and reason codes to %s", out_dir)
