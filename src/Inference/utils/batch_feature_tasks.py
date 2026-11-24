"""
The task functions behind the Airflow batch inference DAG
(dags/inference_batch_dag.py).

They do the same work as score_model_input.py, split into one task per step
so each can fail and be retried on its own in Airflow:

    build_application_features   ─┐
    build_bureau_features         │
    build_previous_application_…  ├─► join_features ─► score_features
    build_pos_cash_features       │
    build_installments_features   │
    build_credit_card_features   ─┘

- The six build tasks don't depend on each other: each one reads just the
  SK_ID_CURR column of this batch's application.csv to know which applicants
  to keep, instead of waiting for another task.
- Tasks hand their results to each other as parquet files in
  DATA_ROOT/Inference/BatchFeatures/<date>/ (Airflow's own hand-off
  mechanism, XCom, is meant for small values, not tables).
- join_features writes the combined feature table to disk too, so a scoring
  failure can be retried without redoing the join, and the table can be
  inspected afterwards.

Every task works out the batch date the same way: the DAG run's
`application_date` param, or the run's own date if that's blank.
"""
import logging
import os
import sys
from pathlib import Path

import pandas as pd

# Make src/Inference importable (it's mounted at /src inside the container)
SRC_ROOT = os.environ.get("HOMECREDIT_SRC_DIR", "/src")
INFERENCE_PKG_SRC = os.environ.get("HOME_CREDIT_INFERENCE_SRC", os.path.join(SRC_ROOT, "Inference"))
if INFERENCE_PKG_SRC not in sys.path:
    sys.path.insert(0, INFERENCE_PKG_SRC)

from utils import config  # noqa: E402
from utils.explainability import explain_contributions  # noqa: E402
from features import application, bureau, credit_card, installments, pos_cash, previous_application  # noqa: E402
from features.pipeline import RAW_COLUMN_REQUIREMENTS, merge_feature_frames  # noqa: E402
from score_model_input import build_band_summary, score_applicants  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# Raw file name for each table
RAW_FILES = {
    "application": "application.csv",
    "bureau": "bureau.csv",
    "previous_application": "previous_application.csv",
    "pos_cash": "POS_CASH_balance.csv",
    "installments": "installments_payments.csv",
    "credit_card": "credit_card_balance.csv",
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def _application_date(**context) -> str:
    """The batch to score: the run's `application_date` param, or the run's own date (ds) if blank."""
    params = context.get("params") or {}
    return params.get("application_date") or context["ds"]


def _app_data_dir(application_date: str) -> Path:
    """Folder holding this batch's application.csv."""
    return Path(config.DATA_ROOT) / "Inference" / "BatchInput" / application_date


def _raw_data_dir() -> Path:
    """Folder holding the shared credit-history tables."""
    return Path(config.DATA_ROOT) / "Input"


def _features_dir(application_date: str) -> Path:
    """Folder for this batch's intermediate feature files (created if needed)."""
    out_dir = Path(config.DATA_ROOT) / "Inference" / "BatchFeatures" / application_date
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _read_application_ids(application_date: str) -> set:
    """The SK_ID_CURR values in this batch (reads just that one column)."""
    path = _app_data_dir(application_date) / RAW_FILES["application"]
    return set(pd.read_csv(path, usecols=["SK_ID_CURR"])["SK_ID_CURR"])


# ── One task per source table ─────────────────────────────────────────────────
# Each reads its raw table (only the columns it needs), keeps this batch's
# applicants, builds that table's features, and saves them as
# BatchFeatures/<date>/<table>.parquet.

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
    # Also needs previous_application, for POS_AVG_CREDIT_TO_APPL (see features/pos_cash.py)
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


# ── Join and score ────────────────────────────────────────────────────────────

def join_features(**context) -> None:
    """
    Combine the six feature files into the model's input table and save it as
    BatchFeatures/<date>/feature_matrix.parquet.

    Uses the same merge_feature_frames() as every other way of scoring. A
    missing history file is skipped with a warning (those features are then
    blank); a feature the model needs that's still missing stops the task.
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
    Score the joined feature table and save the results to
    DATA_ROOT/Inference/BatchOutput/<date>/: scored_applicants.parquet,
    band_summary.csv and reason_codes.parquet.
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

    # Points per feature for every applicant (reason codes). Worked out from
    # the same features and model after scoring; never changes the scores.
    reason_codes_df = explain_contributions(feature_df)
    logger.info("Reason codes shape: %s", reason_codes_df.shape)

    out_dir = Path(config.DATA_ROOT) / "Inference" / "BatchOutput" / application_date
    out_dir.mkdir(parents=True, exist_ok=True)
    scored_df.to_parquet(out_dir / "scored_applicants.parquet", index=False)
    band_summary.to_csv(out_dir / "band_summary.csv", index=False)
    reason_codes_df.to_parquet(out_dir / "reason_codes.parquet", index=False)
    logger.info("Saved scored applicants, band summary, and reason codes to %s", out_dir)
