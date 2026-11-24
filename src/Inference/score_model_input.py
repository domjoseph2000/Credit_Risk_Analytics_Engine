#!/usr/bin/env python
"""
Batch scoring in one go, without Airflow -- plus the scoring function the
other entry points share.

Run it with the project environment active:

    python score_model_input.py --application-date 2026-07-31

Input
-----
- DATA_ROOT/Inference/BatchInput/<date>/application.csv  the new applicants
- DATA_ROOT/Input/  the credit-history tables (bureau, previous applications,
  POS, repayments, credit cards). These cover everyone Home Credit has ever
  seen and are shared by every batch; only application.csv changes per batch.

Output (DATA_ROOT/Inference/BatchOutput/<date>/)
------------------------------------------------
- scored_applicants.parquet  PD, credit score and risk band per applicant
- band_summary.csv           applicants, mean score and mean PD per risk band
- reason_codes.parquet       credit-score points per feature, per applicant

How it relates to the other entry points
----------------------------------------
- The Airflow DAG (dags/inference_batch_dag.py) doesn't run this script. It
  runs the same steps as separate tasks (utils/batch_feature_tasks.py), using
  score_applicants() and build_band_summary() from here, and writes the
  same files.
- The real-time API (realtime_api.py) also uses score_applicants(), so batch
  and real-time scores always come from the same model and formulas.

Before running: a model version must have the "prod" alias in MLflow (you
set it in the MLflow UI), and selected_features.xlsx must list that
version's features.
"""
import argparse
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

# Make src/Inference importable when this is run from another folder
SRC_ROOT = os.environ.get("HOMECREDIT_SRC_DIR", r"a:\Dominic\My Projects\Home Credit\src")
INFERENCE_PKG_SRC = os.environ.get("HOME_CREDIT_INFERENCE_SRC", os.path.join(SRC_ROOT, "Inference"))
sys.path.insert(0, INFERENCE_PKG_SRC)

from utils import config  # noqa: E402
from utils import pdo_scorecard  # noqa: E402
from utils.explainability import explain_contributions  # noqa: E402
from utils.model import predict_proba  # noqa: E402
from features.pipeline import assemble_features, RAW_COLUMN_REQUIREMENTS  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def score_applicants(feature_df: pd.DataFrame) -> pd.DataFrame:
    """
    Feature table in, scores out: PD -> credit score -> risk band.

    feature_df: the output of features.pipeline.assemble_features (or the
    joined feature file in the Airflow DAG). Used by every way of scoring --
    this script, the Airflow DAG and the real-time API -- which is what
    guarantees they give the same results for the same input.

    Returns one row per applicant: SK_ID_CURR, PROBA_DEFAULT, score, Risk_Band.
    """
    proba = predict_proba(feature_df)  # Series of PDs, indexed by SK_ID_CURR
    score = pd.Series(pdo_scorecard.proba_to_score(proba.to_numpy()), index=proba.index, name="score")
    band = pd.Series(pdo_scorecard.score_to_risk_band(score.to_numpy()), index=proba.index, name="Risk_Band")

    # Column names "score" and "Risk_Band" match Credit_Scoring.ipynb, so the
    # results can be compared with the scorecard analysis directly
    result = pd.concat([proba, score, band], axis=1)
    result.index.name = "SK_ID_CURR"
    return result.reset_index()


# Raw file name for each table. application.csv is read from the dated
# BatchInput/<date>/ folder; the rest from DATA_ROOT/Input.
TABLE_FILES = {
    "application": "application.csv",
    "bureau": "bureau.csv",
    "previous_application": "previous_application.csv",
    "pos_cash": "POS_CASH_balance.csv",
    "installments": "installments_payments.csv",
    "credit_card": "credit_card_balance.csv",
}


def parse_date(value: str) -> str:
    """Accept only YYYY-MM-DD dates for --application-date."""
    try:
        datetime.strptime(value, "%Y-%m-%d")
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"'{value}' is not a valid date, expected YYYY-MM-DD") from e
    return value


def load_shared_history_tables(raw_data_dir: Path) -> dict[str, pd.DataFrame]:
    """
    Load the five credit-history tables (everything except application).

    These cover every applicant Home Credit has ever seen and are the same
    for every batch. Kept separate from load_raw_tables() so the real-time
    API can load them once at startup and reuse them for every request.

    Only the columns the feature code needs are read (see
    RAW_COLUMN_REQUIREMENTS) -- these are multi-million-row files, so this
    saves a lot of memory and time. A missing file is skipped with a warning.
    """
    tables: dict[str, pd.DataFrame] = {}
    for key, filename in TABLE_FILES.items():
        if key == "application":
            continue
        path = raw_data_dir / filename
        if not path.exists():
            logger.warning("Table %s not found at %s, skipping", key, path)
            continue

        usecols = RAW_COLUMN_REQUIREMENTS.get(key)
        tables[key] = pd.read_csv(path, usecols=usecols)
        logger.info("Loaded %-22s %s", key, tables[key].shape)

    return tables


def load_raw_tables(app_data_dir: Path, raw_data_dir: Path) -> dict[str, pd.DataFrame]:
    """
    This batch's application.csv plus the shared history tables. Fails if
    application.csv isn't there.
    """
    app_path = app_data_dir / TABLE_FILES["application"]
    if not app_path.exists():
        raise FileNotFoundError(f"application.csv is required and was not found at {app_path}")

    tables: dict[str, pd.DataFrame] = {
        "application": pd.read_csv(app_path, usecols=RAW_COLUMN_REQUIREMENTS.get("application")),
    }
    logger.info("Loaded %-22s %s", "application", tables["application"].shape)
    tables.update(load_shared_history_tables(raw_data_dir))

    return tables


def build_band_summary(scored_df: pd.DataFrame) -> pd.DataFrame:
    """Per risk band: number of applicants, mean score, mean PD and share of the batch."""
    band_summary = (
        scored_df.groupby("Risk_Band", observed=True)
        .agg(
            Applicant_Count=("SK_ID_CURR", "count"),
            Mean_Score=("score", "mean"),
            Mean_PD=("PROBA_DEFAULT", "mean"),
        )
        .reset_index()
    )
    band_summary["Volume_Pct"] = band_summary["Applicant_Count"] / band_summary["Applicant_Count"].sum()
    return band_summary


def main(application_date: str) -> None:
    """Score one batch: load the data, build features, score, explain, and save the three output files."""
    data_root = Path(config.DATA_ROOT)
    raw_data_dir = data_root / "Input"
    app_data_dir = data_root / "Inference" / "BatchInput" / application_date
    output_dir = data_root / "Inference" / "BatchOutput" / application_date
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Application date : %s", application_date)
    logger.info("Application dir  : %s", app_data_dir)
    logger.info("Shared raw dir   : %s", raw_data_dir)
    logger.info("Output dir       : %s", output_dir)
    logger.info("Model            : %s@%s", config.MODEL_NAME, config.MODEL_ALIAS)
    logger.info("Feature list     : %s", config.SELECTED_FEATURES_PATH)

    raw_tables = load_raw_tables(app_data_dir, raw_data_dir)
    logger.info("Scoring %d new applicants", len(raw_tables["application"]))

    feature_df = assemble_features(raw_tables)
    logger.info("Feature matrix shape: %s", feature_df.shape)

    scored_df = score_applicants(feature_df)
    logger.info(
        "Scored %d applicants | PD range [%.4f, %.4f] | score range [%d, %d]",
        len(scored_df),
        scored_df["PROBA_DEFAULT"].min(),
        scored_df["PROBA_DEFAULT"].max(),
        scored_df["score"].min(),
        scored_df["score"].max(),
    )

    # Logged so you can sanity-check the band mix: a batch very different from
    # the test set's mix (Credit_Scoring.ipynb section 5) may be a genuinely
    # different set of applicants -- or a problem in the feature code.
    band_summary = build_band_summary(scored_df)
    logger.info("Risk band distribution:\n%s", band_summary.to_string(index=False, float_format="{:.4f}".format))

    # Points per feature for every applicant (reason codes). Worked out from
    # the same features and model after scoring; never changes the scores.
    reason_codes_df = explain_contributions(feature_df)
    logger.info("Reason codes shape: %s", reason_codes_df.shape)

    scored_path = output_dir / "scored_applicants.parquet"
    scored_df.to_parquet(scored_path, index=False)
    band_summary_path = output_dir / "band_summary.csv"
    band_summary.to_csv(band_summary_path, index=False)
    reason_codes_path = output_dir / "reason_codes.parquet"
    reason_codes_df.to_parquet(reason_codes_path, index=False)

    logger.info("Saved scored applicants to %s", scored_path)
    logger.info("Saved band summary to %s", band_summary_path)
    logger.info("Saved reason codes to %s", reason_codes_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--application-date",
        required=True,
        type=parse_date,
        help="Date (YYYY-MM-DD) of the new-applicant batch to score. Selects "
        "DATA_ROOT/Inference/BatchInput/<date>/application.csv as input and "
        "DATA_ROOT/Inference/BatchOutput/<date>/ as output. Same flag whether "
        "run manually or by the Airflow DAG.",
    )
    args = parser.parse_args()

    # Exit with an error code on failure, so a scheduler or script calling this notices
    try:
        main(args.application_date)
    except Exception:
        logger.exception("Batch scoring failed for application date %s", args.application_date)
        sys.exit(1)
