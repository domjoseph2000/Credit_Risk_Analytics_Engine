#!/usr/bin/env python
"""
Batch scoring: shared score_applicants() + the CLI that runs it, both
manually and via Airflow.

Manual:
    python score_model_input.py --application-date 2026-07-31

Airflow (inference_batch_dag.py runs the exact same command):
    python score_model_input.py --application-date {{ ds }}

Only application.csv is date-partitioned (a new batch of applicants each
run); bureau/previous_application/pos_cash/installments are shared credit
history and are always read from DATA_ROOT/Input, not re-dumped per run.

score_applicants() below is also the one function realtime_api.py calls,
so batch and real-time scoring can't silently diverge -- both paths go
through the same model load, PDO transform, and risk-band assignment.

Outputs:
    DATA_ROOT/Inference/BatchOutput/<application-date>/scored_applicants.parquet
    DATA_ROOT/Inference/BatchOutput/<application-date>/band_summary.csv

Prerequisite: HomeCredit_Scorecard_Model must have a version with the
"prod" alias assigned (done manually in the MLflow UI after review -- see
Model_LightGBM_v3.ipynb section 8), and selected_features.xlsx must exist.
"""
import argparse
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

# Same env-var pattern as the other scripts in this package, so this drops
# into the same layout without edits.
SRC_ROOT = os.environ.get("HOMECREDIT_SRC_DIR", r"a:\Dominic\My Projects\Home Credit\src")
INFERENCE_PKG_SRC = os.environ.get("HOME_CREDIT_INFERENCE_SRC", os.path.join(SRC_ROOT, "Inference"))
sys.path.insert(0, INFERENCE_PKG_SRC)

import config  # noqa: E402
import pdo_scorecard  # noqa: E402
from model import predict_proba  # noqa: E402
from features.pipeline import assemble_features, RAW_COLUMN_REQUIREMENTS  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def score_applicants(feature_df: pd.DataFrame) -> pd.DataFrame:
    """
    Score applicants based on the features in feature_df (output of
    features.pipeline.assemble_features). Also called directly by
    realtime_api.py -- keeping this one function shared is what actually
    guarantees batch and real-time give identical scores for identical
    inputs.
    """
    proba = predict_proba(feature_df)  # pd.Series, indexed by SK_ID_CURR
    score = pd.Series(pdo_scorecard.proba_to_score(proba.to_numpy()), index=proba.index, name="score")
    band = pd.Series(pdo_scorecard.score_to_risk_band(score.to_numpy()), index=proba.index, name="Risk_Band")

    # Column names ("score", "Risk_Band") match Credit_Scoring.ipynb so this
    # output can be concatenated directly with your scorecard analysis.
    result = pd.concat([proba, score, band], axis=1)
    result.index.name = "SK_ID_CURR"
    return result.reset_index()


# application.csv lives under the dated BatchInput dir; the rest are shared
# history read from DATA_ROOT/Input regardless of application-date.
TABLE_FILES = {
    "application": "application.csv",
    "bureau": "bureau.csv",
    "previous_application": "previous_application.csv",
    "pos_cash": "POS_CASH_balance.csv",
    "installments": "installments_payments.csv",
}


def parse_date(value: str) -> str:
    try:
        datetime.strptime(value, "%Y-%m-%d")
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"'{value}' is not a valid date, expected YYYY-MM-DD") from e
    return value


def load_raw_tables(app_data_dir: Path, raw_data_dir: Path) -> dict[str, pd.DataFrame]:
    """
    Reads only the raw columns each feature module actually consumes (see
    features/pipeline.py::RAW_COLUMN_REQUIREMENTS) instead of full tables --
    bureau/previous_application/installments are multi-million-row CSVs, so
    this is a large, free memory/IO win over reading every raw column.
    """
    tables: dict[str, pd.DataFrame] = {}
    for key, filename in TABLE_FILES.items():
        source_dir = app_data_dir if key == "application" else raw_data_dir
        path = source_dir / filename
        if not path.exists():
            if key == "application":
                raise FileNotFoundError(f"application.csv is required and was not found at {path}")
            logger.warning("Table %s not found at %s, skipping", key, path)
            continue

        usecols = RAW_COLUMN_REQUIREMENTS.get(key)
        tables[key] = pd.read_csv(path, usecols=usecols)
        logger.info("Loaded %-22s %s", key, tables[key].shape)

    return tables


def build_band_summary(scored_df: pd.DataFrame) -> pd.DataFrame:
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

    # A batch whose band distribution is wildly different from the
    # validation-set distribution (Credit_Scoring.ipynb section 5) is worth
    # a second look before it feeds downstream decisions -- could be a
    # genuinely different applicant pool, or a feature-assembly bug upstream.
    band_summary = build_band_summary(scored_df)
    logger.info("Risk band distribution:\n%s", band_summary.to_string(index=False, float_format="{:.4f}".format))

    scored_path = output_dir / "scored_applicants.parquet"
    scored_df.to_parquet(scored_path, index=False)
    band_summary_path = output_dir / "band_summary.csv"
    band_summary.to_csv(band_summary_path, index=False)

    logger.info("Saved scored applicants to %s", scored_path)
    logger.info("Saved band summary to %s", band_summary_path)


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

    try:
        main(args.application_date)
    except Exception:
        logger.exception("Batch scoring failed for application date %s", args.application_date)
        sys.exit(1)
