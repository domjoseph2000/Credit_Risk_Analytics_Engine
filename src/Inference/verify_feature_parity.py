#!/usr/bin/env python
"""
Checks that inference calculates the model's features exactly as training did.

Run this after promoting a new model or changing anything in features/.
If inference calculated a feature differently from training, the model
would quietly give wrong scores rather than an error -- this is what catches it.

Two checks
----------
1. Same feature list? Inference reads its list from selected_features.xlsx
   (config.SELECTED_FEATURES_PATH). The @prod model was trained on the list
   in HyperParameters/<PROD_MODEL_FOLDER>/selected_features.json. They're
   separate files that must be kept in step by hand; this reports any
   feature in one but not the other.

2. Same values? For a sample of applicants, rebuilds the features from the
   raw CSVs with the inference code and compares them, applicant by
   applicant, with the training dataset (model_dataset_final.parquet). A
   feature can have the right name but still be calculated differently
   (e.g. a different time window), which only a value check catches.
   Prints each feature's match rate; 1.0 means every value matched.

Usage (project environment active):
    python verify_feature_parity.py [--sample-n 20000] [--out report.csv]

Needs the raw Kaggle CSVs in DATA_ROOT/Input and
DATA_ROOT/Model-Training/ModelDataset/model_dataset_final.parquet.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

# Make this folder's modules (config, features) importable
sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils import config  # noqa: E402
from features.pipeline import assemble_features, load_expected_feature_list  # noqa: E402

RAW_DIR = Path(config.DATA_ROOT) / "Input"
# The training dataset: the "right answers" to compare against
GROUND_TRUTH_PATH = Path(config.DATA_ROOT) / "Model-Training" / "ModelDataset" / "model_dataset_final.parquet"

# The feature list the @prod model was actually trained on: its tuning
# notebook wrote it and its training notebook read it. Change
# PROD_MODEL_FOLDER when you promote a different version
# (LightGBM_v4 = MLflow version 23, the current @prod).
PROD_MODEL_FOLDER = "LightGBM_v4"
TRAINING_FEATURE_LIST_PATH = (
    Path(config.DATA_ROOT) / "Model-Training" / "HyperParameters" / PROD_MODEL_FOLDER / "selected_features.json"
)

# Raw files to rebuild the features from. application_train.csv (not a
# batch's application.csv), because those are the applicants in the training data.
RAW_FILES = {
    "application": "application_train.csv",
    "bureau": "bureau.csv",
    "previous_application": "previous_application.csv",
    "pos_cash": "POS_CASH_balance.csv",
    "installments": "installments_payments.csv",
    "credit_card": "credit_card_balance.csv",
}


def check_feature_list_drift() -> list[str]:
    """Check 1: print any feature that's in inference's list but not training's, or the other way round. Returns inference's list."""
    inference_list = load_expected_feature_list()
    with open(TRAINING_FEATURE_LIST_PATH) as f:
        training_list = json.load(f)["features"]

    missing_from_inference = sorted(set(training_list) - set(inference_list))
    extra_in_inference = sorted(set(inference_list) - set(training_list))

    print(f"Training feature list  ({TRAINING_FEATURE_LIST_PATH.name} under {PROD_MODEL_FOLDER}): "
          f"{len(training_list)} features")
    print(f"Inference feature list ({config.SELECTED_FEATURES_PATH.name} under Inference): "
          f"{len(inference_list)} features")
    if missing_from_inference:
        print(f"  DRIFT: in training but NOT in inference's list: {missing_from_inference}")
        print("  -> the deployed model was fit on a feature these predictions never populate.")
    if extra_in_inference:
        print(f"  DRIFT: in inference but NOT in training's list: {extra_in_inference}")
    if not missing_from_inference and not extra_in_inference:
        print("  OK: identical feature sets.")
    print()
    return inference_list


def load_raw_tables(sample_n: int | None) -> dict[str, pd.DataFrame]:
    """
    Load the raw tables, optionally for a random sample of sample_n applicants
    (fixed seed, so reruns use the same sample). History tables are cut down
    to those applicants.
    """
    app = pd.read_csv(RAW_DIR / RAW_FILES["application"])
    if sample_n:
        app = app.sample(n=min(sample_n, len(app)), random_state=42)
    ids = set(app["SK_ID_CURR"])

    tables = {"application": app}
    for key in ("bureau", "previous_application", "pos_cash", "installments", "credit_card"):
        df = pd.read_csv(RAW_DIR / RAW_FILES[key])
        tables[key] = df[df["SK_ID_CURR"].isin(ids)]
    return tables


def compare_feature(train_col: pd.Series, infer_col: pd.Series) -> float:
    """
    Share of applicants whose value matches, for one feature (0 to 1).
    Numbers count as equal within a tiny rounding tolerance; blank on both
    sides counts as a match.
    """
    if pd.api.types.is_numeric_dtype(train_col) and pd.api.types.is_numeric_dtype(infer_col):
        match = np.isclose(
            train_col.to_numpy(dtype="float64"),
            infer_col.to_numpy(dtype="float64"),
            rtol=1e-5, atol=1e-6, equal_nan=True,
        )
    else:
        both_na = train_col.isna() & infer_col.isna()
        match = (train_col.astype(str) == infer_col.astype(str)) | both_na
    return float(np.mean(match))


def main(sample_n: int | None, out_path: str | None):
    """Run both checks and print a match rate per feature (optionally also saved as a CSV)."""
    # Check 1: same feature list
    inference_list = check_feature_list_drift()

    # Check 2: same values -- rebuild the features with the inference code...

    print(f"Loading raw tables from {RAW_DIR} ...")
    raw_tables = load_raw_tables(sample_n)

    print("Running inference feature pipeline (features/pipeline.py) ...")
    infer_df = assemble_features(raw_tables)
    infer_df["SK_ID_CURR"] = infer_df["SK_ID_CURR"].astype("int64")

    # ...load the same applicants' features from the training dataset...
    print(f"Loading ground-truth training features from {GROUND_TRUTH_PATH} ...")
    cols_available = pq.ParquetFile(GROUND_TRUTH_PATH).schema.names
    train_df = pd.read_parquet(
        GROUND_TRUTH_PATH,
        columns=["SK_ID_CURR"] + [c for c in inference_list if c in cols_available],
    )
    train_df["SK_ID_CURR"] = train_df["SK_ID_CURR"].astype("int64")

    # ...and line them up applicant by applicant
    merged = train_df.merge(infer_df, on="SK_ID_CURR", how="inner", suffixes=("_train", "_infer"))
    print(f"Compared population: {len(merged)} applicants "
          f"(train had {len(train_df)}, inference produced {len(infer_df)})\n")

    rows = []
    for feat in inference_list:
        t_col, i_col = f"{feat}_train", f"{feat}_infer"
        if t_col not in merged.columns or i_col not in merged.columns:
            rows.append({"feature": feat, "match_rate": np.nan, "note": "not found in ground truth"})
            continue
        rate = compare_feature(merged[t_col], merged[i_col])
        rows.append({"feature": feat, "match_rate": rate, "note": ""})

    report = pd.DataFrame(rows).sort_values("match_rate")
    pd.set_option("display.width", 120)
    print("Feature parity report (lower match_rate = worse):")
    print(report.to_string(index=False))

    n_bad = (report["match_rate"] < 1.0).sum()
    print(f"\n{n_bad}/{len(report)} compared features have at least one mismatched value.")

    if out_path:
        report.to_csv(out_path, index=False)
        print(f"Full report written to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-n", type=int, default=None, help="Sample N applicants for speed")
    parser.add_argument("--out", type=str, default=None, help="Write CSV report to this path")
    args = parser.parse_args()
    main(args.sample_n, args.out)
