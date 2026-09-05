#!/usr/bin/env python
"""
Verifies that the features the inference pipeline produces
(features/pipeline.py::assemble_features) match what the model was actually
trained on, restricted to the "final attributes" -- the columns in the
selected-features list that actually reach the model.

Two distinct checks, because they catch different failure modes:

1. Structural: does config.SELECTED_FEATURES_PATH (what assemble_features
   reads at runtime) actually match the feature list the training notebook
   (Model_LightGBM_v3.ipynb) read when the deployed model was fit? These are
   two separate files on disk with no shared source of truth, so they can
   silently drift -- assemble_features() only fails loudly if a name it
   expects is entirely absent from the joined table, not if its own expected
   list is already wrong/stale.

2. Value-level: for features present in both, are the *values* the same for
   the same SK_ID_CURR? A column can exist under the same name on both sides
   and still be computed differently (different aggregation window, "365243
   is a sentinel" only handled on one side, etc.) -- assemble_features()
   cannot catch this by construction, since it only checks column names.

Usage:
    python verify_feature_parity.py [--sample-n 20000] [--out report.csv]

Requires the real Home Credit raw CSVs in DATA_ROOT/Input and the training
notebook's output at
DATA_ROOT/Model-Training/ModelDataset/model_dataset_final.parquet.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config  # noqa: E402
from features.pipeline import assemble_features, load_expected_feature_list  # noqa: E402

RAW_DIR = Path(config.DATA_ROOT) / "Input"
GROUND_TRUTH_PATH = Path(config.DATA_ROOT) / "Model-Training" / "ModelDataset" / "model_dataset_final.parquet"

# The file the training notebook (Model_LightGBM_v3.ipynb, cell 4,
# PATH_SELECTED_VARS) actually reads -- the authoritative feature list for
# whatever model is currently registered. config.SELECTED_FEATURES_PATH is a
# separate copy meant to mirror it; nothing enforces they stay in sync.
TRAINING_FEATURE_LIST_PATH = (
    Path(config.SRC_ROOT) / "Model-Training" / "LightGBM_v3" / "selected_features.xlsx"
)

RAW_FILES = {
    "application": "application_train.csv",
    "bureau": "bureau.csv",
    "previous_application": "previous_application.csv",
    "pos_cash": "POS_CASH_balance.csv",
    "installments": "installments_payments.csv",
}


def check_feature_list_drift() -> list[str]:
    """Structural check: does the inference feature list match training's?"""
    inference_list = load_expected_feature_list()
    training_list = pd.read_excel(TRAINING_FEATURE_LIST_PATH)["feature"].tolist()

    missing_from_inference = sorted(set(training_list) - set(inference_list))
    extra_in_inference = sorted(set(inference_list) - set(training_list))

    print(f"Training feature list  ({TRAINING_FEATURE_LIST_PATH.name} under LightGBM_v3): "
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
    app = pd.read_csv(RAW_DIR / RAW_FILES["application"])
    if sample_n:
        app = app.sample(n=min(sample_n, len(app)), random_state=42)
    ids = set(app["SK_ID_CURR"])

    tables = {"application": app}
    for key in ("bureau", "previous_application", "pos_cash", "installments"):
        df = pd.read_csv(RAW_DIR / RAW_FILES[key])
        tables[key] = df[df["SK_ID_CURR"].isin(ids)]
    return tables


def compare_feature(train_col: pd.Series, infer_col: pd.Series) -> float:
    """Row-aligned match rate for one feature; NaN==NaN counts as a match."""
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
    inference_list = check_feature_list_drift()

    print(f"Loading raw tables from {RAW_DIR} ...")
    raw_tables = load_raw_tables(sample_n)

    print("Running inference feature pipeline (features/pipeline.py) ...")
    infer_df = assemble_features(raw_tables)
    infer_df["SK_ID_CURR"] = infer_df["SK_ID_CURR"].astype("int64")

    print(f"Loading ground-truth training features from {GROUND_TRUTH_PATH} ...")
    cols_available = pq.ParquetFile(GROUND_TRUTH_PATH).schema.names
    train_df = pd.read_parquet(
        GROUND_TRUTH_PATH,
        columns=["SK_ID_CURR"] + [c for c in inference_list if c in cols_available],
    )
    train_df["SK_ID_CURR"] = train_df["SK_ID_CURR"].astype("int64")

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
