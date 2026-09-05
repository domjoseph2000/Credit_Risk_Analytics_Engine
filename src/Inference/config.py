"""
Central config for the inference pipeline.

Values below are pinned to what Model_LightGBM_v3.ipynb and
Credit_Scoring.ipynb actually use, not generic defaults -- keep this file
in sync if those notebooks change.
"""
import os
from pathlib import Path

# --- Paths (same env-var pattern as your notebooks) ---
DATA_ROOT = os.environ.get("HOMECREDIT_DATA_ROOT", r"a:\Dominic\My Projects\Home Credit\data")
SRC_ROOT = os.environ.get("HOMECREDIT_SRC_DIR", r"a:\Dominic\My Projects\Home Credit\src")

# --- MLflow ---
# Your training notebook uses a local file-based tracking store, not a
# server -- whatever process runs inference (batch worker, API container)
# needs filesystem access to this path, e.g. a mounted volume in Docker.
MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI", f"file:{DATA_ROOT}/mlruns".replace("\\", "/")
)
MODEL_NAME = os.getenv("HOME_CREDIT_MODEL_NAME", "HomeCredit_Scorecard_Model")
# Your notebook registers versions but leaves promotion as a manual step:
# assign the "prod" alias yourself in the MLflow UI after reviewing a run.
# Aliases, not stages -- MLflow stages are deprecated. Loading uses
# "models:/{MODEL_NAME}@{MODEL_ALIAS}", not "models:/{MODEL_NAME}/Production".
MODEL_ALIAS = "prod"

# --- Feature contract ---
# The exact feature list + order the model was trained on. This is the file
# Model_LightGBM_v3.ipynb reads as `features_df` -- not a separate export
# step, it already exists once you've run that notebook.
SELECTED_FEATURES_PATH = Path(
    os.getenv(
        "HOME_CREDIT_SELECTED_FEATURES",
        str(Path(SRC_ROOT) / "Inference" / "selected_features.xlsx"),
    )
)

# --- Raw data locations (batch mode) ---
# Not currently read by score_model_input.py/batch_feature_tasks.py -- both
# build these same paths locally from DATA_ROOT instead -- but kept correct
# here in case something starts relying on them.
APP_DATA_DIR = os.path.join(DATA_ROOT, "Inference", "BatchInput")
SCORES_OUTPUT_DIR = os.path.join(DATA_ROOT, "Inference", "BatchOutput")
RAW_DATA_DIR = Path(os.getenv("HOME_CREDIT_APP_DATA_DIR", os.path.join(DATA_ROOT, "Input")))


# --- PDO (Points-to-Double-the-Odds) scorecard transform ---
# Pinned to Credit_Scoring.ipynb section 3 & 5. If you change these there,
# change them here too -- there's no shared import between the two repos
# (yet), so this is a manual sync point.
PDO = 20
BASE_SCORE = 600
BASE_ODDS = 50
MIN_SCORE = 300
MAX_SCORE = 850

# Risk bands -- fixed thresholds (not equal-volume deciles), matching
# Credit_Scoring.ipynb section 5. Band edges are the score at which each
# band starts; pd.cut uses right=False so a score of exactly 620 falls in B.
BAND_BINS = [MIN_SCORE, 560, 590, 620, 650, MAX_SCORE + 1]
BAND_LABELS = [
    "E - Very High Risk",
    "D - High Risk",
    "C - Medium Risk",
    "B - Low Risk",
    "A - Very Low Risk",
]
