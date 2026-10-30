"""
Central config for the inference pipeline.

Values below are pinned to what the currently-promoted (@prod) training
notebook and Credit_Scoring.ipynb actually use, not generic defaults --
keep this file (and src/Inference/features/, src/Inference/selected_features.xlsx)
in sync whenever a new model version is promoted. Currently: 02_training_LightGBM_v4.ipynb.
"""
import os
from pathlib import Path

# --- Paths (same env-var pattern as your notebooks) ---
DATA_ROOT = os.environ.get("HOMECREDIT_DATA_ROOT", r"a:\Dominic\My Projects\Home Credit\data")
SRC_ROOT = os.environ.get("HOMECREDIT_SRC_DIR", r"a:\Dominic\My Projects\Home Credit\src")

# --- MLflow ---
# The docker-compose `mlflow` tracking server: it keeps metadata in
# DATA_ROOT/mlflow.db (SQLite) and serves artifacts over HTTP, so model files
# are downloaded through it rather than opened from a path baked in at logging
# time. Inside a compose container use http://mlflow:5000. Model versions
# registered before the switch still point at file:///data/mlruns/... and
# only load where /data exists (see README's "Known gaps").
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MODEL_NAME = os.getenv("HOME_CREDIT_MODEL_NAME", "HomeCredit_Scorecard_Model")
# Your notebook registers versions but leaves promotion as a manual step:
# assign the "prod" alias yourself in the MLflow UI after reviewing a run.
# Aliases, not stages -- MLflow stages are deprecated. Loading uses
# "models:/{MODEL_NAME}@{MODEL_ALIAS}", not "models:/{MODEL_NAME}/Production".
MODEL_ALIAS = "prod"

# --- Feature contract ---
# The exact feature list the currently-promoted model was trained on --
# copy the 'feature' column out of that model's lgbm_feature_importance.xlsx
# (data/Model-Training/ModelOutput/<model_folder>/) whenever a new model is
# promoted to @prod.
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
    "E — Very High Risk",
    "D — High Risk",
    "C — Medium Risk",
    "B — Low Risk",
    "A — Very Low Risk",
]
