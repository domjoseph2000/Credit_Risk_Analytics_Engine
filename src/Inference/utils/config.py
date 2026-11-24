"""
Settings for the inference code (batch scoring, real-time API, dashboard).

Everything here is tied to the model currently promoted to @prod in MLflow --
right now MLflow version 23 (02_training_LightGBM_v4.ipynb, 23 features).
When you promote a different model, update this file together with
src/Inference/features/ and src/Inference/selected_features.xlsx, then run
verify_feature_parity.py (see the README's "Known gaps").

Each path can be overridden with an environment variable. The defaults are
for running on your own PC; inside Docker, docker-compose sets them to /data
and /src.
"""
import os
from pathlib import Path

# --- Project folders ---
DATA_ROOT = os.environ.get("HOMECREDIT_DATA_ROOT", r"a:\Dominic\My Projects\Home Credit\data")
SRC_ROOT = os.environ.get("HOMECREDIT_SRC_DIR", r"a:\Dominic\My Projects\Home Credit\src")

# --- MLflow ---
# Address of the MLflow server the model is loaded from. On your PC that's
# localhost:5000 (the Docker `mlflow` service or scripts/start_mlflow_server.py);
# inside Docker, docker-compose sets it to http://mlflow:5000.
# Model files are downloaded through the server. Versions registered before
# the switch to the server (1–11) still point at file:///data/mlruns/... and
# only load inside Docker (see the README's "Known gaps").
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MODEL_NAME = os.getenv("HOME_CREDIT_MODEL_NAME", "HomeCredit_Scorecard_Model")
# The model is loaded as "models:/<MODEL_NAME>@prod". Training registers new
# versions but never promotes them: you set the "prod" alias yourself in the
# MLflow UI after reviewing a run. (MLflow aliases replace the older, deprecated
# "Production" stage.)
MODEL_ALIAS = "prod"

# --- Feature list ---
# The exact features (names and order) the @prod model was trained on. When
# you promote a new model, replace this file's list with that version's
# data/Model-Training/HyperParameters/<model_folder>/selected_features.json.
SELECTED_FEATURES_PATH = Path(
    os.getenv(
        "HOME_CREDIT_SELECTED_FEATURES",
        str(Path(SRC_ROOT) / "Inference" / "selected_features.xlsx"),
    )
)

# --- Batch input/output folders ---
# Batch scoring (score_model_input.py, batch_feature_tasks.py) builds these
# same paths from DATA_ROOT itself; the dashboard reads SCORES_OUTPUT_DIR and
# the real-time API reads RAW_DATA_DIR.
APP_DATA_DIR = os.path.join(DATA_ROOT, "Inference", "BatchInput")       # <date>/application.csv per batch
SCORES_OUTPUT_DIR = os.path.join(DATA_ROOT, "Inference", "BatchOutput")  # <date>/ results per batch
RAW_DATA_DIR = Path(os.getenv("HOME_CREDIT_APP_DATA_DIR", os.path.join(DATA_ROOT, "Input")))  # shared credit history


# --- Turning PD into a credit score (PDO scorecard) ---
# Must match Credit_Scoring.ipynb (sections 3 and 5) -- nothing links the two
# automatically, so change both together.
#   PDO         points that double the odds of repaying (good:bad)
#   BASE_SCORE  the score given at BASE_ODDS
#   BASE_ODDS   good:bad odds of 50:1 score BASE_SCORE (600)
# Scores are rounded and kept between MIN_SCORE and MAX_SCORE.
PDO = 20
BASE_SCORE = 600
BASE_ODDS = 50
MIN_SCORE = 300
MAX_SCORE = 850

# Risk bands, from fixed score cut-offs (not equal-sized groups), matching
# Credit_Scoring.ipynb section 5. Each number is where a band starts:
#   300–559 E, 560–589 D, 590–619 C, 620–649 B, 650–850 A
# (a score of exactly 620 is in B).
BAND_BINS = [MIN_SCORE, 560, 590, 620, 650, MAX_SCORE + 1]
BAND_LABELS = [
    "E — Very High Risk",
    "D — High Risk",
    "C — Medium Risk",
    "B — Low Risk",
    "A — Very Low Risk",
]
