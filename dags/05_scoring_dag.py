"""
Home Credit – 05 Scoring DAG (final model only)
===============================================
Step 5 of the training pipeline: builds the credit scorecard for the model you
have chosen to deploy.

    check_inputs → score_model

When to run it
--------------
After comparing versions trained with 04_model_pipeline (test metrics,
segmentation analysis) and picking one. The scorecard -- turning each
predicted default probability (PD) into a 300–850 credit score, A–E risk
bands, score charts and the KS test -- is only needed for the final model,
so it isn't part of the per-version pipeline. Start it manually.

What it does
------------
Runs src/Training/06_Scoring/Credit_Scoring.ipynb. Nothing is retrained: the
notebook reads the predictions the training notebook already saved in
data/Model-Training/ModelOutput/<model_folder>/test_predictions.parquet and
writes its results to data/Model-Training/ScoreOutput/<model_folder>/.

Params:
- `model_folder`: the final model. The choices are the versions that already have
  a test_predictions.parquet, i.e. have been trained; the default is the most
  recently trained one.
"""
from datetime import datetime, timedelta
import os
from airflow import DAG
from airflow.exceptions import AirflowFailException
from airflow.models.param import Param
from airflow.operators.python import PythonOperator
import papermill as pm

# ── Paths ─────────────────────────────────────────────────────────────────────
# Inside the container: data/ is mounted at /data and src/ at /src.
DATA_ROOT = os.environ.get("HOMECREDIT_DATA_ROOT", "/data")
SRC_DIR = os.environ.get("HOMECREDIT_SRC_DIR", "/src")
MODEL_OUTPUT_ROOT = os.path.join(DATA_ROOT, "Model-Training", "ModelOutput")   # one folder per trained version
SCORING_NB = os.path.join(SRC_DIR, "Training", "06_Scoring", "Credit_Scoring.ipynb")


def _predictions_path(model_folder: str) -> str:
    return os.path.join(MODEL_OUTPUT_ROOT, model_folder, "test_predictions.parquet")


# Versions that can be scored = the trained ones (their test_predictions.parquet
# exists). Re-read whenever Airflow re-parses this file.
TRAINED_FOLDERS = sorted(
    d for d in (os.listdir(MODEL_OUTPUT_ROOT) if os.path.isdir(MODEL_OUTPUT_ROOT) else [])
    if os.path.exists(_predictions_path(d))
)
# Default = the most recently trained version (newest test_predictions.parquet),
# re-evaluated each time Airflow parses this file. The fallback only applies when
# nothing is trained or visible at parse time.
DEFAULT_MODEL_FOLDER = max(
    TRAINED_FOLDERS, key=lambda d: os.path.getmtime(_predictions_path(d)), default="LightGBM_v1"
)

# Settings shared by every task: one retry after 5 minutes, no emails
DEFAULT_ARGS = {
    "owner": "Admin",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
    "email_on_failure": False,
    "email_on_retry": False,
}

# Options on the "Trigger DAG" screen: a dropdown of the trained versions
# (a free-text box if none were found at parse time)
_model_folder_choices = {"enum": TRAINED_FOLDERS} if TRAINED_FOLDERS else {}
UI_PARAMS = {
    "model_folder": Param(
        default=DEFAULT_MODEL_FOLDER,
        type="string",
        title="Final model",
        description="The model to build the scorecard for. Listed: versions with a "
        "data/Model-Training/ModelOutput/<model_folder>/test_predictions.parquet (already trained). "
        "Defaults to the most recently trained one.",
        **_model_folder_choices,
    ),
}


# ── Task functions ────────────────────────────────────────────────────────────
def check_inputs(**context) -> None:
    """Stop early, with a clear message, if the chosen version hasn't been trained yet."""
    model_folder = context["params"]["model_folder"]
    path = _predictions_path(model_folder)
    if not os.path.exists(path):
        raise AirflowFailException(
            f"{path} doesn't exist -- train {model_folder} first (04_model_pipeline).")
    print(f"Scoring {model_folder} from {path}")


def run_scoring_notebook(task_id: str, **context) -> None:
    """
    Run Credit_Scoring.ipynb for the chosen version. The executed copy is saved
    as notebook_outputs/<task>__<model_folder>__<run date>.ipynb.
    """
    model_folder = context["params"]["model_folder"]
    # Tell the notebook which version to score (it reads this into MODEL_FOLDER).
    # Set as an environment variable before the notebook starts, because the
    # notebook's Python process inherits this process's environment.
    os.environ["HOMECREDIT_SCORING_MODEL_FOLDER"] = model_folder

    out_dir = os.path.join(DATA_ROOT, "Model-Training", "notebook_outputs")
    os.makedirs(out_dir, exist_ok=True)
    run_date = context.get("ds_nodash") or datetime.now().strftime("%Y%m%d")
    output_nb = os.path.join(out_dir, f"{task_id}__{model_folder}__{run_date}.ipynb")

    print(f"[{task_id}] Running: {SCORING_NB}")
    print(f"[{task_id}] Output : {output_nb}")
    pm.execute_notebook(
        input_path=SCORING_NB,
        output_path=output_nb,
        kernel_name="python3",
        parameters={},  # none needed -- the notebook reads its settings from env vars
        progress_bar=False,
        request_save_on_cell_execute=True,  # save after every cell, so a failed run still shows how far it got
    )
    print(f"[{task_id}] ✓ Complete")


# ── DAG definition ────────────────────────────────────────────────────────────
with DAG(
    dag_id="05_scoring",
    description="Credit scorecard (PDO score, risk bands, KS) for the final model",
    default_args=DEFAULT_ARGS,
    # Airflow requires a start date even for a DAG with no schedule
    start_date=datetime(2024, 1, 1),
    schedule=None,  # start it manually once you've picked the final model
    params=UI_PARAMS,
    max_active_runs=1,
    catchup=False,
    tags=["homecredit", "scoring"],
) as dag:

    task_check_inputs = PythonOperator(task_id="check_inputs", python_callable=check_inputs)

    # doc_md is the task description shown in the Airflow UI
    score_model = PythonOperator(
        task_id="score_model",
        python_callable=run_scoring_notebook,
        op_kwargs={"task_id": "score_model"},
        doc_md="""
        **Credit scoring** -- PD to PDO credit score, A-E risk bands, score charts and KS;
        writes to `data/Model-Training/ScoreOutput/<model_folder>/`.
        """,
    )

    task_check_inputs >> score_model
