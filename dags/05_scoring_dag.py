"""
Home Credit – Scoring DAG (final model only)
=============================================
Builds the credit scorecard for the model you've chosen to deploy:

    check_inputs → score_model

Run it after comparing versions with 04_model_pipeline (test metrics, segmentation
analysis) -- the scorecard (PD → PDO credit score, A–E risk bands, KS) belongs to
the final model only, so it isn't part of the per-version pipeline.

No retraining: Credit_Scoring.ipynb reads the version's saved
data/Model-Training/ModelOutput/<model_folder>/test_predictions.parquet (written by
its training notebook) and writes to data/Model-Training/ScoreOutput/<model_folder>/.

Params:
- `model_folder`: the final model. The choices are the versions that already have
  a test_predictions.parquet, i.e. have been trained.
"""
from datetime import datetime, timedelta
import os
from airflow import DAG
from airflow.exceptions import AirflowFailException
from airflow.models.param import Param
from airflow.operators.python import PythonOperator
import papermill as pm

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_ROOT = os.environ.get("HOMECREDIT_DATA_ROOT", "/data")
SRC_DIR = os.environ.get("HOMECREDIT_SRC_DIR", "/src")
MODEL_OUTPUT_ROOT = os.path.join(DATA_ROOT, "Model-Training", "ModelOutput")
SCORING_NB = os.path.join(SRC_DIR, "Training", "06_Scoring", "Credit_Scoring.ipynb")


def _predictions_path(model_folder: str) -> str:
    return os.path.join(MODEL_OUTPUT_ROOT, model_folder, "test_predictions.parquet")


# Scorable versions = trained ones (their test_predictions.parquet exists)
TRAINED_FOLDERS = sorted(
    d for d in (os.listdir(MODEL_OUTPUT_ROOT) if os.path.isdir(MODEL_OUTPUT_ROOT) else [])
    if os.path.exists(_predictions_path(d))
)

DEFAULT_ARGS = {
    "owner": "Admin",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
    "email_on_failure": False,
    "email_on_retry": False,
}

# Dropdown of trained versions (free text if none are visible at parse time)
_model_folder_choices = {"enum": TRAINED_FOLDERS} if TRAINED_FOLDERS else {}
UI_PARAMS = {
    "model_folder": Param(
        default="LightGBM_v3" if "LightGBM_v3" in TRAINED_FOLDERS or not TRAINED_FOLDERS else TRAINED_FOLDERS[0],
        type="string",
        title="Final model",
        description="The model to build the scorecard for. Listed: versions with a "
        "data/Model-Training/ModelOutput/<model_folder>/test_predictions.parquet (already trained).",
        **_model_folder_choices,
    ),
}


# ── Task callables ────────────────────────────────────────────────────────────
def check_inputs(**context) -> None:
    """Fail fast, with a clear message, if the chosen version hasn't been trained."""
    model_folder = context["params"]["model_folder"]
    path = _predictions_path(model_folder)
    if not os.path.exists(path):
        raise AirflowFailException(
            f"{path} doesn't exist -- train {model_folder} first (04_model_pipeline).")
    print(f"Scoring {model_folder} from {path}")


def run_scoring_notebook(task_id: str, **context) -> None:
    model_folder = context["params"]["model_folder"]
    # Read by Credit_Scoring.ipynb's MODEL_FOLDER -- set before execute_notebook so
    # the kernel subprocess inherits it
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
        parameters={},  # the notebook reads its settings from env vars
        progress_bar=False,
        request_save_on_cell_execute=True,
    )
    print(f"[{task_id}] ✓ Complete")


# ── DAG definition ────────────────────────────────────────────────────────────
with DAG(
    dag_id="05_scoring",
    description="Credit scorecard (PDO score, risk bands, KS) for the final model",
    default_args=DEFAULT_ARGS,
    # Even with schedule=None, Airflow requires a start_date.
    start_date=datetime(2024, 1, 1),
    schedule=None,  # manual: run once you've picked the final model
    params=UI_PARAMS,
    max_active_runs=1,
    catchup=False,
    tags=["homecredit", "scoring"],
) as dag:

    task_check_inputs = PythonOperator(task_id="check_inputs", python_callable=check_inputs)

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
