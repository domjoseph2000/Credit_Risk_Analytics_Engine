"""
Home Credit – Model Pipeline DAG (tuning → training → segmentation)
====================================================================
One run builds and evaluates one model version:

    check_inputs → tune_model → train_model → run_segmentation

Scoring (PDO scorecard, risk bands) is not part of this DAG: it's only for the
final model you pick after comparing versions -- run 05_scoring for that.

A model version is a folder under src/Training/04_Model-Development/ holding
exactly one tuning notebook (01_tuning*.ipynb) and one training notebook
(02_training*.ipynb). The DAG discovers the versions and their notebooks from
those folders, so adding a version needs no DAG edit, and tuning/training
can't be mismatched.

Tuning and training are one because training depends on tuning's
outputs for the same version -- `selected_features.json` (the feature list) and
`*_resolved_best.json` (hyperparameters).

Params:
- `tuning`:
    'search'   -- full Hyperopt search (hours), then train
    'previous' -- reuse the saved hyperparameters, refresh the feature list (minutes)
    'skip'     -- no tuning step; only allowed once this version's tuning outputs exist
- `run_training` / `run_segmentation`: stop early (e.g. run_training=false to
  only tune).

Training always uses the tuned hyperparameters here; the training notebooks'
HYPERPARAMS_SOURCE='defaults' switch (library defaults, a baseline) is for manual runs.
"""
from datetime import datetime, timedelta
import glob
import os
from airflow import DAG
from airflow.decorators import task
from airflow.exceptions import AirflowFailException
from airflow.models.param import Param
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import PythonOperator
import papermill as pm

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_ROOT = os.environ.get("HOMECREDIT_DATA_ROOT", "/data")
SRC_DIR = os.environ.get("HOMECREDIT_SRC_DIR", "/src")
MODEL_DIR = os.path.join(SRC_DIR, "Training", "04_Model-Development")
HPO_ROOT = os.path.join(DATA_ROOT, "Model-Training", "HyperParameters")
TUNING_PREFIX, TRAINING_PREFIX = "01_tuning", "02_training"

# Model versions = the folders under 04_Model-Development
MODEL_FOLDERS = sorted(
    d for d in (os.listdir(MODEL_DIR) if os.path.isdir(MODEL_DIR) else [])
    if os.path.isdir(os.path.join(MODEL_DIR, d)) and not d.startswith(".")
)


def find_notebook(model_folder: str, prefix: str) -> str:
    """The single <prefix>*.ipynb in 04_Model-Development/<model_folder>/ -- fails if there are none or several."""
    folder = os.path.join(MODEL_DIR, model_folder)
    notebooks = sorted(f for f in os.listdir(folder)
                       if f.startswith(prefix) and f.endswith(".ipynb")) if os.path.isdir(folder) else []
    if len(notebooks) != 1:
        raise AirflowFailException(
            f"Expected exactly one {prefix}*.ipynb in {folder}, found {notebooks or 'none'} -- "
            f"each version folder must hold one {TUNING_PREFIX}*.ipynb and one {TRAINING_PREFIX}*.ipynb.")
    return os.path.join(folder, notebooks[0])


DEFAULT_ARGS = {
    "owner": "Admin",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
    "email_on_failure": False,
    "email_on_retry": False,
}

# Dropdown of discovered versions (free text if the source folders aren't visible at parse time)
_model_folder_choices = {"enum": MODEL_FOLDERS} if MODEL_FOLDERS else {}
UI_PARAMS = {
    "model_folder": Param(
        default="LightGBM_v1",
        type="string",
        title="Model version",
        description="A folder under src/Training/04_Model-Development/. Also names "
        "data/Model-Training/ModelOutput/<model_folder>/, which segmentation reads.",
        **_model_folder_choices,
    ),
    "tuning": Param(
        default="previous",
        type="string",
        enum=["search", "previous", "skip"],
        title="Tuning",
        description="'search': full Hyperopt search (hours). " \
        "'previous' (default): reuse the saved hyperparameters and " \
        "refresh the feature list (minutes) -- needs a previous search. " \
        "'skip': no tuning step; only allowed once this version's tuning outputs exist.",
    ),
    "run_training": Param(
        default=True, type="boolean", title="Train after tuning?",
        description="Set false to only run tuning.",
    ),
    "run_segmentation": Param(
        default=True, type="boolean", title="Run segmentation analysis after training?",
        description="If true (default), runs 05_Segmentation-Analysis once training completes.",
    ),
}


def _execute(task_id: str, input_nb: str, context) -> None:
    """Run one notebook with papermill; the executed copy is kept for audit."""
    out_dir = os.path.join(DATA_ROOT, "Model-Training", "notebook_outputs")
    os.makedirs(out_dir, exist_ok=True)
    run_date = context.get("ds_nodash") or datetime.now().strftime("%Y%m%d")
    model_folder = context["params"]["model_folder"]
    output_nb = os.path.join(out_dir, f"{task_id}__{model_folder}__{run_date}.ipynb")

    print(f"[{task_id}] Running: {input_nb}")
    print(f"[{task_id}] Output : {output_nb}")
    pm.execute_notebook(
        input_path=input_nb,
        output_path=output_nb,
        kernel_name="python3",
        parameters={},  # notebooks read their settings from env vars
        progress_bar=False,
        request_save_on_cell_execute=True,
    )
    print(f"[{task_id}] ✓ Complete")


# ── Task callables ────────────────────────────────────────────────────────────
def check_inputs(**context) -> None:
    """Fail fast, with a clear message, if this run can't work -- before any notebook starts."""
    params = context["params"]
    model_folder, tuning = params["model_folder"], params["tuning"]
    tuning_nb = find_notebook(model_folder, TUNING_PREFIX)
    training_nb = find_notebook(model_folder, TRAINING_PREFIX)
    hpo_dir = os.path.join(HPO_ROOT, model_folder)
    has_hyperparams = bool(glob.glob(os.path.join(hpo_dir, "*_resolved_best.json")))
    has_features = os.path.exists(os.path.join(hpo_dir, "selected_features.json"))
    print(f"{model_folder}: tuning={tuning} | tuning notebook {os.path.basename(tuning_nb)} | "
          f"training notebook {os.path.basename(training_nb)} | saved hyperparameters={has_hyperparams}, "
          f"saved feature list={has_features}")

    if tuning == "previous" and not has_hyperparams:
        raise AirflowFailException(
            f"tuning='previous' needs saved hyperparameters, but {hpo_dir} has no *_resolved_best.json "
            "yet -- run with tuning='search' first.")
    if tuning == "skip" and params["run_training"] and not (has_hyperparams and has_features):
        raise AirflowFailException(
            f"tuning='skip', but {hpo_dir} doesn't have both *_resolved_best.json and "
            "selected_features.json yet -- run with tuning='search' (or 'previous') first.")


def run_tuning_notebook(task_id: str, **context) -> None:
    params = context["params"]
    # Read by the tuning notebook's HYPERPARAMS_SOURCE -- set before execute_notebook so
    # the kernel subprocess inherits it (same pattern as HOMECREDIT_DATA_ROOT)
    os.environ["HPO_HYPERPARAMS_SOURCE"] = params["tuning"]
    _execute(task_id, find_notebook(params["model_folder"], TUNING_PREFIX), context)


def run_training_notebook(task_id: str, **context) -> None:
    # Always the tuned hyperparameters under Airflow (defaults are for manual baseline runs)
    os.environ["HOMECREDIT_HYPERPARAMS"] = "tuned"
    _execute(task_id, find_notebook(context["params"]["model_folder"], TRAINING_PREFIX), context)


def run_segmentation_notebook(task_id: str, **context) -> None:
    """Segmentation_Analysis.ipynb, pointed at this run's model_folder via the same env var
    Credit_Scoring.ipynb uses -- one variable selects which model's test_predictions.parquet
    both notebooks read."""
    os.environ["HOMECREDIT_SCORING_MODEL_FOLDER"] = context["params"]["model_folder"]
    _execute(task_id, os.path.join(SRC_DIR, "Training", "05_Segmentation-Analysis", "Segmentation_Analysis.ipynb"), context)


# ── DAG definition ────────────────────────────────────────────────────────────
with DAG(
    dag_id="04_model_pipeline",
    description="Tune, train and segment one model version -- one linear pipeline",
    default_args=DEFAULT_ARGS,
    # Even with schedule=None, Airflow requires a start_date.
    start_date=datetime(2024, 1, 1),
    schedule=None,  # triggered by 03_mds_creation_and_multicollinearity_test, or manually
    params=UI_PARAMS,
    max_active_runs=1,
    catchup=False,
    tags=["homecredit", "tuning", "training"],
) as dag:

    task_check_inputs = PythonOperator(task_id="check_inputs", python_callable=check_inputs)

    # ── Tuning ──
    @task.branch
    def choose_tuning(**context):
        return "skip_tuning" if context["params"]["tuning"] == "skip" else "tune_model"

    branch_tuning = choose_tuning()
    tune_model = PythonOperator(
        task_id="tune_model",
        python_callable=run_tuning_notebook,
        op_kwargs={"task_id": "tune_model"},
        execution_timeout=timedelta(hours=6),
        doc_md="""
        **Hyperparameter tuning** -- runs this version's tuning notebook: applies its
        feature selection rule (saved to `selected_features.json`) and, with
        tuning='search', a Hyperopt search saved to `*_resolved_best.json`.
        """,
    )
    skip_tuning = EmptyOperator(task_id="skip_tuning")
    # Continues after whichever branch ran -- a skipped task isn't a failure
    tuning_done = EmptyOperator(task_id="tuning_done", trigger_rule="none_failed_min_one_success")

    # ── Training ──
    @task.branch
    def choose_training(**context):
        return "train_model" if context["params"]["run_training"] else "skip_training"

    branch_training = choose_training()
    train_model = PythonOperator(
        task_id="train_model",
        python_callable=run_training_notebook,
        op_kwargs={"task_id": "train_model"},
        doc_md="""
        **Training** -- 5-fold cross-validation on the 80% training split (number of
        trees, threshold), final model on the full 80%, evaluation on the fixed 20% test
        set, MLflow logging and model registration.
        """,
    )
    skip_training = EmptyOperator(task_id="skip_training")

    # ── Segmentation analysis ──
    @task.branch
    def choose_segmentation(**context):
        return "run_segmentation" if context["params"]["run_segmentation"] else "skip_segmentation"

    branch_segmentation = choose_segmentation()
    run_segmentation = PythonOperator(
        task_id="run_segmentation",
        python_callable=run_segmentation_notebook,
        op_kwargs={"task_id": "run_segmentation"},
        doc_md="""
        **Segmentation analysis** -- discrimination (AUC/KS) and calibration within key
        subgroups; writes to `data/Model-Training/ModelOutput/<model_folder>/SegmentationAnalysis/`.
        """,
    )
    skip_segmentation = EmptyOperator(task_id="skip_segmentation")

    task_check_inputs >> branch_tuning >> [tune_model, skip_tuning] >> tuning_done
    tuning_done >> branch_training >> [train_model, skip_training]
    train_model >> branch_segmentation >> [run_segmentation, skip_segmentation]
