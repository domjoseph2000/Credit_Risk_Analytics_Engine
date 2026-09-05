"""
Home Credit – Hyperparameter Tuning DAG
=========================================
Runs a single hyperparameter-tuning notebook, selected via DAG run params
(model_folder + notebook_name) instead of a fixed per-model fan-out. This
project tunes one model family per run, not several in parallel, so the
old static LightGBM/XGBoost/RandomForest fan-out was two dead EmptyOperator
stubs for every one real notebook. Point this DAG at any notebook under
src/Hyperparameter-Tuning/<model_folder>/<notebook_name> by setting those
params on the trigger screen (or via --conf) -- no DAG edit needed to
tune a different model version.
"""
from datetime import datetime, timedelta
import os
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.empty import EmptyOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.models.param import Param
from airflow.decorators import task
import papermill as pm

execution_timeout = timedelta(hours=6)

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_ROOT = os.environ.get("HOMECREDIT_DATA_ROOT", "/data")
SRC_DIR = os.environ.get("HOMECREDIT_SRC_DIR", "/src")

DEFAULT_ARGS = {
    "owner": "Admin",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
    "email_on_failure": False,
    "email_on_retry": False,
}

UI_PARAMS = {
    "model_folder": Param(
        default="LightGBM_v1",
        type="string",
        title="Model folder",
        description="Subfolder under src/Hyperparameter-Tuning/ containing the tuning notebook.",
    ),
    "notebook_name": Param(
        default="LightGBM_hyperpars_tuning.ipynb",
        type="string",
        title="Notebook name",
        description="Notebook filename (inside model_folder) to execute for tuning.",
    ),
    "trigger_training": Param(
        default=True,
        type="boolean",
        title="Trigger training after tuning?",
        description="If true, triggers 05_model_training once tuning completes.",
    ),
    "use_previous_best": Param(
        default=False,
        type="boolean",
        title="Use previous best params?",
        description="Maps to the notebook's USE_PREVIOUS_BEST flag (read from the "
        "HOMECREDIT_USE_PREVIOUS_BEST env var, same self-configuration pattern as "
        "HOMECREDIT_DATA_ROOT/HOMECREDIT_SRC_DIR). If true, skips the live Hyperopt "
        "search and loads the previously-saved lgbm_resolved_best.json instead. If "
        "false (default), runs the full live search. Only takes effect for notebooks "
        "that read HOMECREDIT_USE_PREVIOUS_BEST (currently just LightGBM_v1's tuning "
        "notebook) -- it's a harmless no-op env var for any other.",
    ),
}


# ── Papermill runner ──────────────────────────────────────────────────────────
def run_notebook(task_id: str, **context) -> None:
    """
    Execute the tuning notebook named by this run's model_folder /
    notebook_name params.
    """
    params = context["params"]
    model_folder = params["model_folder"]
    nb_name = params["notebook_name"]
    use_previous_best = params["use_previous_best"]

    current_nb_dir = os.path.join(SRC_DIR, "Hyperparameter-Tuning", model_folder)
    current_out_dir = os.path.join(DATA_ROOT, "Model-Training", "notebook_outputs")

    # Ensure the output folder exists before writing to prevent OS errors
    os.makedirs(current_out_dir, exist_ok=True)

    input_nb = os.path.join(current_nb_dir, nb_name)
    run_date = context.get("ds_nodash") or datetime.now().strftime("%Y%m%d")
    output_nb = os.path.join(current_out_dir, f"{task_id}__{run_date}.ipynb")

    print(f"[{task_id}] Running: {input_nb}")
    print(f"[{task_id}] Output : {output_nb}")
    print(f"[{task_id}] HOMECREDIT_USE_PREVIOUS_BEST: {use_previous_best}")

    # Set before execute_notebook so the notebook's kernel subprocess inherits it --
    # same self-configuration pattern as HOMECREDIT_DATA_ROOT/HOMECREDIT_SRC_DIR,
    # rather than papermill's parameters= cell-injection mechanism.
    os.environ["HPO_USE_PREVIOUS_BEST"] = str(use_previous_best)

    pm.execute_notebook(
        input_path=input_nb,
        output_path=output_nb,
        kernel_name="python3",
        parameters={},  # notebooks read paths (and USE_PREVIOUS_BEST) from env vars
        progress_bar=False,
        request_save_on_cell_execute=True,
    )
    print(f"[{task_id}] ✓ Complete")


# ── DAG Definition ────────────────────────────────────────────────────────────
with DAG(
    dag_id="04_hyperparameter_tuning",
    default_args=DEFAULT_ARGS,
    # CRITICAL: Even with schedule_interval=None, Airflow requires a start_date.
    start_date=datetime(2024, 1, 1),
    schedule=None,  # Triggered by the Prep DAG, or manually
    params=UI_PARAMS,
    max_active_runs=1,
    catchup=False,
) as dag:

    task_tune_model = PythonOperator(
        task_id="tune_model",
        python_callable=run_notebook,
        execution_timeout=execution_timeout,
        op_kwargs={"task_id": "tune_model"},
    )

    @task.branch
    def check_if_should_train(**context):
        if context["params"]["trigger_training"]:
            return "trigger_model_training"
        return "skip_training"

    branch_task = check_if_should_train()
    skip_training = EmptyOperator(task_id="skip_training")

    # Forward this run's model_folder to the training DAG so the model you
    # just tuned is the one that gets trained -- notebook_name isn't
    # forwarded since the training notebook for a model version is usually
    # named differently from its tuning notebook (e.g. Model_LightGBM.ipynb
    # vs LightGBM_hyperpars_tuning.ipynb); override it on the training
    # trigger screen if needed.
    trigger_training_pipeline = TriggerDagRunOperator(
        task_id="trigger_model_training",
        trigger_dag_id="05_model_training",
        conf={"model_folder": "{{ params.model_folder }}"},
        wait_for_completion=False,  # Set to True if you want tuning to wait until training finishes
    )

    # Dependencies
    task_tune_model >> branch_task >> [trigger_training_pipeline, skip_training]
