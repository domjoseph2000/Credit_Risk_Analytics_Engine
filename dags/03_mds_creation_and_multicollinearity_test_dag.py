"""
Home Credit – MDS Creation & Multicollinearity Test DAG
=========================================
Orchestrates model dataset (MDS) creation followed by VIF-based
multicollinearity reduction, then funnels into hyperparameter tuning or
straight to training.

**Pipeline**
1. `create_mds` — runs `model_dataset_creation.ipynb`, joining every staged
   parquet into `model_dataset_initial.parquet`.
2. `multicollinearity_reduction` — runs `01b_multicollinearity_reduction.ipynb`,
   which iteratively drops the highest-VIF feature (protecting features with
   strong predictive scores from `02_variable_selection`'s output) and writes
   the reduced `model_dataset_final.parquet` that hyperparameter tuning and
   training consume. This step used to be a manual, undocumented step run
   outside any DAG — it's now wired in so the full pipeline runs end-to-end.
3. Branches to hyperparameter tuning or straight to training based on a UI
   toggle (`run_hyperopt`).
"""
from datetime import datetime, timedelta
import os
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.empty import EmptyOperator
from airflow.models.param import Param
from airflow.decorators import task
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
import papermill as pm

# ── Paths ─────────────────────────────────────────────────────────────────────
# 1. Define your Root Environment Variables
DATA_ROOT   = os.environ.get("HOMECREDIT_DATA_ROOT", "/data")
SRC_DIR     = os.environ.get("HOMECREDIT_SRC_DIR",   "/src")

DEFAULT_ARGS = {
    "owner"           : "Admin",
    "depends_on_past" : False,
    "retries"         : 1,
    "retry_delay"     : timedelta(minutes=5),
    "email_on_failure": False,
    "email_on_retry"  : False,
}

UI_PARAMS = {
        "run_hyperopt": Param(
            default=False,
            type="boolean",
            description="Perform Hyperparameter tuning after MDS creation? If False, the pipeline will proceed to model training using default hyperparameters"
    )
}

# Notebook filename map: task_id -> (subfolder under SRC_DIR, notebook filename)
NOTEBOOKS = {
    "create_mds"                  : ("MDS-Creation", "model_dataset_creation.ipynb"),
    "multicollinearity_reduction" : ("Variable-Selection", "01b_multicollinearity_reduction.ipynb"),
}

# ── Papermill runner ──────────────────────────────────────────────────────────
def run_notebook(task_id: str, **context) -> None:
    """
    Execute a notebook dynamically, looked up from NOTEBOOKS by task_id.
    Reused across MDS creation and multicollinearity reduction since both are
    single fixed notebooks that self-configure from env vars.
    """
    subfolder, nb_name = NOTEBOOKS[task_id]
    current_nb_dir  = os.path.join(SRC_DIR, subfolder)
    current_out_dir = os.path.join(DATA_ROOT, "Model-Training", "notebook_outputs")

    # Ensure the output folder exists before writing to prevent OS errors
    os.makedirs(current_out_dir, exist_ok=True)

    input_nb   = os.path.join(current_nb_dir, nb_name)
    run_date   = context.get("ds_nodash") or datetime.now().strftime("%Y%m%d")
    output_nb  = os.path.join(current_out_dir, f"{task_id}__{run_date}.ipynb")

    print(f"[{task_id}] Running: {input_nb}")
    print(f"[{task_id}] Output : {output_nb}")

    pm.execute_notebook(
        input_path=input_nb,
        output_path=output_nb,
        kernel_name="python3",
        parameters={},          # notebooks read paths from env vars
        progress_bar=False,
        request_save_on_cell_execute=True,
    )
    print(f"[{task_id}] ✓ Complete")


# ── DAG Definition ────────────────────────────────────────────────────────────
with DAG(
    dag_id="03_mds_creation_and_multicollinearity_test",
    description="MDS creation + VIF-based multicollinearity reduction, then branch to tuning/training",
    default_args=DEFAULT_ARGS,
    # CRITICAL: Even with schedule_interval=None, Airflow requires a start_date.
    start_date=datetime(2024, 1, 1),
    params=UI_PARAMS,
    schedule=None, # Triggered by the Variable Selection DAG
    max_active_runs=1,
    catchup=False
) as dag:

    # 1. MDS creation task
    task_create_mds = PythonOperator(
        task_id="create_mds",
        python_callable=run_notebook,
        op_kwargs={"task_id": "create_mds"},
        doc_md="""
        **MDS Creation**
        Joins every processed parquet under `Staging/` on `SK_ID_CURR`,
        downcasts dtypes, and writes `model_dataset_initial.parquet`.
        """,
    )

    # 2. Multicollinearity reduction task
    task_multicollinearity_reduction = PythonOperator(
        task_id="multicollinearity_reduction",
        python_callable=run_notebook,
        op_kwargs={"task_id": "multicollinearity_reduction"},
        doc_md="""
        **Multicollinearity Reduction (VIF Elimination)**
        Reads `model_dataset_initial.parquet`, iteratively drops the
        highest-VIF feature (protecting features with strong predictive
        scores from `02_variable_selection`'s output), and writes
        `model_dataset_final.parquet` — the dataset hyperparameter tuning
        and training consume.
        """,
    )

    # 3. The Branching Logic
    @task.branch
    def choose_path(**context):
        if context["params"]["run_hyperopt"]:
            return "trigger_model_tuning_dag"
        else:
            return "trigger_training_dag"

    branch_task = choose_path()

    # 4. Path A: Trigger Hyperopt
    trigger_tuning_pipeline = TriggerDagRunOperator(
        task_id="trigger_model_tuning_dag",
        trigger_dag_id="04_hyperparameter_tuning",
        wait_for_completion=False # Set to True if you want Prep to wait until Training finishes
    )

    # 5. Path B: Trigger Training directly
    trigger_training_pipeline = TriggerDagRunOperator(
        task_id="trigger_training_dag",
        trigger_dag_id="05_model_training"
    )

    # Dependencies
    task_create_mds >> task_multicollinearity_reduction >> branch_task >> [trigger_tuning_pipeline, trigger_training_pipeline]
