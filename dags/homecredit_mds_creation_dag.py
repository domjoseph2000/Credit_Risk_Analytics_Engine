"""
Home Credit – Model Training DAG
=========================================
Orchestrates the full HyperParameterTuning pipeline for the Home Credit
default-prediction project. Executes multiple candidate models via
papermill in parallel, and funnels them into a champion selection task.
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
# ── Papermill runner ──────────────────────────────────────────────────────────
def run_notebook(task_id: str, nb_name: str, **context) -> None:
    """
    Execute a notebook dynamically. 
    It accepts the specific model folder and notebook name as arguments
    so we can reuse this exact same function for all 3 models.
    """
    current_nb_dir  = os.path.join(SRC_DIR, "MDS-Creation")
    current_out_dir = os.path.join(DATA_ROOT, "notebook_outputs")
    
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
    dag_id="homecredit_create_model_dataset",
    default_args=DEFAULT_ARGS,
    # CRITICAL: Even with schedule_interval=None, Airflow requires a start_date.
    start_date=datetime(2024, 1, 1),
    params=UI_PARAMS,
    schedule=None, # Triggered by the Prep DAG
    max_active_runs=1,
    catchup=False
) as dag:

    # 1. MDS creation task
    task_create_mds = PythonOperator(
        task_id="create_mds",
        python_callable=run_notebook,
        op_kwargs={
            "task_id": "create_mds", 
            "nb_name": "model_dataset_creation.ipynb"
        }
    )

    # 2. The Branching Logic
    @task.branch
    def choose_path(**context):
        if context["params"]["run_hyperopt"]:
            return "trigger_model_tuning_dag"
        else:
            return "trigger_training_dag"

    branch_task = choose_path()

    # 3. Path A: Trigger Hyperopt
    # This goes at the bottom of your Data Prep DAG
    trigger_tuning_pipeline = TriggerDagRunOperator(
        task_id="trigger_model_tuning_dag",
        trigger_dag_id="homecredit_model_tuning", # The exact name of model training DAG
        wait_for_completion=False # Set to True if you want Prep to wait until Training finishes
    )

    # 4. Path B: Trigger Training directly
    trigger_training_pipeline = TriggerDagRunOperator(
        task_id="trigger_training_dag",
        trigger_dag_id="homecredit_model_training"
    )



    # Dependencies in Data Prep DAG
    task_create_mds >> branch_task >> [trigger_tuning_pipeline, trigger_training_pipeline]