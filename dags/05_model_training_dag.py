"""
Home Credit – Model Training DAG
=========================================
Runs a single model-training notebook, selected via DAG run params
(model_folder + notebook_name) instead of a fixed fan-out across model
families. This project trains one model version per run, not several in
parallel, so the old static LightGBM/XGBoost/RandomForest fan-out was two
dead EmptyOperator stubs for every one real notebook, with champion
selection between them never implemented. Point this DAG at any notebook
under src/Model-Training/<model_folder>/<notebook_name> by setting those
params on the trigger screen (or via --conf) -- e.g. model_folder=
"LightGBM_v3", notebook_name="Model_LightGBM_v3.ipynb" for the current
production model, without editing this file.
"""
from datetime import datetime, timedelta
import os
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.models.param import Param
import papermill as pm

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
        description="Subfolder under src/Model-Training/ containing the training notebook.",
    ),
    "notebook_name": Param(
        default="Model_LightGBM.ipynb",
        type="string",
        title="Notebook name",
        description="Notebook filename (inside model_folder) to execute for training.",
    ),
}


# ── Papermill runner ──────────────────────────────────────────────────────────
def run_notebook(task_id: str, **context) -> None:
    """
    Execute the training notebook named by this run's model_folder /
    notebook_name params.
    """
    params = context["params"]
    model_folder = params["model_folder"]
    nb_name = params["notebook_name"]

    current_nb_dir = os.path.join(SRC_DIR, "Model-Training", model_folder)
    current_out_dir = os.path.join(DATA_ROOT, "Model-Training", "notebook_outputs")

    # Ensure the output folder exists before writing to prevent OS errors
    os.makedirs(current_out_dir, exist_ok=True)

    input_nb = os.path.join(current_nb_dir, nb_name)
    run_date = context.get("ds_nodash") or datetime.now().strftime("%Y%m%d")
    output_nb = os.path.join(current_out_dir, f"{task_id}__{run_date}.ipynb")

    print(f"[{task_id}] Running: {input_nb}")
    print(f"[{task_id}] Output : {output_nb}")

    pm.execute_notebook(
        input_path=input_nb,
        output_path=output_nb,
        kernel_name="python3",
        parameters={},  # notebooks read paths from env vars
        progress_bar=False,
        request_save_on_cell_execute=True,
    )
    print(f"[{task_id}] ✓ Complete")


# ── DAG Definition ────────────────────────────────────────────────────────────
with DAG(
    dag_id="05_model_training",
    default_args=DEFAULT_ARGS,
    # CRITICAL: Even with schedule_interval=None, Airflow requires a start_date.
    start_date=datetime(2024, 1, 1),
    schedule=None,  # Triggered by the tuning/MDS DAG, or manually
    params=UI_PARAMS,
    max_active_runs=1,
    catchup=False,
) as dag:

    task_train_model = PythonOperator(
        task_id="train_model",
        python_callable=run_notebook,
        op_kwargs={"task_id": "train_model"},
    )
