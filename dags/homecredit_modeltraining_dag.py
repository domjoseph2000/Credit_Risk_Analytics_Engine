"""
Home Credit – Model Training DAG
=========================================
Orchestrates the full Model Training pipeline for the Home Credit
default-prediction project. Executes multiple candidate models via
papermill in parallel, and funnels them into a champion selection task.
"""
from datetime import datetime, timedelta
import os
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.empty import EmptyOperator
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

# ── Papermill runner ──────────────────────────────────────────────────────────
def run_notebook(task_id: str, model_folder: str, nb_name: str, **context) -> None:
    """
    Execute a notebook dynamically. 
    It accepts the specific model folder and notebook name as arguments
    so we can reuse this exact same function for all 3 models.
    """
    current_nb_dir  = os.path.join(SRC_DIR, "Model-Training", model_folder)
    current_out_dir = os.path.join(DATA_ROOT, "ModelOutput", model_folder)
    
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


# ── Champion Selection ────────────────────────────────────────────────────────
# def select_champion(**context):
#     """
#     Runs after all 3 models finish. This is where you would eventually 
#     add code to query MLflow for the highest AUC and promote the winner!
#     """
#     print("All models finished training!")
#     print("Querying MLflow to compare LightGBM, XGBoost, and Random Forest...")


# ── DAG Definition ────────────────────────────────────────────────────────────
with DAG(
    dag_id="homecredit_model_training",
    default_args=DEFAULT_ARGS,
    # CRITICAL: Even with schedule_interval=None, Airflow requires a start_date.
    start_date=datetime(2024, 1, 1),
    schedule=None, # Triggered by the Prep DAG
    max_active_runs=1,
    catchup=False
) as dag:

    # --- THE FAN OUT (Train 3 models) ---
    task_train_lgb = PythonOperator(
        task_id="train_lightgbm",
        python_callable=run_notebook,
        op_kwargs={
            "task_id": "train_lgb", 
            "model_folder": "LightGBM_v1", 
            "nb_name": "Model_LightGBM.ipynb"
        }
    )

    task_train_xgb = EmptyOperator(
        task_id="train_xgboost",
        # python_callable=run_notebook,
        # op_kwargs={
        #     "task_id": "train_xgb", 
        #     "model_folder": "XGBoost_v1", 
        #     "nb_name": "Model_XGBoost.ipynb" 
        # }
    )

    task_train_rf = EmptyOperator(
        task_id="train_randomforest",
        # python_callable=run_notebook,
        # op_kwargs={
        #     "task_id": "train_rf", 
        #     "model_folder": "RandomForest_v1", 
        #     "nb_name": "Model_RandomForest.ipynb" 
        # }
    )

    # --- THE FAN IN (Compare the results) ---
    # task_select_champion = PythonOperator(
    #     task_id="select_champion_model",
    #     python_callable=select_champion
    # )

    # --- DEPENDENCIES ---
    # The brackets group the 3 models so they all must finish before selection
    # [task_train_lgb, task_train_xgb, task_train_rf] >> task_select_champion