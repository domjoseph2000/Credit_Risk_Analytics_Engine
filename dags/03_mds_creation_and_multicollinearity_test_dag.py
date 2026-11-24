"""
Home Credit – 03 MDS Creation & Multicollinearity DAG
=====================================================
Step 3 of the training pipeline: builds the single table the models are
trained on (the "model dataset", MDS), removes features that mostly repeat
each other, then starts model training.

Tasks
-----
1. `create_mds` — runs 03_MDS-Creation/model_dataset_creation.ipynb. Joins
   every processed parquet from Step 1 into one row per applicant
   (SK_ID_CURR) and saves it as model_dataset_initial.parquet.
2. `multicollinearity_reduction` — runs
   02_Variable-Selection/01b_multicollinearity_reduction.ipynb. Repeatedly
   drops the feature with the highest VIF (the one best explained by the
   others), but protects features that scored well in Step 2. Saves the
   result as model_dataset_final.parquet, which tuning and training read.
3. Starts 04_model_pipeline for LightGBM_v1, the version that uses every
   feature in the new dataset (later versions pick their features from it):
   - `run_hyperopt` on  -> a full hyperparameter search ('search', hours)
   - `run_hyperopt` off -> reuse the saved hyperparameters and just refresh
     the feature list for the new dataset ('previous', minutes). Default.

The other model versions aren't retrained automatically; run them yourself
in 04_model_pipeline afterwards.
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
# Inside the container: data/ is mounted at /data and src/ at /src.
DATA_ROOT   = os.environ.get("HOMECREDIT_DATA_ROOT", "/data")
SRC_DIR     = os.environ.get("HOMECREDIT_SRC_DIR",   "/src")

# Settings shared by every task: one retry after 5 minutes, no emails
DEFAULT_ARGS = {
    "owner"           : "Admin",
    "depends_on_past" : False,
    "retries"         : 1,
    "retry_delay"     : timedelta(minutes=5),
    "email_on_failure": False,
    "email_on_retry"  : False,
}

# Options shown on the "Trigger DAG" screen in the Airflow UI
UI_PARAMS = {
        "run_hyperopt": Param(
            default=False,
            type="boolean",
            description="After MDS creation, trigger 04_model_pipeline for LightGBM_v1 with a full "
            "hyperparameter search (true) or with tuning='previous' (false, default): reuse the saved "
            "hyperparameters and refresh the feature list for the new dataset, then train."
    )
}

# Which notebook each task runs: task id -> (folder under src/Training, notebook file)
NOTEBOOKS = {
    "create_mds"                  : ("03_MDS-Creation", "model_dataset_creation.ipynb"),
    "multicollinearity_reduction" : ("02_Variable-Selection", "01b_multicollinearity_reduction.ipynb"),
}

# ── Notebook runner ───────────────────────────────────────────────────────────
def run_notebook(task_id: str, **context) -> None:
    """
    Run the notebook listed for `task_id` in NOTEBOOKS with papermill.

    Both notebooks find their own input/output folders from environment
    variables, so no parameters are passed. The executed copy is saved as
    notebook_outputs/<task>__<run date>.ipynb.
    """
    subfolder, nb_name = NOTEBOOKS[task_id]
    current_nb_dir  = os.path.join(SRC_DIR, "Training", subfolder)
    current_out_dir = os.path.join(DATA_ROOT, "Model-Training", "notebook_outputs")

    # Make sure the output folder exists before papermill writes to it
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
        parameters={},          # none needed -- the notebooks read their paths from env vars
        progress_bar=False,
        request_save_on_cell_execute=True,  # save after every cell, so a failed run still shows how far it got
    )
    print(f"[{task_id}] ✓ Complete")


# ── DAG definition ────────────────────────────────────────────────────────────
with DAG(
    dag_id="03_mds_creation_and_multicollinearity_test",
    description="MDS creation + VIF-based multicollinearity reduction, then branch to tuning/training",
    default_args=DEFAULT_ARGS,
    # Airflow requires a start date even for a DAG with no schedule
    start_date=datetime(2024, 1, 1),
    params=UI_PARAMS,
    schedule=None, # started by 02_variable_selection, or manually
    max_active_runs=1,
    catchup=False
) as dag:

    # 1. Build the model dataset (doc_md is the task description in the Airflow UI)
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

    # 2. Remove features that largely duplicate others
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

    # 3. Choose how to train LightGBM_v1 on the new dataset. Both options run
    # the full model pipeline (tuning -> training -> segmentation); they only
    # differ in the tuning mode. Returns the id of the trigger task to run.
    @task.branch
    def choose_path(**context):
        if context["params"]["run_hyperopt"]:
            return "trigger_pipeline_search"
        else:
            return "trigger_pipeline_previous"

    branch_task = choose_path()

    # 4a. Full hyperparameter search, then train (doesn't wait for 04 to finish)
    trigger_tuning_pipeline = TriggerDagRunOperator(
        task_id="trigger_pipeline_search",
        trigger_dag_id="04_model_pipeline",
        conf={"model_folder": "LightGBM_v1", "tuning": "search"},
        wait_for_completion=False,
    )

    # 4b. Reuse the saved hyperparameters (refreshing the feature list), then train
    trigger_training_pipeline = TriggerDagRunOperator(
        task_id="trigger_pipeline_previous",
        trigger_dag_id="04_model_pipeline",
        conf={"model_folder": "LightGBM_v1", "tuning": "previous"},
    )

    # Task order
    task_create_mds >> task_multicollinearity_reduction >> branch_task >> [trigger_tuning_pipeline, trigger_training_pipeline]
