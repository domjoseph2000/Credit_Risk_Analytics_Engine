"""
Home Credit – 02 Variable Selection DAG
=======================================
Step 2 of the training pipeline: measures how useful each engineered feature
is for predicting default, so weak or redundant features can be dropped later.

What it does
------------
Runs src/Training/02_Variable-Selection/select_vars.ipynb. The notebook loads
the processed parquets from Step 1 (Model-Training/Staging/) and, for every
feature, calculates: AUC, a t-test, Cohen's d, a one-feature logistic
regression coefficient, VIF (overlap with other features) and correlation
with the target. The scores are written to
data/Model-Training/Variable-Selection/numeric_selection_metrics_full_df.xlsx.

How it's started
----------------
- Automatically by 01_preprocessing when that finishes (unless its
  `trigger_variable_selection` param is false).
- Or on its own, e.g. after changing select_vars.ipynb:

    docker compose exec pipeline airflow dags trigger 02_variable_selection

What happens next
-----------------
When the notebook finishes, this DAG starts
03_mds_creation_and_multicollinearity_test, unless you set the
`trigger_mds_creation` param to false.

It's a separate DAG (it used to be a task inside 01_preprocessing) so it can
be rerun or skipped without redoing preprocessing.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta

from airflow import DAG
from airflow.decorators import task
from airflow.models.param import Param
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import PythonOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator

import papermill as pm

# ── Paths ─────────────────────────────────────────────────────────────────────
# Inside the container: data/ is mounted at /data and src/ at /src.
DATA_ROOT   = os.environ.get("HOMECREDIT_DATA_ROOT", "/data")
SRC_DIR     = os.environ.get("HOMECREDIT_SRC_DIR",   "/src")

VAR_SEL_DIR = os.path.join(SRC_DIR, "Training", "02_Variable-Selection")   # notebook to run
NB_OUT_DIR  = os.path.join(DATA_ROOT, "Model-Training", "notebook_outputs")  # executed notebook copies

os.makedirs(NB_OUT_DIR, exist_ok=True)

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
    "trigger_mds_creation": Param(
        default=True,
        type="boolean",
        title="Trigger MDS creation?",
        description="If true (default), triggers the "
        "03_mds_creation_and_multicollinearity_test DAG once variable "
        "selection finishes. Set to false to stop after variable selection.",
    )
}


# ── Notebook runner ───────────────────────────────────────────────────────────
def run_notebook(task_id: str, **context) -> None:
    """
    Run select_vars.ipynb with papermill.

    No parameters are passed in: the notebook finds its folders itself from
    the HOMECREDIT_DATA_ROOT environment variable. The executed copy is saved
    as notebook_outputs/<task>__<run date>.ipynb.
    """
    input_nb   = os.path.join(VAR_SEL_DIR, "select_vars.ipynb")
    run_date   = context.get("ds_nodash") or datetime.now().strftime("%Y%m%d")
    output_nb  = os.path.join(NB_OUT_DIR, f"{task_id}__{run_date}.ipynb")

    print(f"[{task_id}] Running: {input_nb}")
    print(f"[{task_id}] Output : {output_nb}")

    pm.execute_notebook(
        input_path=input_nb,
        output_path=output_nb,
        kernel_name="python3",
        parameters={},          # none needed -- the notebook reads its paths from env vars
        progress_bar=False,
        request_save_on_cell_execute=True,  # save after every cell, so a failed run still shows how far it got
    )
    print(f"[{task_id}] ✓ Complete")


# ── DAG definition ────────────────────────────────────────────────────────────
with DAG(
    dag_id="02_variable_selection",
    description="Score engineered features and write selection metrics to Excel",
    default_args=DEFAULT_ARGS,
    start_date=datetime(2024, 1, 1),
    schedule=None,           # started by 01_preprocessing, or manually
    params=UI_PARAMS,
    catchup=False,
    max_active_runs=1,       # never two runs at once
    tags=["homecredit", "variable-selection", "feature-engineering"],
) as dag:

    # doc_md is the description shown for the task in the Airflow UI
    task_select_vars = PythonOperator(
        task_id="select_vars",
        python_callable=run_notebook,
        op_kwargs={"task_id": "select_vars"},
        doc_md="""
        **Variable Selection**
        Loads all processed parquets, computes per-feature AUC, t-test,
        Cohen's d, univariate logistic coefficient, VIF, and target
        correlation. Scores each feature and writes results to Excel.
        Output: `data/Model-Training/Variable-Selection/numeric_selection_metrics_full_df.xlsx`
        """,
    )

    # ── Decide whether to start MDS creation ──────────────────────────────────
    # Returns the id of the task to run next; Airflow skips the other one.

    @task.branch
    def choose_path(**context):
        if context["params"]["trigger_mds_creation"]:
            return "trigger_mds_creation"
        return "skip_mds_creation"

    branch_task = choose_path()

    # Starts the 03 DAG and finishes straight away, without waiting for it
    trigger_mds_creation = TriggerDagRunOperator(
        task_id="trigger_mds_creation",
        trigger_dag_id="03_mds_creation_and_multicollinearity_test",
        wait_for_completion=False,
    )

    # Does nothing -- just marks "stopped after variable selection" in the graph
    skip_mds_creation = EmptyOperator(task_id="skip_mds_creation")

    task_select_vars >> branch_task >> [trigger_mds_creation, skip_mds_creation]
