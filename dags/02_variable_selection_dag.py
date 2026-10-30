"""
Home Credit – Variable Selection DAG
=========================================
Runs `select_vars.ipynb`, which scores every engineered feature (AUC,
t-test, Cohen's d, univariate logistic coefficient, VIF, target correlation)
across the processed parquets staged by the preprocessing DAG, and writes
the scored results to Excel.

Split out of `01_preprocessing` (was that DAG's `task_select_vars`) so
variable selection can be retried, re-run, or skipped independently of the
upstream feature-engineering tasks.

Triggered automatically by `01_preprocessing` once its feature tasks finish,
unless that DAG's `trigger_variable_selection` param is set to false. Can
also be run standalone, e.g.:

    docker compose exec pipeline airflow dags trigger 02_variable_selection

By default, once `select_vars.ipynb` finishes, this DAG triggers
`03_mds_creation_and_multicollinearity_test` in turn -- set the
`trigger_mds_creation` param to false to stop after variable selection
instead.
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
DATA_ROOT   = os.environ.get("HOMECREDIT_DATA_ROOT", "/data")
SRC_DIR     = os.environ.get("HOMECREDIT_SRC_DIR",   "/src")

VAR_SEL_DIR = os.path.join(SRC_DIR, "Training", "02_Variable-Selection")
NB_OUT_DIR  = os.path.join(DATA_ROOT, "Model-Training", "notebook_outputs")

os.makedirs(NB_OUT_DIR, exist_ok=True)

DEFAULT_ARGS = {
    "owner"           : "Admin",
    "depends_on_past" : False,
    "retries"         : 1,
    "retry_delay"     : timedelta(minutes=5),
    "email_on_failure": False,
    "email_on_retry"  : False,
}

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


# ── Papermill runner ──────────────────────────────────────────────────────────
def run_notebook(task_id: str, **context) -> None:
    """
    Execute `select_vars.ipynb` with papermill.

    The notebook self-configures from environment variables set inside the
    Docker container. Executed notebook is saved to NB_OUT_DIR for
    auditing / debugging.
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
        parameters={},          # notebooks read paths from env vars
        progress_bar=False,
        request_save_on_cell_execute=True,
    )
    print(f"[{task_id}] ✓ Complete")


# ── DAG definition ────────────────────────────────────────────────────────────
with DAG(
    dag_id="02_variable_selection",
    description="Score engineered features and write selection metrics to Excel",
    default_args=DEFAULT_ARGS,
    start_date=datetime(2024, 1, 1),
    schedule=None,           # triggered by 01_preprocessing, or manually
    params=UI_PARAMS,
    catchup=False,
    max_active_runs=1,
    tags=["homecredit", "variable-selection", "feature-engineering"],
) as dag:

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

    # ── Branch on whether to trigger MDS creation ─────────────────────────────

    @task.branch
    def choose_path(**context):
        if context["params"]["trigger_mds_creation"]:
            return "trigger_mds_creation"
        return "skip_mds_creation"

    branch_task = choose_path()

    trigger_mds_creation = TriggerDagRunOperator(
        task_id="trigger_mds_creation",
        trigger_dag_id="03_mds_creation_and_multicollinearity_test",
        wait_for_completion=False,
    )

    skip_mds_creation = EmptyOperator(task_id="skip_mds_creation")

    task_select_vars >> branch_task >> [trigger_mds_creation, skip_mds_creation]
