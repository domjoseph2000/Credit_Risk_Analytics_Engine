"""
Home Credit – Preprocessing Pipeline DAG
=========================================
Orchestrates the full feature-engineering pipeline for the Home Credit
default-prediction project. Each task executes one Jupyter notebook via
papermill and writes its output to /data/Model-Training/Staging/.

Dependency graph
----------------

    application_train.csv ──► task_application    ──┐
                                                    │
    previous_application.csv ► task_prevloans     ──┼──► task_pos
                                                    ├──► task_creditcard
                                                    └──► task_installments

    bureau.csv               ► task_bureau  ───────────────┐
    bureau_balance.csv   ────┘                             │
                                                           ▼
    (all above) ─────────────────────────────────────► branch_task
                                                           │
                                    trigger_variable_selection param
                                           ┌───────true───────┐false
                                           ▼                   ▼
                              trigger_variable_selection   skip_variable_selection
                              (runs 02_variable_selection)

Parallel execution
------------------
Stage 1  (no deps)  : task_application, task_prevloans, task_bureau
Stage 2  (needs prevloans) : task_pos, task_creditcard, task_installments
Stage 3  (needs all)      : branch on `trigger_variable_selection` param,
                             which by default triggers the `02_variable_selection`
                             DAG (was `task_select_vars` in this DAG).
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
from airflow.utils.trigger_rule import TriggerRule

import papermill as pm

# ── Paths ─────────────────────────────────────────────────────────────────────
# 1. Define your Root Environment Variables
DATA_ROOT   = os.environ.get("HOMECREDIT_DATA_ROOT", "/data")
SRC_DIR     = os.environ.get("HOMECREDIT_SRC_DIR",   "/src")

# 2. Build your nested paths dynamically
NB_DIR      = os.path.join(SRC_DIR, "Training", "01_Preprocess")

OUTPUT_DIR  = os.path.join(DATA_ROOT, "Model-Training", "Staging")
NB_OUT_DIR  = os.path.join(DATA_ROOT, "Model-Training", "notebook_outputs")

 # Create the output folders
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(NB_OUT_DIR, exist_ok=True)
#os.makedirs(NB_DIR, exist_ok=True)


# Notebook filename map
NOTEBOOKS = {
    "application"   : "Application_data.ipynb",
    "prevloans"     : "Prevloans_data.ipynb",
    "pos"           : "POS_data.ipynb",
    "creditcard"    : "CreditCard_data.ipynb",
    "installments"  : "Payment_installments_data.ipynb",
    "bureau"        : "Bureau_data.ipynb",
}

# ── Default args ──────────────────────────────────────────────────────────────
DEFAULT_ARGS = {
    "owner"           : "Admin",
    "depends_on_past" : False,
    "retries"         : 1,
    "retry_delay"     : timedelta(minutes=5),
    "email_on_failure": False,
    "email_on_retry"  : False,
}

UI_PARAMS = {
    "trigger_variable_selection": Param(
        default=True,
        type="boolean",
        title="Trigger variable selection?",
        description="If true (default), triggers the 02_variable_selection DAG once "
        "all preprocessing tasks finish. Set to false to stop after preprocessing.",
    )
}

# ── Papermill runner ──────────────────────────────────────────────────────────
def run_notebook(task_id: str, **context) -> None:
    """
    Execute a notebook with papermill.

    The notebook is expected to self-configure from environment variables
    set inside the Docker container:  HOMECREDIT_DATA_ROOT  – project root that replaces hard-coded path 'a:\\Dominic\\My Projects\\Home Credit'
    Executed notebooks are saved to NB_OUT_DIR for auditing / debugging.
    """
    nb_name = NOTEBOOKS[task_id]

    input_nb   = os.path.join(NB_DIR, nb_name)
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
    dag_id="01_preprocessing",
    description="Home Credit feature engineering pipeline",
    default_args=DEFAULT_ARGS,
    start_date=datetime(2024, 1, 1),
    schedule=None,          # triggered manually or by upstream event
    params=UI_PARAMS,
    catchup=False,
    max_active_runs=1,
    tags=["homecredit", "preprocessing", "feature-engineering"],
) as dag:

    # ── Stage 1: independent tasks (run in parallel) ──────────────────────────

    task_application = PythonOperator(
        task_id="task_application",
        python_callable=run_notebook,
        op_kwargs={"task_id": "application"},
        doc_md="""
        **Application Data**
        Reads `application_train.csv`, cleans sentinel values (365243, XNA),
        and engineers financial ratios, demographic features, housing indices,
        and binary property flags.
        Output: `application_train_data_processed.parquet`
        """,
    )

    task_prevloans = PythonOperator(
        task_id="task_prevloans",
        python_callable=run_notebook,
        op_kwargs={"task_id": "prevloans"},
        doc_md="""
        **Previous Loans**
        Reads `previous_application.csv`, cleans sentinels, derives credit
        ratios, splits into Approved/Refused subsets, and aggregates to
        customer level across four time windows.
        Outputs:
        - `prev_loan_data_processed.parquet`  (loan-level metadata used by POS/CC/Payments)
        - `prev_loans_aggregate_final_processed.parquet`
        """,
    )

    task_bureau = PythonOperator(
        task_id="task_bureau",
        python_callable=run_notebook,
        op_kwargs={"task_id": "bureau"},
        doc_md="""
        **Bureau & Bureau Balance**
        Reads `bureau.csv` and `bureau_balance.csv`, imputes credit-sum
        values, aggregates bureau balance to loan level across four time
        windows, and rolls up to customer level.
        Output: `bureau_processed.parquet`
        """,
    )

    # ── Stage 2: depend on task_prevloans (run in parallel with each other) ───

    task_pos = PythonOperator(
        task_id="task_pos",
        python_callable=run_notebook,
        op_kwargs={"task_id": "pos"},
        doc_md="""
        **POS Cash Balance**
        Reads `POS_CASH_balance.csv` + `prev_loan_data_processed.parquet`.
        Aggregates DPD features over all-time, 1Y, and 3M windows; derives
        prepayment flag and months-since-last-DPD.
        Output: `posdata_processed.parquet`
        """,
    )

    task_creditcard = PythonOperator(
        task_id="task_creditcard",
        python_callable=run_notebook,
        op_kwargs={"task_id": "creditcard"},
        doc_md="""
        **Credit Card Balance**
        Reads `credit_card_balance.csv` + `prev_loan_data_processed.parquet`.
        Computes utilisation/payment ratios, QoQ changes, DPD flags, and
        aggregates over 3Y/1Y/6M/3M windows.
        Output: `creditcard_processed.parquet`
        """,
    )

    task_installments = PythonOperator(
        task_id="task_installments",
        python_callable=run_notebook,
        op_kwargs={"task_id": "installments"},
        doc_md="""
        **Instalment Payments**
        Reads `installments_payments.csv` + `prev_loan_data_processed.parquet`.
        Derives late-payment days, payment ratios, instalment frequency, and
        aggregates over 1Y and 3M windows.
        Output: `payment_installments_processed.parquet`
        """,
    )

    # ── Stage 3: branch on whether to trigger variable selection ─────────────

    @task.branch(trigger_rule=TriggerRule.ALL_SUCCESS)
    def choose_path(**context):
        if context["params"]["trigger_variable_selection"]:
            return "trigger_variable_selection"
        return "skip_variable_selection"

    branch_task = choose_path()

    trigger_variable_selection = TriggerDagRunOperator(
        task_id="trigger_variable_selection",
        trigger_dag_id="02_variable_selection",
        wait_for_completion=False,
    )

    skip_variable_selection = EmptyOperator(task_id="skip_variable_selection")

    # ── Wire up dependencies ──────────────────────────────────────────────────
    #
    #  Stage 1 (parallel)         Stage 2 (parallel)        Stage 3
    #  ──────────────────         ──────────────────        ───────
    #  task_application ──────────────────────────────────► branch_task
    #  task_prevloans   ──► task_pos           ───────────► branch_task
    #                   └──► task_creditcard   ───────────► branch_task
    #                   └──► task_installments ───────────► branch_task
    #  task_bureau ───────────────────────────────────────► branch_task
    #  branch_task ──► trigger_variable_selection | skip_variable_selection
    # 1. Independent tables run straight to the branch
    [task_application, task_bureau] >> branch_task

    # 2. Dependent tables run through their sequence
    task_prevloans >> [task_pos, task_creditcard, task_installments] >> branch_task

    # 3. Branch to either trigger or skip variable selection
    branch_task >> [trigger_variable_selection, skip_variable_selection]
