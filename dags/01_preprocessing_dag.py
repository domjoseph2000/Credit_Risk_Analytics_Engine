"""
Home Credit – 01 Preprocessing DAG
==================================
Step 1 of the training pipeline: turns the raw Kaggle CSVs into one cleaned,
feature-engineered parquet per source table.

What it does
------------
Each task runs one notebook from src/Training/01_Preprocess/ with papermill.
The notebooks read the raw CSVs from /data/Input and write their results to
/data/Model-Training/Staging/. A copy of every executed notebook (with its
printed output) is saved to /data/Model-Training/notebook_outputs/ so you can
see what happened in each run.

Task order
----------

    application_train.csv ──► task_application    ──┐
                                                    │
    previous_application.csv ► task_prevloans     ──┼──► task_pos
                                                    ├──► task_creditcard
                                                    └──► task_installments

    bureau.csv               ► task_bureau  ───────────────┐
    bureau_balance.csv   ────┘                             │
                                                           ▼
    (all of the above) ──────────────────────────────► branch_task
                                                           │
                                    trigger_variable_selection param
                                           ┌───────true───────┐false
                                           ▼                   ▼
                              trigger_variable_selection   skip_variable_selection
                              (starts 02_variable_selection)

- task_application, task_prevloans and task_bureau don't depend on anything.
- task_pos, task_creditcard and task_installments need task_prevloans first,
  because they reuse its cleaned previous-loan table.
- Once everything has succeeded, the DAG starts 02_variable_selection, unless
  you set the `trigger_variable_selection` param to false.

The graph allows independent tasks to run side by side, but docker-compose
limits Airflow to one task at a time (AIRFLOW__CORE__PARALLELISM=1), so in
practice they run one after another.
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
# Inside the container these default to the folders docker-compose mounts:
# your project's data/ folder at /data and src/ at /src.
DATA_ROOT   = os.environ.get("HOMECREDIT_DATA_ROOT", "/data")
SRC_DIR     = os.environ.get("HOMECREDIT_SRC_DIR",   "/src")

NB_DIR      = os.path.join(SRC_DIR, "Training", "01_Preprocess")           # notebooks to run

OUTPUT_DIR  = os.path.join(DATA_ROOT, "Model-Training", "Staging")         # processed parquets
NB_OUT_DIR  = os.path.join(DATA_ROOT, "Model-Training", "notebook_outputs")  # executed notebook copies

# Make sure the output folders exist before any task writes to them
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(NB_OUT_DIR, exist_ok=True)
#os.makedirs(NB_DIR, exist_ok=True)


# Which notebook each task runs (task name -> notebook file in NB_DIR)
NOTEBOOKS = {
    "application"   : "Application_data.ipynb",
    "prevloans"     : "Prevloans_data.ipynb",
    "pos"           : "POS_data.ipynb",
    "creditcard"    : "CreditCard_data.ipynb",
    "installments"  : "Payment_installments_data.ipynb",
    "bureau"        : "Bureau_data.ipynb",
}

# ── Settings shared by every task ─────────────────────────────────────────────
# One automatic retry after 5 minutes; no emails.
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
    "trigger_variable_selection": Param(
        default=True,
        type="boolean",
        title="Trigger variable selection?",
        description="If true (default), triggers the 02_variable_selection DAG once "
        "all preprocessing tasks finish. Set to false to stop after preprocessing.",
    )
}

# ── Notebook runner ───────────────────────────────────────────────────────────
def run_notebook(task_id: str, **context) -> None:
    """
    Run one preprocessing notebook with papermill.

    No parameters are passed in: each notebook finds its input and output
    folders itself from the HOMECREDIT_DATA_ROOT environment variable (set
    to /data in the container; on your PC it defaults to the project's
    data/ folder). The executed copy is saved as
    notebook_outputs/<task>__<run date>.ipynb.
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
        parameters={},          # none needed -- the notebook reads its paths from env vars
        progress_bar=False,
        request_save_on_cell_execute=True,  # save after every cell, so a failed run still shows how far it got
    )
    print(f"[{task_id}] ✓ Complete")


# ── DAG definition ────────────────────────────────────────────────────────────
with DAG(
    dag_id="01_preprocessing",
    description="Home Credit feature engineering pipeline",
    default_args=DEFAULT_ARGS,
    start_date=datetime(2024, 1, 1),
    schedule=None,          # no schedule: start it manually from the UI or CLI
    params=UI_PARAMS,
    catchup=False,
    max_active_runs=1,      # never two preprocessing runs at once
    tags=["homecredit", "preprocessing", "feature-engineering"],
) as dag:

    # ── Tables with no dependencies ───────────────────────────────────────────
    # (doc_md is the description shown for each task in the Airflow UI)

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

    # ── Tables that need task_prevloans first ─────────────────────────────────

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

    # ── Decide whether to start variable selection ────────────────────────────
    # Runs only if every table above succeeded (ALL_SUCCESS); returns the id
    # of the task to run next, and Airflow skips the other one.

    @task.branch(trigger_rule=TriggerRule.ALL_SUCCESS)
    def choose_path(**context):
        if context["params"]["trigger_variable_selection"]:
            return "trigger_variable_selection"
        return "skip_variable_selection"

    branch_task = choose_path()

    # Starts the 02_variable_selection DAG and finishes straight away,
    # without waiting for that DAG to complete
    trigger_variable_selection = TriggerDagRunOperator(
        task_id="trigger_variable_selection",
        trigger_dag_id="02_variable_selection",
        wait_for_completion=False,
    )

    # Does nothing -- just marks "stopped after preprocessing" in the graph
    skip_variable_selection = EmptyOperator(task_id="skip_variable_selection")

    # ── Task order ────────────────────────────────────────────────────────────
    #
    #  task_application ──────────────────────────────────► branch_task
    #  task_prevloans   ──► task_pos           ───────────► branch_task
    #                   └──► task_creditcard   ───────────► branch_task
    #                   └──► task_installments ───────────► branch_task
    #  task_bureau ───────────────────────────────────────► branch_task
    #  branch_task ──► trigger_variable_selection | skip_variable_selection

    # Tables with no dependencies go straight to the branch
    [task_application, task_bureau] >> branch_task

    # The three tables that reuse task_prevloans' output run after it
    task_prevloans >> [task_pos, task_creditcard, task_installments] >> branch_task

    # Then either start variable selection or stop
    branch_task >> [trigger_variable_selection, skip_variable_selection]
