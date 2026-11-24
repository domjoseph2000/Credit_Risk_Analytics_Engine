"""
Home Credit – Batch Inference DAG
=================================
Scores a batch of new loan applications with the production (@prod) model.

    wait_for_raw_data → 6 × build_*_features → join_features → score_features

Input and output
----------------
- In:  data/Inference/BatchInput/<date>/application.csv (the new applicants),
       plus the shared credit-history tables in data/Input.
- Out: data/Inference/BatchOutput/<date>/
       scored_applicants.parquet, band_summary.csv, reason_codes.parquet.

Tasks
-----
1. wait_for_raw_data -- waits (up to an hour) for that date's application.csv.
2. One feature task per source table: application, bureau,
   previous_application, pos_cash, installments, credit_card. They don't
   depend on each other. (docker-compose limits Airflow to one task at a
   time, so they run one after another.)
3. join_features  -- combines the six into one feature table.
4. score_features -- predicts PD, converts it to a credit score and risk
   band, and calculates per-feature reason codes.

Each task saves its result to data/Inference/BatchFeatures/<date>/, so if
one fails you can rerun just that task. The same steps can also be run
without Airflow, in one go, with src/Inference/score_model_input.py.

How to run it
-------------
No schedule: trigger it with the `application_date` param (YYYY-MM-DD), e.g.

    docker compose exec pipeline airflow dags trigger home_credit_batch_inference \
        --conf '{"application_date": "2026-07-31"}'

Why this file is in dags/
-------------------------
docker-compose only syncs the dags/ folder into Airflow's DAG folder. The
task code itself lives in src/Inference/utils/batch_feature_tasks.py (mounted at
/src), which this file imports.
"""
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.models.param import Param
from airflow.operators.python import PythonOperator
from airflow.sensors.python import PythonSensor

# Make src/Inference importable, so batch_feature_tasks (and the modules it
# imports in turn) can be found below
SRC_ROOT = os.environ.get("HOMECREDIT_SRC_DIR", "/src")
INFERENCE_PKG_SRC = os.environ.get("HOME_CREDIT_INFERENCE_SRC", os.path.join(SRC_ROOT, "Inference"))
if INFERENCE_PKG_SRC not in sys.path:
    sys.path.insert(0, INFERENCE_PKG_SRC)

# Settings shared by every task: two retries, 5 minutes apart
default_args = {
    "owner": "dom",
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="home_credit_batch_inference",
    default_args=default_args,
    description="Daily batch scoring of new loan applications",
    schedule=None ,#"@daily"   <- swap in "@daily" to run it every day automatically
    start_date=datetime(2026, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["home_credit", "credit risk engine", "inference"],
    params={
        # Shown on the "Trigger DAG" screen. Lets a run score any dated
        # input folder (e.g. to rerun a past batch). Left blank, the run's
        # own date is used -- which is what a daily schedule would want.
        "application_date": Param(
            default="",
            type="string",
            title="Application date",
            description="Date (YYYY-MM-DD) of the batch to score -- selects "
            "DATA_ROOT/Inference/BatchInput/<date>/application.csv as input. "
            "Leave blank to use this run's scheduled date.",
        ),
    },
) as dag:

    def _application_file_exists(**context) -> bool:
        """True once this batch's application.csv is in place."""
        # Same date rule as batch_feature_tasks._application_date: the
        # `application_date` param if given, otherwise the run's date (ds).
        # Written out here rather than imported, so this first task doesn't
        # depend on batch_feature_tasks (and pandas etc.) importing cleanly.
        params = context.get("params") or {}
        application_date = params.get("application_date") or context["ds"]
        # Only application.csv is dated; the history tables in data/Input are
        # shared by every batch, so they aren't checked here.
        path = Path(f"/data/Inference/BatchInput/{application_date}/application.csv")
        return path.exists()

    # Calls _application_file_exists every 60 seconds until it returns True,
    # for up to an hour; after that the task fails. mode="reschedule" frees
    # the worker slot between checks, so waiting doesn't block other tasks.
    #
    # (A PythonSensor rather than Airflow's FileSensor: FileSensor needs an
    # "fs_default" connection, which this Airflow 3 setup doesn't create, so
    # it would fail before ever checking the file.)
    wait_for_raw_data = PythonSensor(
        task_id="wait_for_raw_data",
        python_callable=_application_file_exists,
        poke_interval=60,
        timeout=60 * 60,
        mode="reschedule",
    )

    # Imported here rather than at the top of the file: if this import ever
    # fails (e.g. a missing package), only this DAG breaks, not the others
    # Airflow loads.
    from utils.batch_feature_tasks import (  # noqa: E402
        build_application_features,
        build_bureau_features,
        build_credit_card_features,
        build_installments_features,
        build_pos_cash_features,
        build_previous_application_features,
        join_features,
        score_features,
    )

    # One feature task per source table (code in src/Inference/utils/batch_feature_tasks.py)
    build_application = PythonOperator(
        task_id="build_application_features",
        python_callable=build_application_features,
    )
    build_bureau = PythonOperator(
        task_id="build_bureau_features",
        python_callable=build_bureau_features,
    )
    build_previous_application = PythonOperator(
        task_id="build_previous_application_features",
        python_callable=build_previous_application_features,
    )
    build_pos_cash = PythonOperator(
        task_id="build_pos_cash_features",
        python_callable=build_pos_cash_features,
    )
    build_installments = PythonOperator(
        task_id="build_installments_features",
        python_callable=build_installments_features,
    )
    build_credit_card = PythonOperator(
        task_id="build_credit_card_features",
        python_callable=build_credit_card_features,
    )

    # Combine the six feature files, then score. Kept as two tasks so a
    # scoring failure (e.g. MLflow down) can be retried without redoing the join.
    join = PythonOperator(
        task_id="join_features",
        python_callable=join_features,
    )
    score = PythonOperator(
        task_id="score_features",
        python_callable=score_features,
    )

    # Task order
    build_tasks = [
        build_application,
        build_bureau,
        build_previous_application,
        build_pos_cash,
        build_installments,
        build_credit_card,
    ]
    wait_for_raw_data >> build_tasks >> join >> score
