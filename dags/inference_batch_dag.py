"""
Batch inference DAG.

Feature engineering runs as one task per raw table (application, bureau,
previous_application, pos_cash, installments) -- all five in parallel,
since none of them depend on each other -- followed by join_features
(merges the five into one matrix) and score_features (scores it), kept
as separate tasks so a scoring failure can be retried without redoing
the join. Mirrors 01_preprocessing_dag.py's per-table task
pattern rather than score_model_input.py's single sequential script, so
a failure/slowdown in one table is isolated and retried independently
instead of rerunning everything.

Lives in dags/ (not src/Inference/) because that's the only directory
docker-compose live-syncs into the Airflow scheduler's DAG folder
(./dags:/opt/airflow/dags). The actual task logic lives in
src/Inference/batch_feature_tasks.py, which is separately mounted at
/src -- see HOME_CREDIT_INFERENCE_SRC below.
"""
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.models.param import Param
from airflow.operators.python import PythonOperator
from airflow.sensors.python import PythonSensor

# batch_feature_tasks.py lives in src/Inference/, not dags/ -- has to be on
# sys.path before it (or anything inside it) can be imported below. Same
# /src mount batch_feature_tasks.py itself relies on for its own imports.
SRC_ROOT = os.environ.get("HOMECREDIT_SRC_DIR", "/src")
INFERENCE_PKG_SRC = os.environ.get("HOME_CREDIT_INFERENCE_SRC", os.path.join(SRC_ROOT, "Inference"))
if INFERENCE_PKG_SRC not in sys.path:
    sys.path.insert(0, INFERENCE_PKG_SRC)

default_args = {
    "owner": "dom",
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="home_credit_batch_inference",
    default_args=default_args,
    description="Daily batch scoring of new loan applications",
    schedule=None ,#"@daily"
    start_date=datetime(2026, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["home_credit", "inference"],
    params={
        # Exposed on the "Trigger DAG w/ config" screen so a run can score
        # an arbitrary application-data folder instead of always being
        # pinned to this run's scheduled logical date (ds) -- e.g. to
        # rerun a specific past batch, or score a folder that doesn't
        # follow the @daily schedule at all.
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
        # Resolved the same way batch_feature_tasks.py's own _application_date
        # helper does (params override, falling back to the scheduled date) --
        # kept as a self-contained function here, not imported from
        # batch_feature_tasks, so this first task doesn't depend on that
        # module's (heavier) import chain succeeding.
        params = context.get("params") or {}
        application_date = params.get("application_date") or context["ds"]
        # Matches score_model_input.py's app_data_dir: DATA_ROOT/Inference/
        # BatchInput/<date>/application.csv -- only application.csv is
        # date-partitioned, bureau/previous/pos/installments are static.
        path = Path(f"/data/Inference/BatchInput/{application_date}/application.csv")
        return path.exists()

    # PythonSensor instead of FileSensor: FileSensor needs an Airflow
    # Connection (fs_default) to resolve its base path, and this project's
    # entrypoint.sh runs `airflow db migrate` (Airflow 3.x), which -- unlike
    # the old `airflow db init` -- doesn't auto-create that connection, so
    # FileSensor fails with AirflowNotFoundException before ever checking
    # the file. A plain Path.exists() check needs no connection at all.
    wait_for_raw_data = PythonSensor(
        task_id="wait_for_raw_data",
        python_callable=_application_file_exists,
        poke_interval=60,
        timeout=60 * 60,
        mode="reschedule",
    )

    # Importing here (not at module top) keeps a DAG-parse error in
    # batch_feature_tasks.py's own import chain (e.g. a missing dependency)
    # from taking down every other DAG the scheduler parses in this file.
    from batch_feature_tasks import (  # noqa: E402
        build_application_features,
        build_bureau_features,
        build_installments_features,
        build_pos_cash_features,
        build_previous_application_features,
        join_features,
        score_features,
    )

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

    join = PythonOperator(
        task_id="join_features",
        python_callable=join_features,
    )
    score = PythonOperator(
        task_id="score_features",
        python_callable=score_features,
    )

    build_tasks = [build_application, build_bureau, build_previous_application, build_pos_cash, build_installments]
    wait_for_raw_data >> build_tasks >> join >> score
