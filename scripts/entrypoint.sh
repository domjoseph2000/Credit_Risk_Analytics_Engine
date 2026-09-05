#!/usr/bin/env bash
# entrypoint.sh
# Patches notebooks then delegates to the Airflow entrypoint.

set -euo pipefail

echo "=== Home Credit Pipeline — Container Start ==="
echo "DATA_ROOT : ${HOMECREDIT_DATA_ROOT:-/data}"
echo "SRC_DIR   : ${HOMECREDIT_SRC_DIR:-/src}"

# 1. Create Data/Output directories
mkdir -p "${HOMECREDIT_DATA_ROOT:-/data}/Input"
mkdir -p "${HOMECREDIT_DATA_ROOT:-/data}/Model-Training/Staging"
mkdir -p "${HOMECREDIT_DATA_ROOT:-/data}/Model-Training/ModelDataset"
mkdir -p "${HOMECREDIT_DATA_ROOT:-/data}/Model-Training/HyperParameters"
mkdir -p "${HOMECREDIT_DATA_ROOT:-/data}/Model-Training/ModelOutput"
mkdir -p "${HOMECREDIT_DATA_ROOT:-/data}/Model-Training/ScoreOutput"
mkdir -p "${HOMECREDIT_DATA_ROOT:-/data}/Model-Training/Variable-Selection"
mkdir -p "${HOMECREDIT_DATA_ROOT:-/data}/Model-Training/notebook_outputs"
mkdir -p "${HOMECREDIT_DATA_ROOT:-/data}/mlruns"
mkdir -p "${HOMECREDIT_DATA_ROOT:-/data}/Inference"


# 2. Create Source/Logic directories
mkdir -p "${HOMECREDIT_SRC_DIR:-/src}/Preprocess"
mkdir -p "${HOMECREDIT_SRC_DIR:-/src}/Variable-Selection"
mkdir -p "${HOMECREDIT_SRC_DIR:-/src}/MDS-Creation"
mkdir -p "${HOMECREDIT_SRC_DIR:-/src}/Hyperparameter-Tuning"
mkdir -p "${HOMECREDIT_SRC_DIR:-/src}/Model-Training"
mkdir -p "${HOMECREDIT_SRC_DIR:-/src}/Scoring"
mkdir -p "${HOMECREDIT_SRC_DIR:-/src}/Inference"

# 3. Initialise Airflow DB on first run
echo ""
echo "--- Initialising Airflow DB ---"
airflow db migrate

# 4. Create default admin user if not already present
airflow users create \
    --username "${AIRFLOW_ADMIN_USER:-admin}" \
    --password "${AIRFLOW_ADMIN_PASSWORD:-admin}" \
    --firstname "Jane" \
    --lastname  "Doe" \
    --role      "Admin" \
    --email     "admin@homecredit.local" \
    2>/dev/null || true

echo ""
echo "--- Cleaning up stale PID files ---"
rm -f /opt/airflow/airflow-webserver.pid /opt/airflow/airflow-scheduler.pidx    

echo ""
echo "--- Starting Airflow standalone ---"
exec airflow standalone
