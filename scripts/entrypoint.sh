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

# 1b. Download raw Kaggle input files if Input/ is empty
INPUT_DIR="${HOMECREDIT_DATA_ROOT:-/data}/Input"
KAGGLE_COMPETITION="${KAGGLE_COMPETITION:-home-credit-default-risk}"

echo ""
echo "--- Checking Kaggle input files (${KAGGLE_COMPETITION}) ---"
if [ -n "$(ls -A "${INPUT_DIR}"/*.csv 2>/dev/null)" ]; then
    echo "CSVs already present in ${INPUT_DIR}, skipping download."
elif [ -f "${KAGGLE_CONFIG_DIR:-$HOME/.kaggle}/kaggle.json" ] || { [ -n "${KAGGLE_USERNAME:-}" ] && [ -n "${KAGGLE_KEY:-}" ]; }; then
    echo "Downloading ${KAGGLE_COMPETITION} into ${INPUT_DIR} ..."
    kaggle competitions download -c "${KAGGLE_COMPETITION}" -p "${INPUT_DIR}"
    unzip -o "${INPUT_DIR}/${KAGGLE_COMPETITION}.zip" -d "${INPUT_DIR}"
    rm -f "${INPUT_DIR}/${KAGGLE_COMPETITION}.zip"
    echo "Kaggle input files downloaded to ${INPUT_DIR}."
else
    echo "WARNING: no Kaggle credentials found (mount kaggle.json into"
    echo "  ${KAGGLE_CONFIG_DIR:-$HOME/.kaggle}/kaggle.json, or set KAGGLE_USERNAME + KAGGLE_KEY)."
    echo "  Skipping download -- place the competition CSVs in ${INPUT_DIR} manually."
fi

# 2. Create Source/Logic directories
mkdir -p "${HOMECREDIT_SRC_DIR:-/src}/Training/01_Preprocess"
mkdir -p "${HOMECREDIT_SRC_DIR:-/src}/Training/02_Variable-Selection"
mkdir -p "${HOMECREDIT_SRC_DIR:-/src}/Training/03_MDS-Creation"
mkdir -p "${HOMECREDIT_SRC_DIR:-/src}/Training/04_Model-Development"
mkdir -p "${HOMECREDIT_SRC_DIR:-/src}/Training/05_Segmentation-Analysis"
mkdir -p "${HOMECREDIT_SRC_DIR:-/src}/Training/06_Scoring"
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
