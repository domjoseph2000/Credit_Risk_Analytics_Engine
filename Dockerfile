# ─────────────────────────────────────────────────────────────────────────────
# Home Credit – Preprocessing Pipeline
# Dockerfile
# ─────────────────────────────────────────────────────────────────────────────
# Build:
#   docker build -t homecredit-pipeline:latest .
#
# Run (standalone Airflow UI + scheduler in one container):
#   docker run -it --rm \
#     -v /path/to/your/data:/data \
#     -p 8080:8080 \
#     homecredit-pipeline:latest
#
# Environment variables (all optional — defaults shown):
#   HOMECREDIT_DATA_ROOT=/data          project root inside the container
#   HOMECREDIT_NB_DIR=/app/notebooks    where patched notebooks are stored
#   AIRFLOW_ADMIN_USER=admin
#   AIRFLOW_ADMIN_PASSWORD=admin
#   AIRFLOW_HOME=/opt/airflow
# ─────────────────────────────────────────────────────────────────────────────

FROM python:3.12-slim

# ── System dependencies ───────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        git \
        curl \
    && rm -rf /var/lib/apt/lists/*

# ── Python dependencies ───────────────────────────────────────────────────────
# Pin Airflow to a stable release; extras: celery not needed for standalone.
ENV AIRFLOW_VERSION=2.9.3
ENV PYTHON_VERSION=3.12
ENV CONSTRAINT_URL="https://raw.githubusercontent.com/apache/airflow/constraints-${AIRFLOW_VERSION}/constraints-${PYTHON_VERSION}.txt"

RUN pip install --no-cache-dir --upgrade pip

# Data science stack
RUN pip install --no-cache-dir \
        pandas==2.2.2 \
        numpy==1.26.4 \
        scipy==1.13.1 \
        scikit-learn==1.5.0 \
        pyarrow==16.1.0 \
        openpyxl==3.1.4

# Notebook execution
RUN pip install --no-cache-dir \
        papermill==2.6.0 \
        ipykernel==6.29.4 \
        jupyter-client==8.6.2

# Airflow (standalone = scheduler + webserver + triggerer in one process)
RUN pip install --no-cache-dir \
        "apache-airflow==${AIRFLOW_VERSION}" \
        --constraint "${CONSTRAINT_URL}"

# Register the python3 kernel so papermill can find it
RUN python -m ipykernel install --name python3 --display-name "Python 3"

# ── Application files ─────────────────────────────────────────────────────────
WORKDIR /app

COPY src/Preprocess/notebooks/   /app/notebooks/
COPY src/Variable-Selection/   /app/variable_selection/
COPY scripts/     /app/scripts/
COPY dags/        /opt/airflow/dags/

RUN chmod +x /app/scripts/entrypoint.sh

# ── Airflow configuration ─────────────────────────────────────────────────────
ENV AIRFLOW_HOME=/opt/airflow
ENV AIRFLOW__CORE__LOAD_EXAMPLES=False
ENV AIRFLOW__CORE__EXECUTOR=SequentialExecutor 
#LocalExecutor
ENV AIRFLOW__DATABASE__SQL_ALCHEMY_CONN=sqlite:////opt/airflow/airflow.db
ENV AIRFLOW__WEBSERVER__EXPOSE_CONFIG=True
ENV AIRFLOW__CORE__DAGS_ARE_PAUSED_AT_CREATION=False

# ── Pipeline environment variables ────────────────────────────────────────────
ENV HOMECREDIT_DATA_ROOT=/data
ENV HOMECREDIT_SRC_DIR=/src
ENV HOMECREDIT_NB_DIR=/src/Preprocess/notebooks
ENV AIRFLOW_ADMIN_USER=admin
ENV AIRFLOW_ADMIN_PASSWORD=admin

# ── Data volume ───────────────────────────────────────────────────────────────
# Mount your project data here:
#   docker run -v /your/local/data:/data ...
# Expected layout under /data:
#   /data/data/Input/application_train.csv
#   /data/data/Input/POS_CASH_balance.csv
#   /data/data/Input/credit_card_balance.csv
#   /data/data/Input/installments_payments.csv
#   /data/data/Input/previous_application.csv
#   /data/data/Input/bureau.csv
#   /data/data/Input/bureau_balance.csv
VOLUME ["/data"]

# ── Ports ─────────────────────────────────────────────────────────────────────
# Airflow webserver UI
EXPOSE 8080

# ── Entrypoint ────────────────────────────────────────────────────────────────
ENTRYPOINT ["/app/scripts/entrypoint.sh"]
