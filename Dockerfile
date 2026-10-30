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
#   AIRFLOW_ADMIN_USER=admin
#   AIRFLOW_ADMIN_PASSWORD=admin
#   AIRFLOW_HOME=/opt/airflow
#   KAGGLE_COMPETITION=home-credit-default-risk
#   KAGGLE_USERNAME / KAGGLE_KEY         Kaggle API credentials (or mount a
#                                        kaggle.json into ~/.kaggle/) -- used
#                                        by entrypoint.sh to download the raw
#                                        input CSVs into DATA_ROOT/Input on
#                                        first run, if they're not there yet
# ─────────────────────────────────────────────────────────────────────────────

FROM python:3.12-slim

# ── System dependencies ───────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        git \
        curl \
        unzip \
    && rm -rf /var/lib/apt/lists/*

# ── Python dependencies ───────────────────────────────────────────────────────
# Pin Airflow to a stable release; extras: celery not needed for standalone.
ENV AIRFLOW_VERSION=3.2.2
ENV PYTHON_VERSION=3.12
ENV CONSTRAINT_URL="https://raw.githubusercontent.com/apache/airflow/constraints-${AIRFLOW_VERSION}/constraints-${PYTHON_VERSION}.txt"

RUN pip install --no-cache-dir --upgrade pip

# Data science stack -- the same requirements.txt used for local runs, so
# models trained here and on a laptop use identical library versions.
# None of these are Airflow dependencies, so the Airflow constraints file
# installed below leaves them untouched.
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt setuptools==78.1.1

# Notebook execution (ipykernel comes from requirements.txt)
RUN pip install --no-cache-dir \
        papermill==2.6.0 \
        jupyter-client==8.6.2

# Kaggle CLI -- entrypoint.sh uses this to fetch the raw competition CSVs
# into HOMECREDIT_DATA_ROOT/Input on first run.
RUN pip install --no-cache-dir kaggle==1.6.17

# Airflow (standalone = scheduler + webserver + triggerer in one process)
RUN pip install --no-cache-dir \
        "apache-airflow==${AIRFLOW_VERSION}" \
        --constraint "${CONSTRAINT_URL}"

# Register the python3 kernel so papermill can find it
RUN python -m ipykernel install --name python3 --display-name "Python 3"

# ── Application files ─────────────────────────────────────────────────────────
WORKDIR /app

COPY src/         ./src/
COPY scripts/     ./scripts/
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
ENV AIRFLOW_ADMIN_USER=admin
ENV AIRFLOW_ADMIN_PASSWORD=admin

# ── Data volume ───────────────────────────────────────────────────────────────
# Mount your project data here:
#   docker run -v /your/local/data:/data ...

VOLUME ["/data"]

# ── Ports ─────────────────────────────────────────────────────────────────────
# Airflow webserver UI
EXPOSE 8080

# ── Entrypoint ────────────────────────────────────────────────────────────────
ENTRYPOINT ["/app/scripts/entrypoint.sh"]
