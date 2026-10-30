#!/usr/bin/env python
"""
Start the MLflow tracking server directly on Windows (no Docker needed), for
running the training notebooks interactively.

Same setup as the docker-compose `mlflow` service: run metadata in
DATA_ROOT/mlflow.db, model files and plots stored under DATA_ROOT/mlruns and
uploaded/downloaded through the server. Notebooks connect to
http://localhost:5000 by default.

Keep this terminal open while you work; stop the server with Ctrl+C. Don't
run it at the same time as the Docker `mlflow` service -- both use port 5000.

Usage (with project_env active):
    python scripts/start_mlflow_server.py
"""
import os
import subprocess
import sys

DATA_ROOT = os.environ.get("HOMECREDIT_DATA_ROOT", r"a:\Dominic\My Projects\Home Credit\data")
DB_PATH = os.path.join(DATA_ROOT, "mlflow.db").replace("\\", "/")
ARTIFACTS_DIR = os.path.join(DATA_ROOT, "mlruns").replace("\\", "/")
PORT = "5000"


def main():
    cmd = [
        sys.executable, "-m", "mlflow", "server",
        "--backend-store-uri", f"sqlite:///{DB_PATH}",
        "--serve-artifacts",
        # A bare "A:/..." path is misread as a URI scheme "a:" on Windows;
        # the file:/// form is required
        "--artifacts-destination", f"file:///{ARTIFACTS_DIR}",
        "--host", "127.0.0.1",
        "--port", PORT,
    ]
    print(f"MLflow server: http://localhost:{PORT}  (Ctrl+C to stop)")
    print(f"  python   : {sys.executable}  (should be your project environment)")
    print(f"  metadata : {DB_PATH}")
    print(f"  files    : {ARTIFACTS_DIR}")
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
