"""
Batch monitoring dashboard: shows each scored batch's numbers (applicant
count, average score and PD, risk band mix) and compares a batch with the
one before it, including a drift check (PSI) on the predicted PDs.

It only reads the files batch scoring already wrote to
DATA_ROOT/Inference/BatchOutput/<date>/. It never loads the model or talks
to MLflow, so it starts instantly and keeps working even when MLflow is down.
It's separate from the real-time API and runs on its own port:

    uvicorn batch_dashboard_api:app --host 0.0.0.0 --port 8001

Then open http://localhost:8001/ui/
"""
import logging
import os

from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from utils import batch_metrics
from utils import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="Home Credit Batch Monitoring Dashboard", version="1.0")

# The dashboard web page (HTML/JS): web/dashboard/index.html
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web", "dashboard")


@app.get("/health")
def health():
    """Simple "is the server up" check."""
    return {"status": "ok"}


@app.get("/api/runs")
def list_runs():
    """
    Summary numbers for every scored batch, oldest first. The page uses this
    for its date picker and the trend lines across all batches.
    """
    dates = batch_metrics.list_batch_runs()
    runs = []
    for date in dates:
        run = batch_metrics.load_run(date)
        metrics = batch_metrics.snapshot_metrics(run["scored_df"])
        runs.append({"date": date, **metrics})
    return {"runs": runs}


@app.get("/api/snapshot/{date}")
def snapshot(date: str):
    """One batch's numbers, plus how they changed since the previous batch (404 if no such batch)."""
    dates = batch_metrics.list_batch_runs()
    if date not in dates:
        raise HTTPException(status_code=404, detail=f"No batch run found for application date {date!r}")

    current = batch_metrics.load_run(date)
    metrics = batch_metrics.snapshot_metrics(current["scored_df"])

    # "Previous" = the batch just before this one by date, not literally the
    # day before, since batches don't necessarily run daily
    idx = dates.index(date)
    previous = batch_metrics.load_run(dates[idx - 1]) if idx > 0 else None
    comparison = batch_metrics.compare_runs(current, previous)

    return {"date": date, "metrics": metrics, "comparison": comparison}


@app.get("/", include_in_schema=False)
def root():
    """Opening the bare address sends you to the dashboard page."""
    return RedirectResponse(url="/ui/")


# Serve the dashboard page at /ui. Added last, and under /ui rather than "/",
# so it can never hide the /api routes above (as in realtime_api.py).
app.mount("/ui", StaticFiles(directory=STATIC_DIR, html=True), name="ui")
