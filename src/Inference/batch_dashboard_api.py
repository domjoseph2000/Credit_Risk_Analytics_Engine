"""
Read-only batch monitoring dashboard: current-snapshot metrics for the most
recent scored batch, compared against the previous run (t vs t-1).

Deliberately separate from realtime_api.py -- this only reads files
score_model_input.py already wrote under DATA_ROOT/Inference/BatchOutput/,
so it has no model/SHAP/MLflow dependency and starts instantly, unlike the
real-time API (which fails to start at all if the @prod model can't be
loaded -- see README "Known gaps"). Run alongside it on a different port:

    uvicorn batch_dashboard_api:app --host 0.0.0.0 --port 8001

Then open http://localhost:8001/ui/
"""
import logging
import os

from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

import batch_metrics
import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="Home Credit Batch Monitoring Dashboard", version="1.0")

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static_batch")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/api/runs")
def list_runs():
    """Every available run, oldest first -- for the date picker and the
    volume/mean-score trend sparkline across more than just t vs t-1."""
    dates = batch_metrics.list_batch_runs()
    runs = []
    for date in dates:
        run = batch_metrics.load_run(date)
        metrics = batch_metrics.snapshot_metrics(run["scored_df"])
        runs.append({"date": date, **metrics})
    return {"runs": runs}


@app.get("/api/snapshot/{date}")
def snapshot(date: str):
    dates = batch_metrics.list_batch_runs()
    if date not in dates:
        raise HTTPException(status_code=404, detail=f"No batch run found for application date {date!r}")

    current = batch_metrics.load_run(date)
    metrics = batch_metrics.snapshot_metrics(current["scored_df"])

    # t-1 = the run immediately before this one in the sorted list of
    # available dates, not literally "yesterday" -- batches don't
    # necessarily run daily, so this is "the previous completed run."
    idx = dates.index(date)
    previous = batch_metrics.load_run(dates[idx - 1]) if idx > 0 else None
    comparison = batch_metrics.compare_runs(current, previous)

    return {"date": date, "metrics": metrics, "comparison": comparison}


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/ui/")


# Mounted last, under /ui rather than "/", so it can never shadow the API
# routes above -- same pattern as realtime_api.py's /ui.
app.mount("/ui", StaticFiles(directory=STATIC_DIR, html=True), name="ui")
