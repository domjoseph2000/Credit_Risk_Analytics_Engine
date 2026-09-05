"""
Real-time single-applicant scoring API.

Run locally:
    uvicorn realtime_api:app --host 0.0.0.0 --port 8000

The model is loaded once at startup (via model.load_model's lru_cache)
and reused across requests -- no MLflow round trip per call.
"""
import logging
import os
import sys

SRC_ROOT = os.environ.get("HOMECREDIT_SRC_DIR", r"a:\Dominic\My Projects\Home Credit\src")
INFERENCE_PKG_SRC = os.environ.get("HOME_CREDIT_INFERENCE_SRC", os.path.join(SRC_ROOT, "Inference"))
sys.path.insert(0, INFERENCE_PKG_SRC)

import pandas as pd  # noqa: E402
from fastapi import FastAPI, HTTPException  # noqa: E402

import model as model_module  # noqa: E402
from score_model_input import score_applicants  # noqa: E402
from schemas import ApplicationRequest, ScoreResponse  # noqa: E402
from features.pipeline import assemble_features  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="Home Credit Default Scoring API", version="1.0")


@app.on_event("startup")
def warm_model():
    # Force the model to load at startup rather than on first request,
    # so the first real caller doesn't eat the MLflow load latency.
    model_module.load_model()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/score", response_model=ScoreResponse)
def score(request: ApplicationRequest):
    logger.info("Scoring request for SK_ID_CURR=%s", request.SK_ID_CURR)
    app_df = pd.DataFrame([request.dict()])
    try:
        feature_df = assemble_features({"application": app_df})
        result = score_applicants(feature_df)
    except ValueError as e:
        # Missing required features -> bad request, not a server error.
        logger.warning("Rejecting SK_ID_CURR=%s: %s", request.SK_ID_CURR, e)
        raise HTTPException(status_code=422, detail=str(e))
    except Exception:
        logger.exception("Scoring failed for SK_ID_CURR=%s", request.SK_ID_CURR)
        raise

    row = result.iloc[0]
    return ScoreResponse(
        SK_ID_CURR=int(row["SK_ID_CURR"]),
        PROBA_DEFAULT=float(row["PROBA_DEFAULT"]),
        CREDIT_SCORE=int(row["score"]),
        RISK_BAND=str(row["Risk_Band"]),
    )
