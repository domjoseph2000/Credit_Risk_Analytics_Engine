"""
Real-time scoring API: send one applicant, get back their PD, credit score,
risk band and top reason codes.

Run it from src/Inference with the project environment active:

    uvicorn realtime_api:app --host 0.0.0.0 --port 8000

Then open http://localhost:8000/ui/ for a simple web form, or POST JSON to
/score (see schemas.ApplicationRequest for the fields).

At startup (about 35 seconds) it loads the @prod model from MLflow and the
five credit-history tables from DATA_ROOT/Input into memory. Requests then
need no MLflow call or disk read. Restart it after promoting a new model.
"""
import functools
import logging
import os
import sys

# Make src/Inference importable when uvicorn is started from another folder
SRC_ROOT = os.environ.get("HOMECREDIT_SRC_DIR", r"a:\Dominic\My Projects\Home Credit\src")
INFERENCE_PKG_SRC = os.environ.get("HOME_CREDIT_INFERENCE_SRC", os.path.join(SRC_ROOT, "Inference"))
sys.path.insert(0, INFERENCE_PKG_SRC)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.responses import RedirectResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402

from utils import config  # noqa: E402
from utils import model as model_module  # noqa: E402
from utils.explainability import explain_contributions, load_explainer, top_reason_codes  # noqa: E402
from score_model_input import load_shared_history_tables, score_applicants  # noqa: E402
from utils.schemas import ApplicationRequest, ScoreResponse  # noqa: E402
from features.pipeline import assemble_features  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="Home Credit Default Scoring API", version="1.0")

# The web form (web/realtime/index.html)
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web", "realtime")


@functools.lru_cache(maxsize=1)
def load_shared_history() -> dict[str, pd.DataFrame]:
    """
    Load the credit-history tables once and keep them in memory.

    Of the @prod model's 23 features, only 10 come from the application form.
    The other 13 (BUREAUCREDIT_*, PMNT_INSTLMT_*, POS_*, PREVIOUS_*,
    CREDITCARD_*) summarise the applicant's credit history, which a request
    doesn't include. So the API keeps the same history tables batch scoring
    uses (bureau, previous applications, POS, repayments, credit cards --
    about 30 million rows) in memory, and each request picks out its
    applicant's rows. Reading them per request would take seconds.

    An SK_ID_CURR with no history simply gets blank history features, which
    the model handles (as it did in training).
    """
    tables = load_shared_history_tables(config.RAW_DATA_DIR)
    logger.info("Loaded shared history tables: %s", {k: v.shape for k, v in tables.items()})
    return tables


@app.on_event("startup")
def warm_model():
    """Load the model, the reason-code explainer and the history tables at startup, so the first request isn't slow."""
    model_module.load_model()
    load_explainer()
    load_shared_history()


@app.get("/health")
def health():
    """Simple "is the server up" check."""
    return {"status": "ok"}


def _sanitize_optional_fields(data: dict) -> dict:
    """
    Turn None (a field left empty) into NaN (pandas' "missing").

    A one-row table built from a value of None gets the type 'object', which
    the model rejects; NaN gives a proper number column with a blank in it,
    the same as a blank cell in a batch CSV. Without this, leaving any
    optional field empty (e.g. no car, so no OWN_CAR_AGE) would fail.
    """
    return {k: (np.nan if v is None else v) for k, v in data.items()}


@app.post("/score", response_model=ScoreResponse)
def score(request: ApplicationRequest):
    """
    Score one applicant.

    Returns PD, credit score, risk band, the top 5 reason codes and whether
    any credit history was found. 422 if a feature the model needs couldn't
    be built from the request.
    """
    logger.info("Scoring request for SK_ID_CURR=%s", request.SK_ID_CURR)
    app_df = pd.DataFrame([_sanitize_optional_fields(request.dict())])

    # This applicant's rows from each history table. From here on the
    # feature code can't tell one real-time applicant from a one-row batch.
    raw_tables = {"application": app_df}
    for key, table in load_shared_history().items():
        raw_tables[key] = table[table["SK_ID_CURR"] == request.SK_ID_CURR]

    # Did this applicant have any history rows at all? Checked directly here,
    # because the reason codes can't tell you (see schemas.ScoreResponse).
    has_credit_history = any(
        len(table) > 0 for key, table in raw_tables.items() if key != "application"
    )

    try:
        feature_df = assemble_features(raw_tables)
        result = score_applicants(feature_df)
    except ValueError as e:
        # A required feature couldn't be built -> reject the request (422), not a server error
        logger.warning("Rejecting SK_ID_CURR=%s: %s", request.SK_ID_CURR, e)
        raise HTTPException(status_code=422, detail=str(e))
    except Exception:
        logger.exception("Scoring failed for SK_ID_CURR=%s", request.SK_ID_CURR)
        raise

    row = result.iloc[0]
    # Reason codes are worked out after scoring and never change the score
    reason_codes_row = explain_contributions(feature_df).iloc[0]
    reasons = top_reason_codes(reason_codes_row, n=5)

    return ScoreResponse(
        SK_ID_CURR=int(row["SK_ID_CURR"]),
        PROBA_DEFAULT=float(row["PROBA_DEFAULT"]),
        CREDIT_SCORE=int(row["score"]),
        RISK_BAND=str(row["Risk_Band"]),
        REASON_CODES=reasons,
        HAS_CREDIT_HISTORY=has_credit_history,
    )


@app.get("/", include_in_schema=False)
def root():
    """Opening the bare address sends you to the web form."""
    return RedirectResponse(url="/ui/")


# Serve the web form (web/realtime/index.html) at /ui; it sends its requests to
# /score on this same server. Added last, and under /ui rather than "/", so
# it can never hide the API routes above.
app.mount("/ui", StaticFiles(directory=STATIC_DIR, html=True), name="ui")
