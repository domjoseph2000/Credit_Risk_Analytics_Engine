"""
Real-time single-applicant scoring API.

Run locally:
    uvicorn realtime_api:app --host 0.0.0.0 --port 8000

The model is loaded once at startup (via model.load_model's lru_cache)
and reused across requests -- no MLflow round trip per call. The shared
history tables (bureau/previous_application/pos_cash/installments/
credit_card) are loaded once the same way -- see warm_shared_history()
below for why a single /score request needs them at all.
"""
import functools
import logging
import os
import sys

SRC_ROOT = os.environ.get("HOMECREDIT_SRC_DIR", r"a:\Dominic\My Projects\Home Credit\src")
INFERENCE_PKG_SRC = os.environ.get("HOME_CREDIT_INFERENCE_SRC", os.path.join(SRC_ROOT, "Inference"))
sys.path.insert(0, INFERENCE_PKG_SRC)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.responses import RedirectResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402

import config  # noqa: E402
import model as model_module  # noqa: E402
from explainability import explain_contributions, load_explainer, top_reason_codes  # noqa: E402
from score_model_input import load_shared_history_tables, score_applicants  # noqa: E402
from schemas import ApplicationRequest, ScoreResponse  # noqa: E402
from features.pipeline import assemble_features  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="Home Credit Default Scoring API", version="1.0")

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")


@functools.lru_cache(maxsize=1)
def load_shared_history() -> dict[str, pd.DataFrame]:
    """
    Of the model's 24 features, only 11 come from the application form
    itself -- the other 13 (BUREAUCREDIT_*, PMNT_INSTLMT_*, POS_*,
    PREVIOUS_*, CREDITCARD_*) are aggregates over that applicant's credit
    history at Home Credit, which a single /score request has no way to
    submit itself. This loads the same shared history tables batch scoring
    reads from DATA_ROOT/Input -- Home Credit's full history for every
    applicant it's ever seen -- ONCE at startup, cached in memory (these
    are multi-million-row CSVs; re-reading them per request would make
    every call take seconds). Each request below filters this down to one
    SK_ID_CURR, the same way batch's per-table build_*_features functions
    filter to a whole batch's worth of IDs.

    This only works for an SK_ID_CURR that already appears in these tables
    (an existing applicant with prior history) -- a genuinely new
    SK_ID_CURR simply gets NaN for all 13 history-derived features, which
    LightGBM handles natively (no imputation needed), rather than a 422.
    """
    tables = load_shared_history_tables(config.RAW_DATA_DIR)
    logger.info("Loaded shared history tables: %s", {k: v.shape for k, v in tables.items()})
    return tables


@app.on_event("startup")
def warm_model():
    # Force the model (and its SHAP explainer, and the shared history
    # tables) to load at startup rather than on first request, so the
    # first real caller doesn't eat that latency.
    model_module.load_model()
    load_explainer()
    load_shared_history()


@app.get("/health")
def health():
    return {"status": "ok"}


def _sanitize_optional_fields(data: dict) -> dict:
    """
    None -> np.nan. A single-row DataFrame built straight from a dict gives
    any column whose only value is Python None dtype "object" (pandas has
    nothing else in the column to infer a numeric dtype from), which
    LightGBM's predict() rejects outright ("pandas dtypes must be int,
    float or bool") -- unlike batch scoring, where NaN comes from
    pandas.read_csv and is already a proper float64 NaN. np.nan is a numpy
    float, so a single-row column of just np.nan infers float64, same as
    batch's NaNs. An applicant leaving any optional numeric field blank
    (e.g. no car, so OWN_CAR_AGE is unset) would otherwise 422 on this alone.
    """
    return {k: (np.nan if v is None else v) for k, v in data.items()}


@app.post("/score", response_model=ScoreResponse)
def score(request: ApplicationRequest):
    logger.info("Scoring request for SK_ID_CURR=%s", request.SK_ID_CURR)
    app_df = pd.DataFrame([_sanitize_optional_fields(request.dict())])

    # Filter the cached shared-history tables down to this one applicant --
    # same tables, same column requirements, same build_features() calls
    # batch scoring uses; assemble_features can't tell the difference
    # between "one row from a real-time request" and "one row of a batch".
    raw_tables = {"application": app_df}
    for key, table in load_shared_history().items():
        raw_tables[key] = table[table["SK_ID_CURR"] == request.SK_ID_CURR]

    # Whether this SK_ID_CURR actually matched any row in the shared history
    # tables -- computed here rather than inferred from REASON_CODES, since
    # SHAP can legitimately surface a NaN history feature as a top
    # contributor (it reflects the model's learned default-split behavior
    # for missing values, not the feature's real value) -- see
    # explainability.py's module docstring.
    has_credit_history = any(
        len(table) > 0 for key, table in raw_tables.items() if key != "application"
    )

    try:
        feature_df = assemble_features(raw_tables)
        result = score_applicants(feature_df)
    except ValueError as e:
        # Missing required features -> bad request, not a server error.
        logger.warning("Rejecting SK_ID_CURR=%s: %s", request.SK_ID_CURR, e)
        raise HTTPException(status_code=422, detail=str(e))
    except Exception:
        logger.exception("Scoring failed for SK_ID_CURR=%s", request.SK_ID_CURR)
        raise

    row = result.iloc[0]
    # Explanation is computed after scoring and never feeds back into it --
    # if this step fails for some reason, that's a bug in the explanation
    # path, not in the score itself.
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
    return RedirectResponse(url="/ui/")


# A small static form that POSTs to /score on this same origin and renders
# the response -- see static/index.html. Mounted last, and under /ui rather
# than "/", so it can never shadow the API routes above.
app.mount("/ui", StaticFiles(directory=STATIC_DIR, html=True), name="ui")
