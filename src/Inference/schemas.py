"""
Request/response schemas for the real-time scoring API.

ApplicationRequest only models the main application table fields, since
that's the minimum needed for a single new applicant. If you want bureau/
previous-loan history factored in at real-time scoring too, extend this
with optional nested tables, or accept SK_ID_CURR and look history up
from a feature store / DB instead of expecting the caller to supply it.
"""
from typing import Optional

from pydantic import BaseModel, Field

class ApplicationRequest(BaseModel):
    SK_ID_CURR: int
    AMT_INCOME_TOTAL: float
    AMT_CREDIT: float
    AMT_ANNUITY: Optional[float] = None
    AMT_GOODS_PRICE: Optional[float] = None
    DAYS_BIRTH: int
    DAYS_EMPLOYED: Optional[int] = None
    DAYS_ID_PUBLISH: Optional[int] = None
    DAYS_LAST_PHONE_CHANGE: Optional[float] = None
    EXT_SOURCE_1: Optional[float] = None
    EXT_SOURCE_2: Optional[float] = None
    EXT_SOURCE_3: Optional[float] = None
    FLOORSMAX_MEDI: Optional[float] = None
    OWN_CAR_AGE: Optional[float] = None
    OCCUPATION_TYPE: Optional[str] = None
    ORGANIZATION_TYPE: Optional[str] = None
    NAME_CONTRACT_TYPE: Optional[str] = None
    CODE_GENDER: Optional[str] = None
    NAME_EDUCATION_TYPE: Optional[str] = None
    NAME_FAMILY_STATUS: Optional[str] = None
    # Every field above except the last four is read directly by
    # features/application.py::build_features (sentinel/XNA cleaning,
    # AGE, GOODSPRICE_TO_AMT_CREDIT, etc.) -- if a field it touches isn't
    # declared here, a request that omits it 500s with a KeyError deep in
    # assemble_features instead of a clean 422. Keep this list in sync with
    # that function's `numeric_cols` / `xna_cols`.

    class Config:
        extra = "allow"  # tolerate extra raw columns; assemble_features will
        # select only what's in the trained feature list and error clearly
        # if something required is truly missing.


class ReasonCode(BaseModel):
    """One attribute's points contribution to CREDIT_SCORE -- see
    explainability.py for how these are derived from SHAP values. Positive
    points push the score up (lower risk); negative points push it down."""
    feature: str
    points: float


class ScoreResponse(BaseModel):
    SK_ID_CURR: int
    PROBA_DEFAULT: float = Field(..., ge=0, le=1)
    CREDIT_SCORE: int
    RISK_BAND: str
    # Top contributors by |points|, not all ~24 features -- see
    # explainability.top_reason_codes(). Doesn't affect CREDIT_SCORE itself,
    # which is computed by pdo_scorecard.py exactly as before.
    REASON_CODES: list[ReasonCode] = Field(default_factory=list)
    # Whether this SK_ID_CURR matched any row in bureau/previous-loan/POS/
    # installments/credit-card history. NOT reliably inferable from
    # REASON_CODES -- SHAP can surface a NaN history feature as a top
    # contributor (the model's learned default-split behavior for missing
    # values), which looks influential without the feature having a real
    # value for this applicant.
    HAS_CREDIT_HISTORY: bool = False
