"""
The shape of a request to, and a response from, the real-time /score API.

A request carries only application-form fields. The applicant's credit
history (bureau, previous loans, repayments, cards) isn't sent: the API looks
it up by SK_ID_CURR in the history tables it loaded at startup (see
realtime_api.py). A brand-new SK_ID_CURR just gets blank history features.
"""
from typing import Optional

from pydantic import BaseModel, Field

class ApplicationRequest(BaseModel):
    """
    One applicant's application-form fields, named as in application.csv.

    Fields without "Optional" must be sent. Optional fields may be left out or
    sent as null; they're treated as missing, as blanks were in training.

    features/application.py reads SK_ID_CURR, AMT_CREDIT, AMT_GOODS_PRICE, the
    DAYS_* fields, EXT_SOURCE_1/2/3, FLOORSMAX_MEDI, OWN_CAR_AGE,
    OCCUPATION_TYPE and ORGANIZATION_TYPE. Any field it reads must be declared
    here: a request without one would otherwise fail deep inside the feature
    code with a server error instead of a clear 422 "invalid request".
    AMT_INCOME_TOTAL, AMT_ANNUITY and the last four fields aren't used by the
    current model.
    """
    SK_ID_CURR: int
    AMT_INCOME_TOTAL: float
    AMT_CREDIT: float
    AMT_ANNUITY: Optional[float] = None
    AMT_GOODS_PRICE: Optional[float] = None
    DAYS_BIRTH: int                                 # negative: -12000 = about 33 years before applying
    DAYS_EMPLOYED: Optional[int] = None
    DAYS_ID_PUBLISH: Optional[int] = None
    DAYS_LAST_PHONE_CHANGE: Optional[float] = None
    EXT_SOURCE_1: Optional[float] = None            # external credit scores, 0 to 1
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

    class Config:
        # Accept extra fields without complaint; the feature code only keeps
        # what the model needs and raises a clear error if something it
        # needs is missing.
        extra = "allow"


class ReasonCode(BaseModel):
    """
    How many credit-score points one feature added (positive = lower risk)
    or removed (negative = higher risk). See explainability.py.
    """
    feature: str
    points: float


class ScoreResponse(BaseModel):
    """The /score result for one applicant."""
    SK_ID_CURR: int
    PROBA_DEFAULT: float = Field(..., ge=0, le=1)   # predicted probability of default (PD)
    CREDIT_SCORE: int                               # 300–850, higher = safer
    RISK_BAND: str                                  # "A — Very Low Risk" … "E — Very High Risk"
    # The 5 features that moved the score most, up or down (not all 23).
    # Explanation only: they don't change CREDIT_SCORE.
    REASON_CODES: list[ReasonCode] = Field(default_factory=list)
    # Whether this SK_ID_CURR was found in any history table (bureau,
    # previous loans, POS, repayments, credit cards). Reported separately
    # because REASON_CODES can't tell you: a history feature that's blank
    # can still show up as a top reason, since the model treats "no history"
    # as information in itself.
    HAS_CREDIT_HISTORY: bool = False
