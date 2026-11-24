"""
Features from monthly statements of previous point-of-sale and cash loans at
Home Credit (POS_CASH_balance.csv), one row per applicant.

Mirrors src/Training/01_Preprocess/POS_data.ipynb. Two features are made:
- POS_AVG_CREDIT_TO_APPL: across the applicant's POS/cash loans, the average
  of (amount granted / amount asked for). Used by the @prod model (version 23).
- POS_MONTHS_SINCE_LAST_DPD_GT0: months since a payment was last overdue.
  Only used by the previous @prod (version 11).

POS_AVG_CREDIT_TO_APPL needs previous_application.csv as well (the amounts
asked for and granted are in that file). The notebook takes them from its own
processed previous-loan table; here they're recalculated from the raw file,
the same way features/previous_application.py does.
"""
import pandas as pd
import numpy as np

# The raw POS_CASH_balance.csv columns build_features() reads (batch scoring loads only these)
REQUIRED_COLUMNS = ['SK_ID_CURR', 'SK_ID_PREV', 'MONTHS_BALANCE', 'SK_DPD']


def build_features(pos_df: pd.DataFrame, prev_df: pd.DataFrame, valid_ids) -> pd.DataFrame:
    """
    Aggregate each applicant's POS/cash loan history into one row.

    pos_df: raw POS_CASH_balance rows. prev_df: raw previous_application rows.
    valid_ids: the SK_ID_CURR values being scored; both tables hold every
    applicant Home Credit has ever seen, so they're cut down to these first.
    """
    POSBal = pos_df[pos_df['SK_ID_CURR'].isin(valid_ids)].copy()
    POSBal = POSBal.astype({'SK_ID_PREV': 'string', 'SK_ID_CURR': 'string'})

    # Months since the last overdue payment (SK_DPD = days past due).
    # MONTHS_BALANCE is negative (-1 = last month), so the latest overdue
    # month is the max, and flipping the sign gives "months ago".
    POSBal_latest_dpd = (
        POSBal[POSBal['SK_DPD'] > 0]          # only months with an overdue payment
        .groupby('SK_ID_CURR')['MONTHS_BALANCE']
        .max()                                  # most recent such month
        .reset_index()
        .rename(columns={'MONTHS_BALANCE': 'POS_MONTHS_SINCE_LAST_DPD_GT0'})
    )
    POSBal_latest_dpd['POS_MONTHS_SINCE_LAST_DPD_GT0'] = -POSBal_latest_dpd['POS_MONTHS_SINCE_LAST_DPD_GT0']

    # Amount granted / amount asked for, per previous application
    PrevLoans = prev_df[prev_df['SK_ID_CURR'].isin(valid_ids)].copy()
    PrevLoans = PrevLoans.astype({'SK_ID_PREV': 'string', 'SK_ID_CURR': 'string'})
    PrevLoans['CREDIT_TO_APPL'] = np.where(
        PrevLoans['AMT_APPLICATION'] > 0,
        PrevLoans['AMT_CREDIT'] / PrevLoans['AMT_APPLICATION'],
        np.nan,
    )

    # Average it over just the loans that appear in POS_CASH_balance (i.e.
    # the applicant's POS/cash loans, not every previous application)
    pos_loan_ids = POSBal[['SK_ID_CURR', 'SK_ID_PREV']].drop_duplicates()
    POSBal_PrevLoans = pos_loan_ids.merge(PrevLoans, on=['SK_ID_PREV', 'SK_ID_CURR'], how='left')
    POSBal_Agg_PrevLoans = (
        POSBal_PrevLoans.groupby('SK_ID_CURR')
        .agg(POS_AVG_CREDIT_TO_APPL=('CREDIT_TO_APPL', 'mean'))
        .reset_index()
    )

    POSBal_final = (
        POSBal[['SK_ID_CURR']].drop_duplicates()
        .merge(POSBal_latest_dpd, on='SK_ID_CURR', how='left')
        .merge(POSBal_Agg_PrevLoans, on='SK_ID_CURR', how='left')
    )
    return POSBal_final
