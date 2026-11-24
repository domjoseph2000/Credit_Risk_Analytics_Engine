"""
Features from Home Credit credit card statements (credit_card_balance.csv),
one row per applicant.

Mirrors src/Training/01_Preprocess/CreditCard_data.ipynb (Sections 7, 11 and 12).

All three features are built from a monthly table that adds together all of
an applicant's cards for each month, not from individual card rows. The @prod
model (MLflow version 23) uses one of them,
CREDITCARD_UTILIZATION_RATIO_MAX_LAST3M. The other two
(CNT_DRAWINGS_ATM_CURRENT_MEAN_LAST3Y and UTILIZATION_RATIO_LAST_MONTH) are
for the previous @prod (version 11). The notebook's other credit card
features (quarter-on-quarter changes, overdue flags, etc.) aren't used by
either model, so they aren't recreated here.
"""
import pandas as pd
import numpy as np

# The raw credit_card_balance.csv columns build_features() reads (batch scoring loads only these)
REQUIRED_COLUMNS = [
    'SK_ID_CURR',
    'MONTHS_BALANCE',
    'AMT_BALANCE',
    'AMT_CREDIT_LIMIT_ACTUAL',
    'CNT_DRAWINGS_ATM_CURRENT',
]


def build_features(cc_df: pd.DataFrame, valid_ids) -> pd.DataFrame:
    """
    Aggregate each applicant's credit card history into one row.

    valid_ids: the SK_ID_CURR values being scored. The file holds every
    applicant Home Credit has ever seen, so it's cut down to these first.
    """
    CCBal = cc_df[cc_df['SK_ID_CURR'].isin(valid_ids)].copy()
    CCBal = CCBal.astype({'SK_ID_CURR': 'string'})

    # One row per applicant per month, adding up all their cards
    # (CreditCard_data.ipynb Section 7). Blanks count as 0 in the sum, which
    # matches the notebook filling missing ATM drawings with 0.
    # MONTHS_BALANCE is months before the application: -1 = last month.
    sum_cols = ['AMT_BALANCE', 'AMT_CREDIT_LIMIT_ACTUAL', 'CNT_DRAWINGS_ATM_CURRENT']
    CCBal_Mon = (
        CCBal.groupby(['SK_ID_CURR', 'MONTHS_BALANCE'])[sum_cols]
        .sum()
        .reset_index()
    )
    # Utilisation % = balance / credit limit, on the monthly totals. Same
    # formula as the notebook's compute_ratios() (the +1 avoids dividing by zero).
    CCBal_Mon['UTILIZATION_RATIO'] = np.where(
        CCBal_Mon['AMT_CREDIT_LIMIT_ACTUAL'] >= 0,
        CCBal_Mon['AMT_BALANCE'] / (1 + CCBal_Mon['AMT_CREDIT_LIMIT_ACTUAL']) * 100,
        np.nan,
    )

    # CNT_DRAWINGS_ATM_CURRENT_MEAN_LAST3Y: average monthly ATM withdrawals over the last 36 months
    CCBal_3Y_Agg = (
        CCBal_Mon[CCBal_Mon['MONTHS_BALANCE'] >= -36]
        .groupby('SK_ID_CURR')['CNT_DRAWINGS_ATM_CURRENT']
        .mean()
        .reset_index(name='CNT_DRAWINGS_ATM_CURRENT_MEAN_LAST3Y')
    )

    # UTILIZATION_RATIO_MAX_LAST3M: highest monthly utilisation in the last 3 months
    # (the notebook's build_window_agg(CCBal_Mon, -3, '_LAST3M', ...))
    CCBal_3M_Agg = (
        CCBal_Mon[CCBal_Mon['MONTHS_BALANCE'] >= -3]
        .groupby('SK_ID_CURR')['UTILIZATION_RATIO']
        .max()
        .reset_index(name='UTILIZATION_RATIO_MAX_LAST3M')
    )

    # UTILIZATION_RATIO_LAST_MONTH: utilisation in the applicant's most recent month.
    # Uses .nth(-1), not .last(): .last() skips blanks and would return the
    # latest month that HAS a value, not the actual latest month (the bug the
    # notebook's Section 12 fixes).
    CCBal_Last_Snapshot = (
        CCBal_Mon.sort_values(['SK_ID_CURR', 'MONTHS_BALANCE'])
        .groupby('SK_ID_CURR')[['SK_ID_CURR', 'UTILIZATION_RATIO']]
        .nth(-1)
        .reset_index(drop=True)
        .rename(columns={'UTILIZATION_RATIO': 'UTILIZATION_RATIO_LAST_MONTH'})
    )

    final = (
        CCBal[['SK_ID_CURR']].drop_duplicates()
        .merge(CCBal_3Y_Agg, on='SK_ID_CURR', how='left')
        .merge(CCBal_3M_Agg, on='SK_ID_CURR', how='left')
        .merge(CCBal_Last_Snapshot, on='SK_ID_CURR', how='left')
    )
    # Add the CREDITCARD_ prefix used in training
    final.columns = [
        'CREDITCARD_' + col if col != 'SK_ID_CURR' else col
        for col in final.columns
    ]
    return final
