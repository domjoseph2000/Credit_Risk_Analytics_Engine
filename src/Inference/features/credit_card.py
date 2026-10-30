"""
Aggregated features from credit_card_balance.csv, one row per SK_ID_CURR.

Rebuilt for the LightGBM_v4 (@prod) feature set -- see
data/Model-Training/ModelOutput/LightGBM_v4/lgbm_feature_importance.xlsx for
the ground truth feature list, and src/Preprocess/CreditCard_data.ipynb for
the logic this mirrors. Previously a TODO stub -- v3 didn't use any credit
card features, v4 does, so this table is now wired into the pipeline (see
features/pipeline.py) for the first time.

Both of v4's credit-card features are derived from a customer-month table
(all of a customer's cards summed together per MONTHS_BALANCE), not the raw
per-card rows -- CreditCard_data.ipynb's Sections 7 & 11/12. None of the
other feature families in that notebook (QoQ, DPD flags, previous-loan
rollup, instalment rate) feed either of these two columns, so they aren't
reproduced here.
"""
import pandas as pd
import numpy as np

# Raw credit_card_balance.csv columns build_features() actually reads.
REQUIRED_COLUMNS = [
    'SK_ID_CURR',
    'MONTHS_BALANCE',
    'AMT_BALANCE',
    'AMT_CREDIT_LIMIT_ACTUAL',
    'CNT_DRAWINGS_ATM_CURRENT',
]


def build_features(cc_df: pd.DataFrame, valid_ids) -> pd.DataFrame:
    """
    valid_ids: SK_ID_CURR values in the application batch being scored --
    filtering here before aggregating avoids burning time on the full
    history of every applicant Home Credit has ever seen.
    """
    CCBal = cc_df[cc_df['SK_ID_CURR'].isin(valid_ids)].copy()
    CCBal = CCBal.astype({'SK_ID_CURR': 'string'})

    # Customer-month roll-up: sum across all of a customer's cards for each
    # MONTHS_BALANCE (CreditCard_data.ipynb Section 7). NaN sub-columns
    # contribute 0 to pandas' sum(), matching the notebook's structural-zero
    # imputation for CNT_DRAWINGS_ATM_CURRENT without needing that step here.
    sum_cols = ['AMT_BALANCE', 'AMT_CREDIT_LIMIT_ACTUAL', 'CNT_DRAWINGS_ATM_CURRENT']
    CCBal_Mon = (
        CCBal.groupby(['SK_ID_CURR', 'MONTHS_BALANCE'])[sum_cols]
        .sum()
        .reset_index()
    )
    # Recompute on the summed customer-month totals, same formula as
    # CreditCard_data.ipynb's compute_ratios().
    CCBal_Mon['UTILIZATION_RATIO'] = np.where(
        CCBal_Mon['AMT_CREDIT_LIMIT_ACTUAL'] >= 0,
        CCBal_Mon['AMT_BALANCE'] / (1 + CCBal_Mon['AMT_CREDIT_LIMIT_ACTUAL']) * 100,
        np.nan,
    )

    # CNT_DRAWINGS_ATM_CURRENT_MEAN_LAST3Y: mean over the last 36 months.
    CCBal_3Y_Agg = (
        CCBal_Mon[CCBal_Mon['MONTHS_BALANCE'] >= -36]
        .groupby('SK_ID_CURR')['CNT_DRAWINGS_ATM_CURRENT']
        .mean()
        .reset_index(name='CNT_DRAWINGS_ATM_CURRENT_MEAN_LAST3Y')
    )

    # UTILIZATION_RATIO_LAST_MONTH: most recent month observed per customer.
    # .nth(-1) (not .last()) -- .last() silently skips NaN and picks the
    # last *non-null* value instead of the actual most recent month, which
    # is exactly the bug CreditCard_data.ipynb's Section 12 fix avoids.
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
        .merge(CCBal_Last_Snapshot, on='SK_ID_CURR', how='left')
    )
    final.columns = [
        'CREDITCARD_' + col if col != 'SK_ID_CURR' else col
        for col in final.columns
    ]
    return final
