"""
Aggregated features from POS_CASH_balance.csv, one row per SK_ID_CURR.

Rebuilt for the LightGBM_v4 (@prod) feature set -- see
data/Model-Training/ModelOutput/LightGBM_v4/lgbm_feature_importance.xlsx for
the ground truth feature list, and src/Preprocess/POS_data.ipynb for the
logic this mirrors.

POS_AVG_CREDIT_TO_APPL (Section 10, "Previous-Loan Enrichment" in the
notebook) needs previous_application.csv too -- the notebook reads it from
its own already-enriched prev_loan_data_processed.parquet, but since
inference works from raw CSVs, build_features() takes the raw
previous_application table and recomputes CREDIT_TO_APPL the same way
previous_application.py does.
"""
import pandas as pd
import numpy as np

# Raw POS_CASH_balance.csv columns build_features() actually reads.
REQUIRED_COLUMNS = ['SK_ID_CURR', 'SK_ID_PREV', 'MONTHS_BALANCE', 'SK_DPD']


def build_features(pos_df: pd.DataFrame, prev_df: pd.DataFrame, valid_ids) -> pd.DataFrame:
    """
    valid_ids: SK_ID_CURR values in the application batch being scored --
    filtering both raw tables here before aggregating avoids burning time
    on the full history of every applicant Home Credit has ever seen.
    """
    POSBal = pos_df[pos_df['SK_ID_CURR'].isin(valid_ids)].copy()
    POSBal = POSBal.astype({'SK_ID_PREV': 'string', 'SK_ID_CURR': 'string'})

    POSBal_latest_dpd = (
        POSBal[POSBal['SK_DPD'] > 0]          # only months with a DPD event
        .groupby('SK_ID_CURR')['MONTHS_BALANCE']
        .max()                                  # most recent DPD month (least negative)
        .reset_index()
        .rename(columns={'MONTHS_BALANCE': 'POS_MONTHS_SINCE_LAST_DPD_GT0'})
    )
    POSBal_latest_dpd['POS_MONTHS_SINCE_LAST_DPD_GT0'] = -POSBal_latest_dpd['POS_MONTHS_SINCE_LAST_DPD_GT0']

    # Previous-loan enrichment: average CREDIT_TO_APPL across this
    # customer's loans that actually appear in POS_CASH_balance (i.e.
    # POS/cash-type loans only, not every previous application).
    PrevLoans = prev_df[prev_df['SK_ID_CURR'].isin(valid_ids)].copy()
    PrevLoans = PrevLoans.astype({'SK_ID_PREV': 'string', 'SK_ID_CURR': 'string'})
    PrevLoans['CREDIT_TO_APPL'] = np.where(
        PrevLoans['AMT_APPLICATION'] > 0,
        PrevLoans['AMT_CREDIT'] / PrevLoans['AMT_APPLICATION'],
        np.nan,
    )

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
