"""Aggregated features from POS_CASH_balance.csv, one row per SK_ID_CURR."""
import pandas as pd
import numpy as np

# Raw POS_CASH_balance.csv columns build_features() actually reads.
REQUIRED_COLUMNS = ['SK_ID_CURR', 'SK_ID_PREV', 'MONTHS_BALANCE', 'SK_DPD']


def build_features(pos_df: pd.DataFrame, valid_ids) -> pd.DataFrame:
    """
    valid_ids: SK_ID_CURR values in the application batch being scored --
    filtering here before aggregating avoids burning time on the full
    history of every applicant Home Credit has ever seen.
    """
    POSBal = pos_df[pos_df['SK_ID_CURR'].isin(valid_ids)].copy()
    POSBal = POSBal.astype({'SK_ID_PREV': 'string', 'SK_ID_CURR': 'string'})
    POSBal.sort_values(
        by=['SK_ID_CURR', 'SK_ID_PREV', 'MONTHS_BALANCE'],
        inplace=True,
        ignore_index=True,
    )

    POSBal_latest_dpd = (
        POSBal[POSBal['SK_DPD'] > 0]          # only months with a DPD event
        .groupby('SK_ID_CURR')['MONTHS_BALANCE']
        .max()                                  # most recent DPD month (least negative)
        .reset_index()
        .rename(columns={'MONTHS_BALANCE': 'POS_MONTHS_SINCE_LAST_DPD_GT0'})
    )
    POSBal_latest_dpd['POS_MONTHS_SINCE_LAST_DPD_GT0'] = (-POSBal_latest_dpd['POS_MONTHS_SINCE_LAST_DPD_GT0'])
    
    POSBal_final = POSBal[['SK_ID_CURR']].drop_duplicates().merge(
        POSBal_latest_dpd, on='SK_ID_CURR', how='left'
    ) 

    return POSBal_final
