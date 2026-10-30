"""Aggregated features from credit_card_balance.csv, one row per SK_ID_CURR."""
import pandas as pd


def build_features(cc_df: pd.DataFrame) -> pd.DataFrame:
    df = cc_df.copy()

    agg = df.groupby("SK_ID_CURR").agg(
        CC_MONTHS_BALANCE_COUNT=("MONTHS_BALANCE", "count"),
        CC_AMT_BALANCE_MEAN=("AMT_BALANCE", "mean"),
        CC_AMT_BALANCE_MAX=("AMT_BALANCE", "max"),
        CC_AMT_CREDIT_LIMIT_MEAN=("AMT_CREDIT_LIMIT_ACTUAL", "mean"),
        CC_SK_DPD_MAX=("SK_DPD", "max"),
    ).reset_index()

    if {"CC_AMT_BALANCE_MEAN", "CC_AMT_CREDIT_LIMIT_MEAN"}.issubset(agg.columns):
        agg["CC_UTILIZATION_MEAN"] = (
            agg["CC_AMT_BALANCE_MEAN"] / agg["CC_AMT_CREDIT_LIMIT_MEAN"].replace(0, pd.NA)
        )

    # TODO: paste your notebook's exact credit card aggregation logic here.

    return agg
