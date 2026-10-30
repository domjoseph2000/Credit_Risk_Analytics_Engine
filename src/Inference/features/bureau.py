"""
Aggregated features from bureau.csv, joined on SK_ID_CURR.

Rebuilt for the LightGBM_v4 (@prod) feature set -- see
data/Model-Training/ModelOutput/LightGBM_v4/lgbm_feature_importance.xlsx for
the ground truth feature list, and src/Preprocess/Bureau_data.ipynb for the
logic this mirrors. None of v4's 5 bureau features depend on
bureau_balance.csv, so that table isn't read here at all.
"""
import pandas as pd
import numpy as np

# Raw bureau.csv columns build_features() actually reads.
REQUIRED_COLUMNS = [
    'SK_ID_CURR',
    'CREDIT_TYPE',
    'CREDIT_ACTIVE',
    'AMT_CREDIT_SUM_DEBT',
    'AMT_CREDIT_SUM',
    'DAYS_CREDIT_UPDATE',
    'DAYS_CREDIT',
]


def build_features(bureau_df: pd.DataFrame, valid_ids) -> pd.DataFrame:
    """
    valid_ids: SK_ID_CURR values in the application batch being scored --
    bureau.csv covers every applicant Home Credit has ever seen, so
    filtering to just this batch before aggregating avoids burning time on
    IDs that get dropped by the join in pipeline.py anyway.
    """
    df = bureau_df.loc[bureau_df['SK_ID_CURR'].isin(valid_ids), REQUIRED_COLUMNS].copy()
    df = df.astype({'SK_ID_CURR': 'string'})

    # Impute AMT_CREDIT_SUM_DEBT nulls (Bureau_data.ipynb Section 3). Only
    # the two branches that touch AMT_CREDIT_SUM_DEBT are needed here --
    # the active-credit-card branch only ever imputes AMT_CREDIT_SUM_LIMIT,
    # which none of v4's features read.
    closed = df['CREDIT_ACTIVE'] == 'Closed'
    df.loc[closed & df['AMT_CREDIT_SUM_DEBT'].isna(), 'AMT_CREDIT_SUM_DEBT'] = 0

    active_noncard = (df['CREDIT_ACTIVE'] == 'Active') & (df['CREDIT_TYPE'] != 'Credit card')
    df.loc[active_noncard & df['AMT_CREDIT_SUM_DEBT'].isna(), 'AMT_CREDIT_SUM_DEBT'] = (
        df.loc[active_noncard & df['AMT_CREDIT_SUM_DEBT'].isna(), 'AMT_CREDIT_SUM']
    )

    df['UTILIZATION_RATIO'] = np.where(
        df['AMT_CREDIT_SUM'] > 0,
        df['AMT_CREDIT_SUM_DEBT'] / df['AMT_CREDIT_SUM'],
        np.nan,
    )

    # All-credit aggregates
    df_agg_all = (
        df.groupby('SK_ID_CURR').agg(
            AMT_CREDIT_SUM_DEBT_MEAN=('AMT_CREDIT_SUM_DEBT', 'mean'),
            DAYS_CREDIT_UPDATE_MEAN=('DAYS_CREDIT_UPDATE', 'mean'),
            DAYS_TO_LAST_CREDIT=('DAYS_CREDIT', 'max'),
            AVG_DAYS_BETWEEN_CREDITS=('DAYS_CREDIT', lambda x: x.sort_values(ascending=True).diff().mean()),
        )
        .reset_index()
    )

    # Active-credit-only aggregate (Bureau_data.ipynb Section 8c)
    df_agg_active = (
        df[df['CREDIT_ACTIVE'] == 'Active']
        .groupby('SK_ID_CURR')
        .agg(ACTIVE_ACCOUNT_UTILIZATION_RATIO_MAX=('UTILIZATION_RATIO', 'max'))
        .reset_index()
    )

    df_agg_final = df_agg_all.merge(df_agg_active, on='SK_ID_CURR', how='left')
    df_agg_final.columns = [
        col if col == 'SK_ID_CURR' else f'BUREAUCREDIT_{col}'
        for col in df_agg_final.columns
    ]
    return df_agg_final
