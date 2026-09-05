"""
Aggregated features from bureau.csv (+ bureau_balance.csv), joined on
SK_ID_CURR. Replace aggregation choices with your actual notebook logic;
the shape (one row per SK_ID_CURR, prefixed columns) is what matters for
compatibility with the rest of the pipeline.
"""
import pandas as pd
import numpy as np

# Raw bureau.csv columns build_features() actually reads -- see
# features/pipeline.py::RAW_COLUMN_REQUIREMENTS for how batch loaders use
# this to avoid pulling in all 17 raw bureau columns for a 1.7M-row table.
REQUIRED_COLUMNS = [
    'SK_ID_CURR',
    'SK_ID_BUREAU',
    'CREDIT_TYPE',
    'CREDIT_ACTIVE',
    'AMT_CREDIT_SUM_LIMIT',
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
    df = df.astype({'SK_ID_CURR': 'string', 'SK_ID_BUREAU': 'string'})
    df.sort_values(by=['SK_ID_CURR', 'DAYS_CREDIT'], ascending=[True, False], inplace=True, ignore_index=True)

    # 1. Impute missing values in AMT_CREDIT_SUM_LIMIT and AMT_CREDIT_SUM_DEBT based on CREDIT_ACTIVE and CREDIT_TYPE

    # Closed credits: limit and debt are 0
    closed = df['CREDIT_ACTIVE'] == 'Closed'
    df.loc[closed & df['AMT_CREDIT_SUM_LIMIT'].isna(), 'AMT_CREDIT_SUM_LIMIT'] = 0
    df.loc[closed & df['AMT_CREDIT_SUM_DEBT'].isna(),  'AMT_CREDIT_SUM_DEBT']  = 0

    # Active non-card: limit = 0, debt = full credit sum
    active_noncredit = (df['CREDIT_ACTIVE'] == 'Active') & (df['CREDIT_TYPE'] != 'Credit card')
    df.loc[active_noncredit & df['AMT_CREDIT_SUM_LIMIT'].isna(), 'AMT_CREDIT_SUM_LIMIT'] = 0
    df.loc[active_noncredit & df['AMT_CREDIT_SUM_DEBT'].isna(),  'AMT_CREDIT_SUM_DEBT']  = (
        df.loc[active_noncredit & df['AMT_CREDIT_SUM_DEBT'].isna(), 'AMT_CREDIT_SUM']
    )

    # Active credit cards: limit = max(credit_sum - debt, 0)
    active_card = (df['CREDIT_ACTIVE'] == 'Active') & (df['CREDIT_TYPE'] == 'Credit card')
    card_limit_null = active_card & df['AMT_CREDIT_SUM_LIMIT'].isna()
    df.loc[card_limit_null, 'AMT_CREDIT_SUM_LIMIT'] = np.where(
        df.loc[card_limit_null, 'AMT_CREDIT_SUM'] > df.loc[card_limit_null, 'AMT_CREDIT_SUM_DEBT'],
        df.loc[card_limit_null, 'AMT_CREDIT_SUM'] - df.loc[card_limit_null, 'AMT_CREDIT_SUM_DEBT'],
        0,
    )

    print("Credit-sum imputation complete.")
    print("Remaining nulls in imputed columns:")
    print(df[['AMT_CREDIT_SUM_LIMIT', 'AMT_CREDIT_SUM_DEBT']].isna().sum().to_string())

    # 2. Create derived features; BUREAUCREDIT_AMT_CREDIT_SUM_DEBT_MEAN, BUREAUCREDIT_UTILIZATION_RATIO_MEAN, BUREAUCREDIT_DAYS_CREDIT_UPDATE_MEAN

    df['UTILIZATION_RATIO'] = np.where(
        df['AMT_CREDIT_SUM'] > 0,
        df['AMT_CREDIT_SUM_DEBT'] / df['AMT_CREDIT_SUM'],
        np.nan,
    )
    print("Derived UTILIZATION_RATIO.")
    print("Aggregating features by SK_ID_CURR...")

    features_to_aggregate = [
        'AMT_CREDIT_SUM_DEBT',
        'UTILIZATION_RATIO',
        'DAYS_CREDIT_UPDATE'
    ]
    df_agg_numeric = (
        df.groupby('SK_ID_CURR')[features_to_aggregate]
        .agg(['mean'])
        .reset_index()
    )

    columns = [ col[0] if not col[1]  else '_'.join(col).upper()
        for col in df_agg_numeric.columns
    ]
    df_agg_numeric.columns = columns

    print("Aggregating count-based features...")
    # 3. Create count-based features; TOTAL_ACTIVE_CREDIT_COUNT, DAYS_TO_LAST_CREDIT, AVG_DAYS_BETWEEN_CREDITS
    df_agg_counts = (
        df.groupby('SK_ID_CURR').agg(
            TOTAL_ACTIVE_CREDIT_COUNT=('CREDIT_ACTIVE', lambda x: (x == 'Active').sum()),
            DAYS_TO_LAST_CREDIT=('DAYS_CREDIT', 'max'),
            AVG_DAYS_BETWEEN_CREDITS=('DAYS_CREDIT', lambda x: x.sort_values(ascending=True).diff().mean())
        )
        .reset_index()  
    )

    # 4. Merge numeric and count-based features
    df_agg_final = df_agg_numeric.merge(df_agg_counts, on='SK_ID_CURR', how='left')
    df_agg_final.columns = [
        col if col == 'SK_ID_CURR' else f'BUREAUCREDIT_{col}'
        for col in df_agg_final.columns
    ]

    print("Final aggregated features:")
    print(df_agg_final.shape[0], "applicants with aggregated bureau features.")

    return df_agg_final
