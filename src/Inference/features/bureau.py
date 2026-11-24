"""
Features from credit bureau records (bureau.csv): the applicant's loans at
other lenders. One row per applicant.

Mirrors src/Training/01_Preprocess/Bureau_data.ipynb. The @prod model (MLflow
version 23) uses all 6 features made here, each prefixed BUREAUCREDIT_:
AMT_CREDIT_SUM_DEBT_MEAN, DAYS_CREDIT_UPDATE_MEAN, TOTAL_ACTIVE_CREDIT_COUNT,
DAYS_TO_LAST_CREDIT, AVG_DAYS_BETWEEN_CREDITS and
ACTIVE_ACCOUNT_UTILIZATION_RATIO_MAX. None of them need bureau_balance.csv,
so that file isn't read.
"""
import pandas as pd
import numpy as np

# The raw bureau.csv columns build_features() reads (batch scoring loads only these)
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
    Aggregate each applicant's bureau records into one row.

    valid_ids: the SK_ID_CURR values being scored. bureau.csv holds every
    applicant Home Credit has ever seen, so it's cut down to these first.
    Applicants with no bureau records simply don't appear in the result; they
    get blanks after the join in features/pipeline.py.
    """
    df = bureau_df.loc[bureau_df['SK_ID_CURR'].isin(valid_ids), REQUIRED_COLUMNS].copy()
    df = df.astype({'SK_ID_CURR': 'string'})

    # Fill in missing outstanding debt, as Bureau_data.ipynb Section 3 does:
    #   closed credit                 -> 0 owed
    #   active credit, not a card     -> assume the full credit amount is owed
    # (The notebook's third rule, for active credit cards, only fills the
    # credit limit, which no feature here uses.)
    closed = df['CREDIT_ACTIVE'] == 'Closed'
    df.loc[closed & df['AMT_CREDIT_SUM_DEBT'].isna(), 'AMT_CREDIT_SUM_DEBT'] = 0

    active_noncard = (df['CREDIT_ACTIVE'] == 'Active') & (df['CREDIT_TYPE'] != 'Credit card')
    df.loc[active_noncard & df['AMT_CREDIT_SUM_DEBT'].isna(), 'AMT_CREDIT_SUM_DEBT'] = (
        df.loc[active_noncard & df['AMT_CREDIT_SUM_DEBT'].isna(), 'AMT_CREDIT_SUM']
    )

    # Share of each credit line in use: amount owed / credit amount
    df['UTILIZATION_RATIO'] = np.where(
        df['AMT_CREDIT_SUM'] > 0,
        df['AMT_CREDIT_SUM_DEBT'] / df['AMT_CREDIT_SUM'],
        np.nan,
    )

    # Summaries over all of an applicant's bureau credits
    df_agg_all = (
        df.groupby('SK_ID_CURR').agg(
            AMT_CREDIT_SUM_DEBT_MEAN=('AMT_CREDIT_SUM_DEBT', 'mean'),
            DAYS_CREDIT_UPDATE_MEAN=('DAYS_CREDIT_UPDATE', 'mean'),
            # Number of credits still open (Bureau_data.ipynb Section 8)
            TOTAL_ACTIVE_CREDIT_COUNT=('CREDIT_ACTIVE', lambda x: (x == 'Active').sum()),
            # Days since the most recent credit was opened (DAYS_CREDIT is negative, so max = latest)
            DAYS_TO_LAST_CREDIT=('DAYS_CREDIT', 'max'),
            # Average gap in days between one credit being opened and the next
            AVG_DAYS_BETWEEN_CREDITS=('DAYS_CREDIT', lambda x: x.sort_values(ascending=True).diff().mean()),
        )
        .reset_index()
    )
    # Make sure the count is a decimal number even when there are no rows at
    # all (e.g. one real-time applicant with no bureau history): pandas then
    # returns it as type 'object', which the model rejects. Training also
    # stored it as a decimal, with blank meaning "no bureau records".
    df_agg_all['TOTAL_ACTIVE_CREDIT_COUNT'] = df_agg_all['TOTAL_ACTIVE_CREDIT_COUNT'].astype('float64')

    # Highest utilisation across the applicant's still-open credits (Bureau_data.ipynb Section 8c)
    df_agg_active = (
        df[df['CREDIT_ACTIVE'] == 'Active']
        .groupby('SK_ID_CURR')
        .agg(ACTIVE_ACCOUNT_UTILIZATION_RATIO_MAX=('UTILIZATION_RATIO', 'max'))
        .reset_index()
    )

    df_agg_final = df_agg_all.merge(df_agg_active, on='SK_ID_CURR', how='left')
    # Add the BUREAUCREDIT_ prefix used in training
    df_agg_final.columns = [
        col if col == 'SK_ID_CURR' else f'BUREAUCREDIT_{col}'
        for col in df_agg_final.columns
    ]
    return df_agg_final
