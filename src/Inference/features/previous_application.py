"""
Features from the applicant's previous applications to Home Credit
(previous_application.csv), one row per applicant.

Mirrors src/Training/01_Preprocess/Prevloans_data.ipynb. The @prod model
(MLflow version 23) uses two of the features made here:
- PREVIOUS_TOTAL_LOAN_CREDIT_TO_APPL_MEAN_LAST1Y: over applications in the
  last year, the average of (amount granted / amount asked for)
- PREVIOUS_REFUSED_LOAN_LAST3YEAR_COUNT: refused applications in the last 3 years
PREVIOUS_REFUSED_LOAN_LAST_REFUSED_DAY is only for the previous @prod (version 11).
"""
import pandas as pd
import numpy as np

# The raw previous_application.csv columns build_features() reads (batch
# scoring loads only these). pos_cash.py reads this same table and needs
# nothing beyond these columns.
REQUIRED_COLUMNS = [
    'SK_ID_CURR',
    'SK_ID_PREV',
    'DAYS_DECISION',
    'AMT_APPLICATION',
    'AMT_CREDIT',
    'NAME_CONTRACT_STATUS',
]


def build_window_agg(df, agg_cols, days_back, suffix, prefix):
    """
    Average of `agg_cols` per applicant over decisions in the last `days_back`
    days (a negative number, e.g. -365 for one year). The new column is named
    <prefix><column>_MEAN<suffix>, all upper case. Same as the notebook's
    build_window_agg.
    """
    window = df[df['DAYS_DECISION'] >= days_back]
    agg = (
        window.groupby('SK_ID_CURR')[agg_cols]
        .agg(['mean'])
        .reset_index()
    )
    agg.columns = [
        f"{prefix}{col[0]}_{col[1]}{suffix}".upper() if col[1] else col[0]
        for col in agg.columns
    ]
    return agg


def build_features(prev_df: pd.DataFrame, valid_ids) -> pd.DataFrame:
    """
    Aggregate each applicant's previous applications into one row.

    valid_ids: the SK_ID_CURR values being scored. The file holds every
    applicant Home Credit has ever seen, so it's cut down to these first.
    """
    PrevLoans = prev_df[prev_df['SK_ID_CURR'].isin(valid_ids)].copy()
    PrevLoans = PrevLoans.astype({'SK_ID_CURR': 'string', 'SK_ID_PREV': 'string'})

    # Amount granted / amount asked for (below 1 = given less than requested)
    PrevLoans['CREDIT_TO_APPL'] = np.where(
        PrevLoans['AMT_APPLICATION'] > 0,
        PrevLoans['AMT_CREDIT'] / PrevLoans['AMT_APPLICATION'],
        np.nan,
    )

    # Its average over the last year -> PREVIOUS_TOTAL_LOAN_CREDIT_TO_APPL_MEAN_LAST1Y
    PrevLoans_1Y_Agg = build_window_agg(
        PrevLoans, ['CREDIT_TO_APPL'], -365, '_LAST1Y', 'PREVIOUS_TOTAL_LOAN_'
    )

    # Refused applications: how many in the last 3 years, and how recent the latest was.
    # DAYS_DECISION counts days before this application (negative), so
    # ">= -365*3" means "within 3 years" and the max is the latest.
    RefusedLoans = PrevLoans[PrevLoans['NAME_CONTRACT_STATUS'] == 'Refused']
    RefusedLoans_Aggregate = (
        RefusedLoans.groupby('SK_ID_CURR')
        .agg(
            PREVIOUS_REFUSED_LOAN_LAST3YEAR_COUNT=('DAYS_DECISION', lambda x: (x >= -365 * 3).sum()),
            PREVIOUS_REFUSED_LOAN_LAST_REFUSED_DAY=('DAYS_DECISION', 'max'),
        )
        .reset_index()
    )
    # Make sure the count is a decimal number even when no one in the batch
    # was ever refused (e.g. one real-time applicant): pandas then returns it
    # as type 'object', which the model rejects. Training also stored it as a
    # decimal, with blank meaning "never refused".
    RefusedLoans_Aggregate['PREVIOUS_REFUSED_LOAN_LAST3YEAR_COUNT'] = (
        RefusedLoans_Aggregate['PREVIOUS_REFUSED_LOAN_LAST3YEAR_COUNT'].astype('float64')
    )

    PreviousLoans_final = (
        PrevLoans[['SK_ID_CURR']].drop_duplicates()
        .merge(PrevLoans_1Y_Agg, on='SK_ID_CURR', how='left')
        .merge(RefusedLoans_Aggregate, on='SK_ID_CURR', how='left')
    )
    return PreviousLoans_final
