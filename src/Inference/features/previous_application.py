"""Aggregated features from previous_application.csv, one row per SK_ID_CURR."""
import pandas as pd
import numpy as np

# Raw previous_application.csv columns build_features() actually reads.
# installments.py::build_features() also joins against this same raw table
# (for CREDIT_TO_APPL) and needs no columns beyond this set.
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
    Aggregate *agg_cols* over the most recent *days_back* days.

    Parameters
    ----------
    df        : DataFrame with DAYS_DECISION and SK_ID_CURR columns
    agg_cols  : list of columns to aggregate
    days_back : int (negative), e.g. -365*3 for last 3 years
    suffix    : str appended after col name, e.g. '_LAST3Y'
    prefix    : str prepended to col name, e.g. 'PREVIOUS_TOTAL_LOAN_'
    """
    window = df[df['DAYS_DECISION'] >= days_back]
    agg = (
        window.groupby('SK_ID_CURR')[agg_cols]
        .agg(['mean'])
        .reset_index()
    )
    # Flatten MultiIndex; SK_ID_CURR has empty second level after reset_index
    agg.columns = [
        f"{prefix}{col[0]}_{col[1]}{suffix}".upper() if col[1] else col[0]
        for col in agg.columns
    ]
    return agg

def build_features(prev_df: pd.DataFrame, valid_ids) -> pd.DataFrame:
    """
    valid_ids: SK_ID_CURR values in the application batch being scored --
    filtering here before aggregating avoids burning time on the full
    history of every applicant Home Credit has ever seen.
    """
    PrevLoans = prev_df[prev_df['SK_ID_CURR'].isin(valid_ids)].copy()
    PrevLoans = PrevLoans.astype({'SK_ID_CURR': 'string', 'SK_ID_PREV': 'string'})
    PrevLoans.sort_values(
        by=['SK_ID_CURR', 'DAYS_DECISION'],
        ascending=[True, False],
        inplace=True,
        ignore_index=True,
    )

    SENTINEL = 365243
    numeric_cols = PrevLoans.select_dtypes(include='number').columns
    sentinel_cols = [col for col in numeric_cols if (PrevLoans[col] == SENTINEL).any()]
    print(f"Columns containing sentinel {SENTINEL}: {sentinel_cols}")

    for col in sentinel_cols:
        PrevLoans.loc[PrevLoans[col] == SENTINEL, col] = np.nan

    print("Sentinel values replaced with NaN.")

    object_cols = PrevLoans.select_dtypes(include='object').columns
    xna_cols = [col for col in object_cols if (PrevLoans[col] == 'XNA').any()]
    print(f"Columns containing 'XNA': {xna_cols}")

    for col in xna_cols:
        PrevLoans.loc[PrevLoans[col] == 'XNA', col] = np.nan

    print("XNA values replaced with NaN.")

    PrevLoans['CREDIT_TO_APPL'] = np.where(
        PrevLoans['AMT_APPLICATION'] > 0,
        PrevLoans['AMT_CREDIT'] / PrevLoans['AMT_APPLICATION'],
        np.nan,
    )

    agg_cols = ['CREDIT_TO_APPL']
    PrevLoans_3Y_Agg = build_window_agg(PrevLoans, agg_cols, -365*3, '_LAST3Y', 'PREVIOUS_TOTAL_LOAN_')
    
    RefusedLoans  = PrevLoans[PrevLoans['NAME_CONTRACT_STATUS'] == 'Refused'].copy()
    RefusedLoans_Aggregate_CurrentID = (
        RefusedLoans.groupby('SK_ID_CURR').agg(
            PREVIOUS_REFUSED_LOAN_COUNT=('NAME_CONTRACT_STATUS', 'count'),
            PREVIOUS_REFUSED_LOAN_LAST_REFUSED_DAY=('DAYS_DECISION', 'max')
        )
        .reset_index()  # FIX: SK_ID_CURR was left in index without this
    )

    PrevLoans_final = PrevLoans['SK_ID_CURR'].drop_duplicates().to_frame()
    PreviousLoans_final = ( 
        PrevLoans_final.merge(
            PrevLoans_3Y_Agg, on='SK_ID_CURR', how='left') .merge(
                RefusedLoans_Aggregate_CurrentID, on='SK_ID_CURR', how='left'
        )
        .reset_index()  
    )

    return PreviousLoans_final
