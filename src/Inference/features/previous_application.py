"""
Aggregated features from previous_application.csv, one row per SK_ID_CURR.

Rebuilt for the LightGBM_v4 (@prod) feature set -- see
data/Model-Training/ModelOutput/LightGBM_v4/lgbm_feature_importance.xlsx for
the ground truth feature list, and src/Preprocess/Prevloans_data.ipynb for
the logic this mirrors.
"""
import pandas as pd
import numpy as np

# Raw previous_application.csv columns build_features() actually reads.
# pos_cash.py::build_features() also reads this same raw table (for
# CREDIT_TO_APPL) and needs no columns beyond this set.
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
    Aggregate *agg_cols* (mean) over the most recent *days_back* days.
    Mirrors Prevloans_data.ipynb's build_window_agg exactly.
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
    valid_ids: SK_ID_CURR values in the application batch being scored --
    filtering here before aggregating avoids burning time on the full
    history of every applicant Home Credit has ever seen.
    """
    PrevLoans = prev_df[prev_df['SK_ID_CURR'].isin(valid_ids)].copy()
    PrevLoans = PrevLoans.astype({'SK_ID_CURR': 'string', 'SK_ID_PREV': 'string'})

    PrevLoans['CREDIT_TO_APPL'] = np.where(
        PrevLoans['AMT_APPLICATION'] > 0,
        PrevLoans['AMT_CREDIT'] / PrevLoans['AMT_APPLICATION'],
        np.nan,
    )

    # v4 needs the LAST1Y window only (PREVIOUS_TOTAL_LOAN_CREDIT_TO_APPL_MEAN_LAST1Y).
    PrevLoans_1Y_Agg = build_window_agg(
        PrevLoans, ['CREDIT_TO_APPL'], -365, '_LAST1Y', 'PREVIOUS_TOTAL_LOAN_'
    )

    RefusedLoans = PrevLoans[PrevLoans['NAME_CONTRACT_STATUS'] == 'Refused']
    RefusedLoans_Aggregate = (
        RefusedLoans.groupby('SK_ID_CURR')
        .agg(PREVIOUS_REFUSED_LOAN_LAST_REFUSED_DAY=('DAYS_DECISION', 'max'))
        .reset_index()
    )

    PreviousLoans_final = (
        PrevLoans[['SK_ID_CURR']].drop_duplicates()
        .merge(PrevLoans_1Y_Agg, on='SK_ID_CURR', how='left')
        .merge(RefusedLoans_Aggregate, on='SK_ID_CURR', how='left')
    )
    return PreviousLoans_final
