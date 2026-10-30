"""
Aggregated features from installments_payments.csv, one row per SK_ID_CURR.

Rebuilt for the LightGBM_v4 (@prod) feature set -- see
data/Model-Training/ModelOutput/LightGBM_v4/lgbm_feature_importance.xlsx for
the ground truth feature list, and src/Preprocess/Payment_installments_data.ipynb
for the logic this mirrors.
"""
import pandas as pd
import numpy as np

# Raw installments_payments.csv columns build_features() actually reads.
REQUIRED_COLUMNS = [
    'SK_ID_CURR',
    'SK_ID_PREV',
    'DAYS_INSTALMENT',
    'DAYS_ENTRY_PAYMENT',
    'AMT_PAYMENT',
    'AMT_INSTALMENT',
]


def build_features(inst_df: pd.DataFrame, valid_ids) -> pd.DataFrame:
    """
    valid_ids: SK_ID_CURR values in the application batch being scored --
    filtering here before aggregating avoids burning time on the full
    history of every applicant Home Credit has ever seen.
    """
    Payments = inst_df[inst_df['SK_ID_CURR'].isin(valid_ids)].copy()
    Payments = Payments.astype({'SK_ID_CURR': 'string', 'SK_ID_PREV': 'string'})

    Payments['MONTHS_BALANCE'] = np.floor(Payments['DAYS_INSTALMENT'] / 30).astype(int)
    Payments.sort_values(
        by=['SK_ID_CURR', 'SK_ID_PREV', 'MONTHS_BALANCE', 'DAYS_ENTRY_PAYMENT'],
        inplace=True,
        ignore_index=True,
    )

    for col in ['DAYS_ENTRY_PAYMENT', 'AMT_PAYMENT']:
        Payments[col] = (
            Payments.groupby('SK_ID_PREV')[col]
            .transform(lambda x: x.ffill(limit=6).bfill(limit=6))
        )

    Payments['LATE_PAYMENT_DAYS'] = Payments['DAYS_ENTRY_PAYMENT'] - Payments['DAYS_INSTALMENT']
    Payments['PAYMENT_AMT_TO_INSTALLMENT_RATIO'] = np.where(
        Payments['AMT_INSTALMENT'] > 0,
        Payments['AMT_PAYMENT'] / Payments['AMT_INSTALMENT'],
        np.nan,
    )

    max_cols = ['LATE_PAYMENT_DAYS', 'PAYMENT_AMT_TO_INSTALLMENT_RATIO']
    Payments_Mon = (
        Payments.groupby(['SK_ID_CURR', 'MONTHS_BALANCE'])[max_cols]
        .max()
        .reset_index()
    )
    Payments_Mon.columns = [
        'PMNT_INSTLMT_' + col if col not in ('SK_ID_CURR', 'MONTHS_BALANCE') else col
        for col in Payments_Mon.columns
    ]

    # v4 needs LATE_PAYMENT_DAYS over the LAST1Y window and
    # PAYMENT_AMT_TO_INSTALLMENT_RATIO over the LAST2Y window.
    Payments_Mon_1Y_Agg = (
        Payments_Mon[Payments_Mon['MONTHS_BALANCE'] >= -12]
        .groupby('SK_ID_CURR')['PMNT_INSTLMT_LATE_PAYMENT_DAYS']
        .mean()
        .reset_index(name='PMNT_INSTLMT_LATE_PAYMENT_DAYS_MEAN_LAST1Y')
    )
    Payments_Mon_2Y_Agg = (
        Payments_Mon[Payments_Mon['MONTHS_BALANCE'] >= -24]
        .groupby('SK_ID_CURR')['PMNT_INSTLMT_PAYMENT_AMT_TO_INSTALLMENT_RATIO']
        .mean()
        .reset_index(name='PMNT_INSTLMT_PAYMENT_AMT_TO_INSTALLMENT_RATIO_MEAN_LAST2Y')
    )

    Payments_final = (
        Payments[['SK_ID_CURR']].drop_duplicates()
        .merge(Payments_Mon_1Y_Agg, on='SK_ID_CURR', how='left')
        .merge(Payments_Mon_2Y_Agg, on='SK_ID_CURR', how='left')
    )
    return Payments_final
