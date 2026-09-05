"""Aggregated features from installments_payments.csv, one row per SK_ID_CURR."""
import pandas as pd
import numpy as np

# Raw installments_payments.csv columns build_features() actually reads.
# (prev_df's requirement is previous_application.REQUIRED_COLUMNS, already
# loaded for previous_application.py -- nothing additional needed there.)
REQUIRED_COLUMNS = [
    'SK_ID_CURR',
    'SK_ID_PREV',
    'DAYS_INSTALMENT',
    'DAYS_ENTRY_PAYMENT',
    'AMT_PAYMENT',
    'AMT_INSTALMENT',
]


def build_features(inst_df: pd.DataFrame, prev_df: pd.DataFrame, valid_ids) -> pd.DataFrame:
    """
    valid_ids: SK_ID_CURR values in the application batch being scored --
    filtering both raw tables here before aggregating avoids burning time
    on the full history of every applicant Home Credit has ever seen.
    """
    Payments = inst_df[inst_df['SK_ID_CURR'].isin(valid_ids)].copy()
    PrevLoans = prev_df[prev_df['SK_ID_CURR'].isin(valid_ids)].copy()
    Payments = Payments.astype({'SK_ID_CURR': 'string', 'SK_ID_PREV': 'string'})
    PrevLoans = PrevLoans.astype({'SK_ID_CURR': 'string', 'SK_ID_PREV': 'string'})

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
    Payments['LATE_PAYMENT_DAYS'] = (
        Payments['DAYS_ENTRY_PAYMENT'] - Payments['DAYS_INSTALMENT']
    )
    Payments['PAYMENT_AMT_TO_INSTALLMENT_RATIO'] = np.where(
        Payments['AMT_INSTALMENT'] > 0,
        Payments['AMT_PAYMENT'] / Payments['AMT_INSTALMENT'],
        np.nan,
    )

    max_cols  = ['LATE_PAYMENT_DAYS', 'PAYMENT_AMT_TO_INSTALLMENT_RATIO']

    Payments_Mon_Max = (
        Payments.groupby(['SK_ID_CURR', 'MONTHS_BALANCE'])[max_cols]
        .max()
        .reset_index()
    )
    Payments_Mon_Max.columns = [
        'PMNT_INSTLMT_' + col if col not in ('SK_ID_CURR', 'MONTHS_BALANCE') else col
        for col in Payments_Mon_Max.columns
    ]
    Payments_Mon = Payments_Mon_Max
    # Join with previous loans to get credit to application ratio and other features
    
    PrevLoans['CREDIT_TO_APPL'] = np.where(
        PrevLoans['AMT_APPLICATION'] > 0,
        PrevLoans['AMT_CREDIT'] / PrevLoans['AMT_APPLICATION'],
        np.nan,
    )
    Payments_PrevLoans = Payments[['SK_ID_CURR', 'SK_ID_PREV']].drop_duplicates().merge(
        PrevLoans, on=['SK_ID_PREV', 'SK_ID_CURR'], how='left'
    )
    Payments_Agg_PrevLoans = (
        Payments_PrevLoans.groupby('SK_ID_CURR')
        .agg(PMNT_INSTLMT_AVG_CREDIT_TO_APPL=('CREDIT_TO_APPL', 'mean'))
        .reset_index() 
    )

        # ── Last 2 Year ───────────────────────────────────────────────────────────
    Payments_Mon_2Y_Agg = (
        Payments_Mon[Payments_Mon['MONTHS_BALANCE'] >= -24]
        .groupby('SK_ID_CURR')['PMNT_INSTLMT_PAYMENT_AMT_TO_INSTALLMENT_RATIO']
        .mean()
        .reset_index(name='PMNT_INSTLMT_PAYMENT_AMT_TO_INSTALLMENT_RATIO_MEAN_2Y')
    )

    # ── Last 1 Year ───────────────────────────────────────────────────────────
    Payments_Mon_1Y_Agg = (
        Payments_Mon[Payments_Mon['MONTHS_BALANCE'] >= -12]
        .groupby('SK_ID_CURR')['PMNT_INSTLMT_LATE_PAYMENT_DAYS']
        .mean()
        .reset_index(name='PMNT_INSTLMT_LATE_PAYMENT_DAYS_MEAN_1Y')
    )

    # Merge all the aggregated features into a single dataframe
    Payments_final = Payments_Agg_PrevLoans.merge(
        Payments_Mon_2Y_Agg, on='SK_ID_CURR', how='left'
    ).merge(
        Payments_Mon_1Y_Agg, on='SK_ID_CURR', how='left'
    )

    return Payments_final
