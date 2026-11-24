"""
Features from repayment history on previous Home Credit loans
(installments_payments.csv), one row per applicant.

Mirrors src/Training/01_Preprocess/Payment_installments_data.ipynb. The @prod
model (MLflow version 23) uses three of the features made here:
- PMNT_INSTLMT_LATE_PAYMENT_DAYS_MAX_LAST1Y   worst lateness in the last year
- PMNT_INSTLMT_LATE_PAYMENT_DAYS_MEAN_LAST6M  average lateness in the last 6 months
- PMNT_INSTLMT_PAYMENT_AMT_TO_INSTALLMENT_RATIO_MEAN_LAST2Y  share of each
  instalment actually paid, averaged over 2 years
PMNT_INSTLMT_LATE_PAYMENT_DAYS_MEAN_LAST1Y is only for the previous @prod (version 11).
"""
import pandas as pd
import numpy as np

# The raw installments_payments.csv columns build_features() reads (batch scoring loads only these)
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
    Aggregate each applicant's repayment history into one row.

    valid_ids: the SK_ID_CURR values being scored. The file holds every
    applicant Home Credit has ever seen, so it's cut down to these first.
    """
    Payments = inst_df[inst_df['SK_ID_CURR'].isin(valid_ids)].copy()
    Payments = Payments.astype({'SK_ID_CURR': 'string', 'SK_ID_PREV': 'string'})

    # Month of each instalment relative to the application (-1 = last month),
    # from its due date in days
    Payments['MONTHS_BALANCE'] = np.floor(Payments['DAYS_INSTALMENT'] / 30).astype(int)
    Payments.sort_values(
        by=['SK_ID_CURR', 'SK_ID_PREV', 'MONTHS_BALANCE', 'DAYS_ENTRY_PAYMENT'],
        inplace=True,
        ignore_index=True,
    )

    # Fill a missing payment date/amount from the nearest payment on the same
    # loan (up to 6 rows away, earlier first), as the notebook does
    for col in ['DAYS_ENTRY_PAYMENT', 'AMT_PAYMENT']:
        Payments[col] = (
            Payments.groupby('SK_ID_PREV')[col]
            .transform(lambda x: x.ffill(limit=6).bfill(limit=6))
        )

    # Days paid after the due date (negative = paid early)
    Payments['LATE_PAYMENT_DAYS'] = Payments['DAYS_ENTRY_PAYMENT'] - Payments['DAYS_INSTALMENT']
    # Amount paid / amount due (1 = paid in full)
    Payments['PAYMENT_AMT_TO_INSTALLMENT_RATIO'] = np.where(
        Payments['AMT_INSTALMENT'] > 0,
        Payments['AMT_PAYMENT'] / Payments['AMT_INSTALMENT'],
        np.nan,
    )

    # One row per applicant per month, keeping the month's worst value
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

    # Summaries over recent periods (notebook Section 9):
    #   lateness: mean and max over the last 12 months, mean over the last 6
    #   share paid: mean over the last 24 months
    Payments_Mon_1Y_Agg = (
        Payments_Mon[Payments_Mon['MONTHS_BALANCE'] >= -12]
        .groupby('SK_ID_CURR')['PMNT_INSTLMT_LATE_PAYMENT_DAYS']
        .agg(['mean', 'max'])
        .reset_index()
        .rename(columns={
            'mean': 'PMNT_INSTLMT_LATE_PAYMENT_DAYS_MEAN_LAST1Y',
            'max': 'PMNT_INSTLMT_LATE_PAYMENT_DAYS_MAX_LAST1Y',
        })
    )
    Payments_Mon_6M_Agg = (
        Payments_Mon[Payments_Mon['MONTHS_BALANCE'] >= -6]
        .groupby('SK_ID_CURR')['PMNT_INSTLMT_LATE_PAYMENT_DAYS']
        .mean()
        .reset_index(name='PMNT_INSTLMT_LATE_PAYMENT_DAYS_MEAN_LAST6M')
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
        .merge(Payments_Mon_6M_Agg, on='SK_ID_CURR', how='left')
        .merge(Payments_Mon_2Y_Agg, on='SK_ID_CURR', how='left')
    )
    return Payments_final
