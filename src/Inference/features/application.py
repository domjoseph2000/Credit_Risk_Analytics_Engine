"""
Feature engineering for the main application table.

Replace the body of `build_features` with the exact logic from your
notebook for preprocessing. Keep the function signature the same
(raw DataFrame in, feature DataFrame out with SK_ID_CURR preserved) so
it's a drop-in swap and nothing downstream has to change.
"""
import numpy as np
import pandas as pd

# Raw application.csv columns build_features() actually reads. Kept in sync
# with the anomaly-handling/derived-feature logic below so callers (batch
# loaders, the notebook) can `pd.read_csv(..., usecols=REQUIRED_COLUMNS)`
# instead of pulling in all ~120 raw application columns.
REQUIRED_COLUMNS = [
    'SK_ID_CURR',
    'DAYS_EMPLOYED',
    'DAYS_LAST_PHONE_CHANGE',
    'DAYS_BIRTH',
    'DAYS_ID_PUBLISH',
    'EXT_SOURCE_1',
    'EXT_SOURCE_2',
    'EXT_SOURCE_3',
    'FLOORSMAX_MEDI',
    'OWN_CAR_AGE',
    'AMT_CREDIT',
    'AMT_GOODS_PRICE',
    'OCCUPATION_TYPE',
    'ORGANIZATION_TYPE',
]


def build_features(app_df: pd.DataFrame) -> pd.DataFrame:
    application_train = app_df.copy()
    application_train = application_train.astype({'SK_ID_CURR': 'string'})
    # --- anomaly handling 365243---
    SENTINEL = 365243
    numeric_cols = [
        'DAYS_EMPLOYED',
        'DAYS_LAST_PHONE_CHANGE',
        'DAYS_BIRTH',
        'DAYS_ID_PUBLISH',
        'EXT_SOURCE_1',
        'EXT_SOURCE_2',
        'EXT_SOURCE_3',
        'FLOORSMAX_MEDI',
        'OWN_CAR_AGE',
        'AMT_CREDIT',
        'AMT_GOODS_PRICE'
    ]
    sentinel_cols = [col for col in numeric_cols if (application_train[col] == SENTINEL).any()]
    print(f"Columns containing sentinel {SENTINEL}: {sentinel_cols}")

    for col in sentinel_cols:
        application_train.loc[application_train[col] == SENTINEL, col] = np.nan

    # --- anomaly handling XNA---
    xna_cols = [
        'OCCUPATION_TYPE',
        'ORGANIZATION_TYPE'
    ]
    print(f"Columns containing 'XNA': {xna_cols}")

    for col in xna_cols:
        application_train.loc[application_train[col] == 'XNA', col] = np.nan

    # --- Variable creation ---
    application_train['AGE'] = application_train['DAYS_BIRTH'] / -365
    
    application_train['GOODSPRICE_TO_AMT_CREDIT'] = np.where(
        application_train['AMT_CREDIT'] > 0,
        application_train['AMT_GOODS_PRICE'] / application_train['AMT_CREDIT'],
        np.nan
    )
    application_train['YEARS_SINCE_PHONE_CHANGE'] = (
        application_train['DAYS_LAST_PHONE_CHANGE'] / -365
    )
    application_train['EMPLOYED_YEARS'] = application_train['DAYS_EMPLOYED'] / -365

    final_attributes = [
        'SK_ID_CURR',
        'AGE',
        'DAYS_EMPLOYED',
        'DAYS_ID_PUBLISH',
        'EMPLOYED_YEARS',
        'EXT_SOURCE_1',
        'EXT_SOURCE_2',
        'EXT_SOURCE_3',
        'FLOORSMAX_MEDI',
        'GOODSPRICE_TO_AMT_CREDIT',
        'OCCUPATION_TYPE',
        'ORGANIZATION_TYPE',
        'OWN_CAR_AGE',
        'YEARS_SINCE_PHONE_CHANGE'
    ]
    application_train_final = application_train[final_attributes]
    return application_train_final
