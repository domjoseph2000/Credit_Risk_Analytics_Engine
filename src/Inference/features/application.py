"""
Feature engineering for the main application table.

Rebuilt for the LightGBM_v4 (@prod) feature set -- see
data/Model-Training/ModelOutput/LightGBM_v4/lgbm_feature_importance.xlsx for
the ground truth feature list, and src/Preprocess/Application_data.ipynb for
the logic this mirrors. Only computes the 11 application-derived columns v4
actually uses; anything else Application_data.ipynb computes is intentionally
left out.
"""
import numpy as np
import pandas as pd

# Raw application.csv columns build_features() actually reads.
REQUIRED_COLUMNS = [
    'SK_ID_CURR',
    'DAYS_BIRTH',
    'DAYS_EMPLOYED',
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

    # --- anomaly handling 365243 ---
    # Notebook scopes this to every numeric column dynamically
    # (select_dtypes), not a fixed list -- doing the same here scopes it
    # automatically to whichever REQUIRED_COLUMNS are present.
    SENTINEL = 365243
    numeric_cols = application_train.select_dtypes(include='number').columns
    sentinel_cols = [col for col in numeric_cols if (application_train[col] == SENTINEL).any()]
    for col in sentinel_cols:
        application_train.loc[application_train[col] == SENTINEL, col] = np.nan

    # --- anomaly handling XNA ---
    object_cols = application_train.select_dtypes(include='object').columns
    xna_cols = [col for col in object_cols if (application_train[col] == 'XNA').any()]
    for col in xna_cols:
        application_train.loc[application_train[col] == 'XNA', col] = np.nan

    # --- Variable creation ---
    application_train['AGE'] = application_train['DAYS_BIRTH'] / -365
    application_train['EMPLOYED_YEARS'] = application_train['DAYS_EMPLOYED'] / -365
    application_train['GOODSPRICE_TO_AMT_CREDIT'] = np.where(
        application_train['AMT_CREDIT'] > 0,
        application_train['AMT_GOODS_PRICE'] / application_train['AMT_CREDIT'],
        np.nan,
    )

    final_attributes = [
        'SK_ID_CURR',
        'AGE',
        'EMPLOYED_YEARS',
        'DAYS_ID_PUBLISH',
        'EXT_SOURCE_1',
        'EXT_SOURCE_2',
        'EXT_SOURCE_3',
        'FLOORSMAX_MEDI',
        'OWN_CAR_AGE',
        'OCCUPATION_TYPE',
        'ORGANIZATION_TYPE',
        'GOODSPRICE_TO_AMT_CREDIT',
    ]
    return application_train[final_attributes]
