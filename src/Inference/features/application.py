"""
Features from the application form itself (application.csv), one row per applicant.

Mirrors src/Training/01_Preprocess/Application_data.ipynb, so a feature here
is calculated exactly as it was for training.

The @prod model (MLflow version 23) uses 10 of the columns returned here:
AGE, DAYS_EMPLOYED, YEARS_SINCE_PHONE_CHANGE, DAYS_ID_PUBLISH, EXT_SOURCE_1/2/3,
OCCUPATION_TYPE, ORGANIZATION_TYPE and GOODSPRICE_TO_AMT_CREDIT.
EMPLOYED_YEARS, FLOORSMAX_MEDI and OWN_CAR_AGE are only used by the previous
@prod (version 11). EXT_SCORE_MEAN (average of the three external scores),
CODE_GENDER and NAME_EDUCATION_TYPE are used by the LightGBM_v4 retrained on
2026-11-21 (not promoted yet). features/pipeline.py keeps just the columns the current
model needs. The full list is in
data/Model-Training/HyperParameters/LightGBM_v4/selected_features.json.
"""
import numpy as np
import pandas as pd

# The raw application.csv columns build_features() reads. Batch scoring loads
# only these columns, to save memory.
REQUIRED_COLUMNS = [
    'SK_ID_CURR',
    'DAYS_BIRTH',
    'DAYS_EMPLOYED',
    'DAYS_ID_PUBLISH',
    'DAYS_LAST_PHONE_CHANGE',
    'EXT_SOURCE_1',
    'EXT_SOURCE_2',
    'EXT_SOURCE_3',
    'FLOORSMAX_MEDI',
    'OWN_CAR_AGE',
    'AMT_CREDIT',
    'AMT_GOODS_PRICE',
    'OCCUPATION_TYPE',
    'ORGANIZATION_TYPE',
    'CODE_GENDER',
    'NAME_EDUCATION_TYPE',
]


def build_features(app_df: pd.DataFrame) -> pd.DataFrame:
    """Clean the raw application rows and derive the model's application features."""
    application_train = app_df.copy()
    application_train = application_train.astype({'SK_ID_CURR': 'string'})

    # --- Placeholder value 365243 -> missing ---
    # The data uses 365243 days (~1,000 years) to mean "not applicable", e.g.
    # DAYS_EMPLOYED for pensioners. As in the notebook, every numeric column
    # is checked, not a fixed list.
    SENTINEL = 365243
    numeric_cols = application_train.select_dtypes(include='number').columns
    sentinel_cols = [col for col in numeric_cols if (application_train[col] == SENTINEL).any()]
    for col in sentinel_cols:
        application_train.loc[application_train[col] == SENTINEL, col] = np.nan

    # --- Placeholder text 'XNA' (not available) -> missing ---
    object_cols = application_train.select_dtypes(include='object').columns
    xna_cols = [col for col in object_cols if (application_train[col] == 'XNA').any()]
    for col in xna_cols:
        application_train.loc[application_train[col] == 'XNA', col] = np.nan

    # --- New features ---
    # The DAYS_* columns count days before the application, as negative
    # numbers, so dividing by -365 gives positive years.
    application_train['AGE'] = application_train['DAYS_BIRTH'] / -365
    application_train['EMPLOYED_YEARS'] = application_train['DAYS_EMPLOYED'] / -365
    application_train['YEARS_SINCE_PHONE_CHANGE'] = application_train['DAYS_LAST_PHONE_CHANGE'] / -365
    # Price of the goods bought / loan amount (below 1 = borrowing more than the goods cost)
    application_train['GOODSPRICE_TO_AMT_CREDIT'] = np.where(
        application_train['AMT_CREDIT'] > 0,
        application_train['AMT_GOODS_PRICE'] / application_train['AMT_CREDIT'],
        np.nan,
    )
    # Average of the three external credit scores, using whichever are present
    # (blank only if all three are missing). Not used by the current @prod
    # model; here so a model trained with it can be scored.
    application_train['EXT_SCORE_MEAN'] = application_train[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].mean(axis=1)

    final_attributes = [
        'SK_ID_CURR',
        'AGE',
        'DAYS_EMPLOYED',
        'EMPLOYED_YEARS',
        'YEARS_SINCE_PHONE_CHANGE',
        'DAYS_ID_PUBLISH',
        'EXT_SOURCE_1',
        'EXT_SOURCE_2',
        'EXT_SOURCE_3',
        'EXT_SCORE_MEAN',
        'FLOORSMAX_MEDI',
        'OWN_CAR_AGE',
        'OCCUPATION_TYPE',
        'ORGANIZATION_TYPE',
        'CODE_GENDER',
        'NAME_EDUCATION_TYPE',
        'GOODSPRICE_TO_AMT_CREDIT',
    ]
    return application_train[final_attributes]
