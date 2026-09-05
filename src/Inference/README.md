# Home Credit Default — Inference Pipeline

## Design goal

One shared feature/scoring path, two entrypoints:

```
                  ┌─────────────────────────┐
                  │      src/Inference      │
                  │  score_model_input.py   │  <- score_applicants()
                  │  model.py               │  <- MLflow load + predict_proba()
                  │  pdo_scorecard.py       │  <- PDO transform (ports pd_to_score)
                  │  features/pipeline.py   │  <- assemble_features()
                  └───────────┬─────────────┘
                  ┌───────────┴─────────────────────────┐
    score_model_input.py --application-date        realtime_api.py
    (run manually or by Airflow)                        │
                  │                            POST /score (1 applicant)
          Batch_Inference.ipynb
        (interactive wrapper, same import)
```

Both call `score_model_input.score_applicants()`, which returns a
DataFrame with columns `SK_ID_CURR`, `PROBA_DEFAULT`, `score`, `Risk_Band`
— the last two named to match `Credit_Scoring.ipynb` so this output can
be concatenated directly with your scorecard analysis if useful.

There used to be a separate `Batch_Inference.py` CLI script; it was
merged into `score_model_input.py` since both scripts read the same
dated-application / static-history data layout and called the same
`score_applicants()` -- one script now serves manual runs and the
Airflow DAG identically (`--application-date`). `Batch_Inference.ipynb`
still exists for interactive/exploratory runs; it imports `score_applicants`
from `score_model_input.py` rather than reimplementing it, so it can't
drift from what the CLI/Airflow path actually does.

## What's already pinned to your notebooks

- **Model**: loads `models:/HomeCredit_Scorecard_Model@prod` — alias-based,
  matching the manual "assign the prod alias in the MLflow UI" step at the
  end of `Model_LightGBM_v3.ipynb`. Whatever process runs inference needs
  filesystem access to the local `mlruns` file store (`MLFLOW_TRACKING_URI`
  in `config.py`) since that notebook uses a file-based store, not a server.

- **Feature list**: read directly from `selected_features.xlsx` (`feature`
  column) at the path `config.SELECTED_FEATURES_PATH` points to
  (`src/Inference/selected_features.xlsx` by default). Nothing to
  export — it already exists once that notebook has run.

- **PDO transform**: `pdo_scorecard.py` is a direct port of `pd_to_score` from
  `Credit_Scoring.ipynb` — same epsilon, same int rounding, same
  `[300, 850]` clip.

- **Risk bands**: same A–E scheme, same fixed cutoffs (560/590/620/650),
  same `right=False` binning as `Credit_Scoring.ipynb` section 5.

- **Model's `predict()` output**: `LightGBMProductionWrapper` always
  returns a plain list of `P(default)`, one per row — `model.py` relies on
  that directly rather than guessing at output shape.

- **Categorical columns**: the wrapper re-casts its stored
  `categorical_features` to `category` dtype internally on every call, so
  nothing extra is needed here — just get the right column names/values in.

- **`src/Inference/features/*.py`** (`application.py`,
  `bureau.py`, `previous_application.py`, `pos_cash.py`, `credit_card.py`,
  `installments.py`). For inference to produce features consistent with what the
  model was trained on, this needs the exact aggregation logic from
  model training notebooks.
  
- **`schemas.py`** only stubs a handful of `application.csv` fields —
  extend once you know which raw columns actually feed into the upstream
  feature engineering.

## Batch inference — manual and Airflow (same script)

```bash
python score_model_input.py --application-date 2026-07-31
```

or interactively via `Batch_Inference.ipynb` (set `RUN_DATE` in cell 1,
Run All) — same feature assembly, same model, same PDO/band logic, just
with a band-distribution sanity check inline as you go.

Reads `application.csv` from a dated `Inference/BatchInput/<date>/` dir
and the shared bureau/previous-application/POS/installments history from
`Input/` (not date-partitioned — that history doesn't need to be
re-dumped every run, only new applications do). Writes:
- `DATA_ROOT/Inference/BatchOutput/<date>/scored_applicants.parquet`
- `DATA_ROOT/Inference/BatchOutput/<date>/band_summary.csv`

The Airflow DAG (`inference_batch_dag.py`) runs this exact same command
(`--application-date {{ ds }}`) as a 5th DAG alongside your existing
four — waits for the day's `application.csv` to land, then runs scoring.
It's independent of the training/HPO DAGs, so a model retrain/promotion
doesn't need to block daily scoring, and a bad day of scoring doesn't
block retraining.

## Real-time API

```bash
uvicorn realtime_api:app --host 0.0.0.0 --port 8000
```

```bash
curl -X POST localhost:8000/score -H "Content-Type: application/json" -d '{
  "SK_ID_CURR": 100001,
  "AMT_INCOME_TOTAL": 180000,
  "AMT_CREDIT": 450000,
  "AMT_ANNUITY": 22000,
  "DAYS_BIRTH": -14000,
  "DAYS_EMPLOYED": -2500
}'
```

Model loads once at startup (`lru_cache`), not per-request. If a request
is missing a feature the model needs, you get a `422` with the specific
missing columns, not a silent bad prediction.

Note: as written, the API only builds features from the main application
table — no bureau/previous-loan history, since that's not something a
caller submitting a single new application would have on hand. If your
model relies heavily on those aggregates, either (a) look history up from
a DB/feature store keyed on `SK_ID_CURR` inside the API before calling
`assemble_features`, or (b) accept that real-time scoring uses a reduced
feature set and validate that doesn't materially hurt AUC before shipping it.

## Testing

```bash
pip install -r requirements.txt
cd src/Inference && pytest
```

`test_inference_smoke.py` checks the plumbing (join logic, feature
contract enforcement, PDO math against known base-odds/clip behavior,
A–E band edges) without needing a real MLflow model or real data.

`test_history_store.py` exercises a `history_store` module that doesn't
exist in this package yet -- it'll fail on import (`ModuleNotFoundError`)
until that module is written or the test file is removed.
