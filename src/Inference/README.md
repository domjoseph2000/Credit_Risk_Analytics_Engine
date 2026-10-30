# Home Credit Default — Inference Pipeline

## Design goal

One shared feature/scoring path, two entrypoints:

```
                  ┌─────────────────────────┐
                  │      src/Inference      │
                  │  score_model_input.py   │  <- score_applicants()
                  │  model.py               │  <- MLflow load + predict_proba()
                  │  pdo_scorecard.py       │  <- PDO transform (ports pd_to_score)
                  │  explainability.py      │  <- SHAP-based points per attribute
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
  end of whichever `02_training_LightGBM_vN.ipynb` is currently promoted
  (`02_training_LightGBM_v4.ipynb` as of the last promotion). The model is loaded
  through the docker-compose `mlflow` tracking server (`MLFLOW_TRACKING_URI`
  in `config.py`, default `http://localhost:5000`; use
  `http://mlflow:5000` from inside a compose container). The server keeps
  metadata in SQLite (`DATA_ROOT/mlflow.db`, migrated from the old file
  store via `scripts/migrate_mlflow_to_sqlite.py`) and serves the model
  files itself, so inference needs network access to the server, not
  filesystem access to `DATA_ROOT`. Start it with
  `docker compose up -d mlflow`. Versions registered before 2026-10-14
  (including the current `@prod`) still point at `file:///data/mlruns/...`
  and load only where `/data` exists (see the top-level `README.md`'s
  "Known gaps").

- **Feature list**: read directly from `selected_features.xlsx` (`feature`
  column) at the path `config.SELECTED_FEATURES_PATH` points to
  (`src/Inference/selected_features.xlsx` by default). This is a copy of
  the currently-promoted model's `lgbm_feature_importance.xlsx` `feature`
  column (`data/Model-Training/ModelOutput/<model_folder>/`) — re-copy it
  whenever a new model is promoted to `@prod`, and rebuild
  `src/Inference/features/*.py` to match if the new model's feature list
  pulls from a raw table or aggregation window the current modules don't
  produce.

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

- **`src/Inference/features/*.py`** (`application.py`, `bureau.py`,
  `previous_application.py`, `pos_cash.py`, `credit_card.py`,
  `installments.py`) — Built for `LightGBM_v4`'s 24-feature list. Each module 
  computes   only the columns the currently-promoted model actually uses, 
  mirroring the exact  aggregation logic in the matching `src/Training/01_Preprocess/*.ipynb` 
  notebook --   verified against `data/Model-Training/ModelDataset/model_dataset_final.parquet`
  at 100% value match. The pre-v4 modules are kept for reference in
  `src/Inference/features_archive_v3/`. When a new model is promoted, these
  need rebuilding again to match its feature list -- diff the new model's
  `lgbm_feature_importance.xlsx` against the current
  `selected_features.xlsx` to see what changed.

- **Real-time API feature coverage**: FIXED -- `realtime_api.py` used to
  only build features from the submitted application fields, and since 13
  of `LightGBM_v4`'s 24 features need bureau/previous-loan/POS/
  installments/credit-card history, every `/score` call 422'd. It now
  loads the same shared history tables batch scoring reads from
  `DATA_ROOT/Input` once at startup (`load_shared_history()`, cached), and
  filters them to the requesting `SK_ID_CURR` per call, exactly like
  batch's per-table `build_*_features` functions filter to a whole batch's
  IDs. This only produces real values for an `SK_ID_CURR` that already has
  history in those tables (an existing applicant); a genuinely new
  `SK_ID_CURR` gets `NaN` for all 13 history-derived features instead of a
  422 -- LightGBM handles that natively, no imputation needed. Verified
  against real data for both cases (see git history for the verification
  script) -- application-only features identical between an existing and a
  synthetic brand-new applicant sharing the same form values; all 13
  history columns correctly `NaN` for the brand-new one, no crashes in any
  of the five feature modules' aggregation on the resulting empty groups.
  
- **`schemas.py`** only stubs a handful of `application.csv` fields —
  extend once you know which raw columns actually feed into the upstream
  feature engineering.

- **`explainability.py`** — per-attribute points contributions ("reason
  codes"), derived from SHAP values, not a separate model or heuristic.
  `pdo_scorecard.proba_to_score()` is a *linear* function of the model's
  raw margin, and SHAP values are an *exact* additive decomposition of
  that same margin, so `points_i(x) = -factor * shap_i(x)` sums exactly
  (verified empirically to ~1e-7 float noise) to that applicant's score
  minus a constant `REFERENCE_SCORE`. This is purely an explanation layer
  computed from the already-trained model *after* scoring — it never
  feeds back into `score_applicants()` or `proba_to_score()`, so it
  cannot change anyone's actual score or band, and a failure in this path
  is a bug in the explanation only, never in the score itself.
  `load_explainer()` gets the raw LightGBM model out of the registered
  pyfunc wrapper via MLflow's `unwrap_python_model()` (`shap.TreeExplainer`
  needs the actual tree ensemble, not a `predict()` callable). Batch output
  (`reason_codes.parquet`) includes every feature's points; the real-time
  API's `REASON_CODES` field is trimmed to the top 5 by `|points|`
  (`top_reason_codes()`) so the JSON response doesn't dump all ~24 columns.

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
- `DATA_ROOT/Inference/BatchOutput/<date>/reason_codes.parquet` -- per-attribute
  points for every applicant in the batch, see `explainability.py`

The Airflow DAG (`inference_batch_dag.py`) runs this exact same command
(`--application-date {{ ds }}`) as a 5th DAG alongside your existing
four — waits for the day's `application.csv` to land, then runs scoring.
It's independent of the training/HPO DAGs, so a model retrain/promotion
doesn't need to block daily scoring, and a bad day of scoring doesn't
block retraining.

### Batch monitoring dashboard

```bash
uvicorn batch_dashboard_api:app --host 0.0.0.0 --port 8001
```

```
http://localhost:8001/ui/
```

Reads every `DATA_ROOT/Inference/BatchOutput/<date>/scored_applicants.parquet`
that exists (`batch_metrics.py`) and shows the selected run's volume, mean
score, and mean PD against **t-1** -- the previous *completed* run in that
list, not literally "yesterday" (batches don't have to run daily), plus:

- **Population Stability Index** on `PROBA_DEFAULT`, t-1 as the expected/
  baseline distribution vs the selected run as actual -- `calculate_psi()`
  in `batch_metrics.py` is a direct port of the train-vs-test PSI check in
  `02_training_LightGBM_v4.ipynb` section 7.d, so batch drift monitoring uses the
  exact same formula and 0.10 / 0.25 stable/moderate/significant thresholds.
- Risk band mix (E-A) for the selected run vs t-1, with the percentage-point
  shift per band.
- A volume and mean-score trend across *all* available runs, not just the
  two being compared.

Deliberately a separate app/port from `realtime_api.py`: it only reads
files `score_model_input.py` already wrote, so it has no model/SHAP/MLflow
dependency, starts instantly, and keeps working even when the real @prod
model can't be loaded (see the real-time API's known gap above). With only
one run on disk, comparison metrics show "No prior run to compare" instead
of erroring -- expected on a fresh deployment before a second batch runs.

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

Model, SHAP explainer, and shared history tables all load once at startup
(`lru_cache`), not per-request. Response includes `REASON_CODES` -- the
top 5 attributes by `|points|` for this applicant, see `explainability.py`.

History lookup: the API filters the same shared history tables (bureau/
previous-loan/POS/installments/credit-card, loaded once from
`DATA_ROOT/Input`) to the request's `SK_ID_CURR`, the same way batch
scoring filters them per-batch -- so this only produces real
history-derived feature values for an `SK_ID_CURR` that already has
history on file. A genuinely new `SK_ID_CURR` (not in any of those
tables) still scores successfully, just with `NaN` for the 13
history-derived features (LightGBM handles that natively) instead of a
`422`. If a request is missing a feature the model needs for some other
reason, you still get a `422` with the specific missing columns, not a
silent bad prediction. `ApplicationRequest` fields left `null` are swapped
for `np.nan` before scoring (not left as Python `None`) -- a single-row
DataFrame built straight from `None` values gets `object` dtype columns,
which LightGBM's `predict()` rejects outright, so leaving e.g. `OWN_CAR_AGE`
unset (a very ordinary case -- no car) would otherwise 422 on that alone.

Response also includes `HAS_CREDIT_HISTORY`: whether this `SK_ID_CURR`
matched any row in the shared history tables. Don't infer this from
`REASON_CODES` instead -- SHAP can legitimately surface a NaN history
feature as a top contributor (it reflects the model's learned
default-split behavior for missing values, not a real value for this
applicant), so a history feature's presence there doesn't mean the
applicant actually has history.

### Web UI

```
http://localhost:8000/ui/
```

A single static page (`static/index.html`, served by this same FastAPI
app -- no separate server, no build step) with a form for every field
`features/application.py` actually consumes, two "load example" buttons
(a real applicant with history vs. a brand-new `SK_ID_CURR` with none),
and a result panel: a speedometer-style risk gauge (colored by
`config.BAND_BINS`, needle at the returned score), the score/band/PD, and
a diverging bar chart of `REASON_CODES`. It POSTs to `/score` on the same
origin, so no CORS setup is needed. `GET /` redirects to `/ui/`.

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
