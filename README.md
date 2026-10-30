# Home Credit — Architecture & Pipeline Guide

Default-prediction ML system for Home Credit loan applications. This doc covers
the overall architecture and how to run both the **training pipeline**
(feature engineering → dataset → variable selection → hyperparameter tuning →
model training → scorecard) and the **inference pipeline** (batch and
real-time scoring of a trained model).

## 1. High-level architecture

![Architecture diagram](architecture_diagram_preview.png)

An interactive version of this diagram (clickable components, hover states) is
in [architecture_diagram.html](architecture_diagram.html) — open it directly in
a browser.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                              TRAINING SIDE                              │
│                                                                         │
│  data/Input/*.csv                                                       │
│       │  (application_train.csv, bureau.csv, bureau_balance.csv,        │
│       │   previous_application.csv, POS_CASH_balance.csv,               │
│       │   credit_card_balance.csv, installments_payments.csv)           │
│       ▼                                                                 │
│  src/Training/01_Preprocess/*.ipynb ──► data/Model-Training/Staging/    │
│       ▼                                                                 │
│  src/Training/02_Variable-Selection/select_vars.ipynb (feature scoring) │
│       ▼                                                                 │
│  src/Training/03_MDS-Creation/model_dataset_creation.ipynb              │
│       ▼                                                                 │
│  data/Model-Training/ModelDataset/model_dataset_{initial,final}.parquet │
│       ▼                                                                 │
│  src/Training/02_Variable-Selection/01b_multicollinearity_reduction     │
│       ▼                                                                 │
│  src/Training/04_Model-Development/<version>/                           │
│       01_tuning_<version>.ipynb  ──►  02_training_<version>.ipynb       │
│       ▼                                                                 │
│  MLflow server (mlflow.db + mlruns) — "HomeCredit_Scorecard_Model"      │
│       ▼                                                                 │
│  src/Training/05_Segmentation-Analysis, 06_Scoring (PDO scorecard)      │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    │  model promoted to @prod alias
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                             INFERENCE SIDE                              │
│                                                                         │
│  src/Inference/features/*.py  ──►  assemble_features()                  │
│       ▲                                    │                            │
│       │                                    ▼                            │
│  data/Input (shared history)      src/Inference/model.py (MLflow load)  │
│  data/Inference/BatchInput        src/Inference/pdo_scorecard.py        │
│       │                                    │                            │
│       ▼                                    ▼                            │
│  score_model_input.py (batch, CLI/Airflow)   realtime_api.py (FastAPI)  │
│       ▼                                                                 │
│  data/Inference/BatchOutput/<date>/scored_applicants.parquet            │
└─────────────────────────────────────────────────────────────────────────┘
```

Orchestration for both sides runs on **Apache Airflow** (standalone mode,
`SequentialExecutor` + SQLite) inside one Docker container defined by
[`Dockerfile`](Dockerfile) and wired up by [`docker-compose.yml`](docker-compose.yml).
A second container runs the **MLflow tracking server** (UI + API at
`localhost:5000`): run records live in `data/mlflow.db` (SQLite) and model
files/plots in `data/mlruns`, and every client — Airflow, notebooks on your
own machine, inference — saves and loads through the server rather than
touching those folders directly (see §6.10).

## 2. Directory layout

```
data/
  Input/                        Raw Kaggle CSVs + shared credit-history CSVs (also read by inference)
  mlflow.db                     MLflow run records + model registry (SQLite; written by the MLflow server)
  mlruns/                       MLflow files: run artifacts (<run_id>/) and models (models/m-<id>/)
  Model-Training/
    Staging/                    Per-table processed parquets (Preprocess notebooks output)
    ModelDataset/                model_dataset_initial.parquet, model_dataset_final.parquet
    Variable-Selection/          VIF/multicollinearity outputs
    HyperParameters/             Hyperopt trial results
    ModelOutput/<model_folder>/  Trained model artifacts, executed notebooks
    ScoreOutput/<model_folder>/  Credit_Scoring.ipynb outputs — final model only (05_scoring)
    notebook_outputs/            Executed (papermill) copies of every DAG-run notebook, for audit
  Inference/
    BatchInput/<date>/            application.csv for that batch (date-partitioned)
    BatchFeatures/<date>/         Per-table feature parquets + joined feature_matrix.parquet
    BatchOutput/<date>/           scored_applicants.parquet, band_summary.csv

src/
  Training/
    01_Preprocess/              Application_data, Bureau_data, CreditCard_data, POS_data,
                                 Payment_installments_data, Prevloans_data (.ipynb)
    02_Variable-Selection/      select_vars.ipynb, 01b_multicollinearity_reduction.ipynb
    03_MDS-Creation/            model_dataset_creation.ipynb
    04_Model-Development/       One folder per model version (LightGBM_v1..v4/, XGB_v1/), each with
                                 01_tuning_<version>.ipynb and 02_training_<version>.ipynb
    05_Segmentation-Analysis/   Segmentation_Analysis.ipynb
    06_Scoring/                 Credit_Scoring.ipynb (PDO scorecard, calibration, KS, PSI)
  Inference/                    Production-facing Python package (see §5)

dags/                           Airflow DAG definitions (live-synced into the container)
scripts/
  entrypoint.sh                 Container startup: creates dirs, inits Airflow DB, starts standalone
  start_mlflow_server.py        Starts the MLflow server on your own machine (no Docker) — §6.9
  point_experiment_at_mlflow_server.py   One-time MLflow database fix (already applied) — §6.10
  migrate_mlflow_to_sqlite.py   One-time file-store → SQLite migration (already applied)
requirements.txt                Python packages for training + MLflow (also installed by the Dockerfile)
Dockerfile, docker-compose.yml  Container build/run definition
```

`data/Input`, `data/mlflow.db` and `data/mlruns` are deliberately kept at the top level of
`data/` — they're shared between the training and inference sides.
Everything else training-related lives under `data/Model-Training/`.

## 3. Orchestration (Airflow DAGs)

All DAGs live in [`dags/`](dags/) — the only directory `docker-compose.yml`
live-syncs into Airflow's DAG folder (`./dags:/opt/airflow/dags`), so edits
there are picked up without a rebuild. Each DAG resolves `HOMECREDIT_DATA_ROOT`
(`/data`) and `HOMECREDIT_SRC_DIR` (`/src`) from environment variables set in
`docker-compose.yml`.

DAGs are numbered in pipeline order instead of sharing a `homecredit_`
prefix, so the run order is visible directly in the Airflow DAGs list.

| DAG id | File | Purpose |
|---|---|---|
| `01_preprocessing` | [01_preprocessing_dag.py](dags/01_preprocessing_dag.py) | Runs the 6 Preprocess notebooks via papermill, one task per raw table, in parallel where there's no dependency. By default, triggers `02_variable_selection` once all tasks succeed — set the `trigger_variable_selection` param to `false` to stop after preprocessing. |
| `02_variable_selection` | [02_variable_selection_dag.py](dags/02_variable_selection_dag.py) | Runs `select_vars.ipynb`. Split out of `01_preprocessing` (was that DAG's `task_select_vars`) so it can be retried/skipped independently. By default, triggers `03_mds_creation_and_multicollinearity_test` once it finishes — set the `trigger_mds_creation` param to `false` to stop after variable selection. |
| `03_mds_creation_and_multicollinearity_test` | [03_mds_creation_and_multicollinearity_test_dag.py](dags/03_mds_creation_and_multicollinearity_test_dag.py) | Runs `model_dataset_creation.ipynb`, then `01b_multicollinearity_reduction.ipynb` (VIF-based feature pruning — previously a manual step run outside any DAG); then triggers `04_model_pipeline` for LightGBM_v1 — with a full hyperparameter search if the `run_hyperopt` toggle is on, otherwise with `tuning='previous'`. |
| `04_model_pipeline` | [04_model_pipeline_dag.py](dags/04_model_pipeline_dag.py) | Builds and evaluates one model version: `check_inputs` → tuning notebook → training notebook → `Segmentation_Analysis.ipynb`. Pick the version with `model_folder`; the DAG looks up both notebooks itself. See below for params. |
| `05_scoring` | [05_scoring_dag.py](dags/05_scoring_dag.py) | Builds the scorecard (`Credit_Scoring.ipynb`: PDO score, A–E risk bands, KS) for the **final model only** — run it manually once you've compared versions and picked one. No retraining: reads that version's saved `test_predictions.parquet`. Its `model_folder` dropdown lists the versions that have been trained. |
| `home_credit_batch_inference` | [inference_batch_dag.py](dags/inference_batch_dag.py) | Batch-scores a dated applicant file: one task per raw table → join → score. |

All numbered training DAGs execute notebooks via
`papermill.execute_notebook()` (each notebook self-configures from env vars
— no parameters are injected) and save the executed copy to
`data/Model-Training/notebook_outputs/` for audit/debugging
(`04_model_pipeline` and `05_scoring` name them `<task>__<model_folder>__<date>.ipynb`).

**`04_model_pipeline` params** (trigger screen, or `--conf`):

| Param | Options | Meaning |
|---|---|---|
| `model_folder` | `LightGBM_v1` (default) … `LightGBM_v4`, `XGB_v1` | Which version. The choices are discovered: every folder under `src/Training/04_Model-Development/`, which must hold exactly one `01_tuning*.ipynb` and one `02_training*.ipynb` — so adding a version (copy a folder, rename the notebooks) needs no DAG edit |
| `tuning` | `search` / `previous` (default) / `skip` | Full Hyperopt search (hours); reuse saved hyperparameters and refresh the feature list (minutes); or no tuning step (only allowed once that version's tuning outputs exist) |
| `run_training`, `run_segmentation` | true (default) / false | Stop early, e.g. `run_training=false` to only tune |

`check_inputs` stops the run up front with a clear message if the
combination can't work — e.g. `tuning='previous'` for a version that has
never been searched, or `tuning='skip'` before a feature list exists.

Tuning and training were two DAGs (`04_hyperparameter_tuning`,
`05_model_training`) until 2026-10-25. They were merged because training
needs the same version's tuning outputs (`selected_features.json`,
`*_resolved_best.json`), and the old hand-off between the DAGs only
forwarded `model_folder`, not the training notebook name — so automatic
chaining only worked for LightGBM_v1. On 2026-10-27 the merged DAG was
renumbered `04_model_pipeline` → `04_model_pipeline`, and scoring moved out
of it into `05_scoring`: the scorecard is only needed for the final model,
and scoring every version overwrote the chosen model's scorecard (outputs
now go to `ScoreOutput/<model_folder>/`).

**Note:** the model actually registered to MLflow and referenced by the
inference side (`config.SELECTED_FEATURES_PATH`, `verify_feature_parity.py`)
is **LightGBM_v3**, while `04_model_pipeline` defaults to LightGBM_v1 — pick
`model_folder` explicitly when (re)training the production model.

### DAG dependency chain (Preprocessing)

```
application_train.csv ──► task_application    ──┐
                                                │
previous_application.csv ► task_prevloans     ──┼──► task_pos
                                                ├──► task_creditcard
                                                └──► task_installments
bureau.csv              ┌─► task_bureau ────────────────┐
bureau_balance.csv  ────┘                               │
(all above) ──────────────────────────────────────► branch_task
                                                          │
                                trigger_variable_selection param (default true)
                                     ┌true───────────false┐
                                     ▼                    ▼
                     trigger 02_variable_selection   skip_variable_selection
                                     │
                                     ▼
                          select_vars (02_variable_selection DAG)
                                     │
                       trigger_mds_creation param (default true)
                                     ┌true───────────false┐
                                     ▼                    ▼
     trigger 03_mds_creation_and_multicollinearity_test   skip_mds_creation
                                     │
                                     ▼
                  create_mds ──► multicollinearity_reduction ──► branch_task
                     (03_mds_creation_and_multicollinearity_test DAG)
                                     │
                          run_hyperopt param (default false)
                                     ┌true───────────false┐
                                     ▼                    ▼
            trigger 04_model_pipeline          trigger 04_model_pipeline
            (LightGBM_v1, tuning=search)      (LightGBM_v1, tuning=previous)
                                     │
                                     ▼
           check_inputs ──► tune_model ──► train_model ──► run_segmentation
                 (04_model_pipeline DAG; each step can be skipped via params)

     ... compare versions, pick the final model, then trigger manually:

                         check_inputs ──► score_model
                       (05_scoring DAG, final model only)
```

### DAG chain (Batch Inference)

```
wait_for_raw_data (PythonSensor, polls for application.csv)
  ─► [build_application_features, build_bureau_features,
      build_previous_application_features, build_pos_cash_features,
      build_installments_features]   (parallel)
  ─► join_features   (merges the 5 parquets into feature_matrix.parquet)
  ─► score_features   (scores + writes scored_applicants.parquet, band_summary.csv)
```

`wait_for_raw_data` uses a `PythonSensor` (plain `Path.exists()` check), not
`FileSensor` — `FileSensor` requires an Airflow `fs_default` Connection that
isn't auto-created under Airflow 3.x's `airflow db migrate`.

## 4. Training pipeline — what each stage does

1. **Preprocess** (`src/Preprocess/*.ipynb`) — one notebook per raw table.
   Cleans sentinel values (e.g. `365243`, `XNA`), engineers ratios/flags, and
   aggregates history tables (Bureau, POS, CreditCard, Payment_installments)
   into rolling time windows (`_LAST3M/6M/1Y/2Y/3Y`) plus recent-vs-historical
   ratio features (`_LAST6M_TO_3Y_RATIO`, via `add_period_ratio_features()`
   in `CreditCard_data.ipynb`, reused in `POS_data.ipynb` and
   `Payment_installments_data.ipynb`). Output: one processed parquet per
   table in `data/Model-Training/Staging/`.
2. **Variable selection (first pass)** (`select_vars.ipynb`, run by its own
   `02_variable_selection` DAG, triggered by `01_preprocessing` by default) —
   computes per-feature AUC, t-test, Cohen's d, univariate logistic
   coefficient, VIF, and target correlation across all staged parquets;
   writes a scored Excel.
3. **MDS creation** (`model_dataset_creation.ipynb`, run by the
   `03_mds_creation_and_multicollinearity_test` DAG) — joins every processed
   parquet under `Staging/` on `SK_ID_CURR`, downcasts dtypes, and writes
   `model_dataset_initial.parquet`.
4. **Multicollinearity reduction** (`01b_multicollinearity_reduction.ipynb`,
   same DAG, runs right after MDS creation) — VIF-based feature pruning
   against `model_dataset_initial.parquet`, protecting features with strong
   predictive scores from step 2's output; writes `model_dataset_final.parquet`
   (initial file is left untouched).
5. **Hyperparameter tuning** (`src/Training/04_Model-Development/<model_folder>/01_tuning_<model_folder>.ipynb`)
   — each model version (LightGBM_v1–v4, XGB_v1) is one folder holding its
   tuning and training notebooks side by side. The tuning notebook is
   where each version's **feature selection rule** lives, and it chooses
   that version's **hyperparameters**, writing to
   `data/Model-Training/HyperParameters/<model_folder>/`:
   - `selected_features.json` — the feature list, the rule that produced it
     and when; written on every run. **The training notebook reads its
     features from this file**, so tuning and training always use the same
     list;
   - `*_resolved_best.json` — best hyperparameters from a `hyperopt` search
     (~150 trials, 5-fold CV on the 80% training split).

   For speed, the search's models are capped (`max_bin=63`, at most 500
   trees, early stopping after 20 rounds), which is fine for comparing
   settings; the number of trees and the threshold are worked out in
   training with the full model. `HYPERPARAMS_SOURCE` (`04_model_pipeline`'s
   `tuning` param) is `search` (default) or `previous` (keep the
   saved hyperparameters, skip the search — still refreshes the feature
   list). **Run a version's tuning before training it** (training stops if
   `selected_features.json` is missing), and again whenever its features
   should change — e.g. v2 after v1 is retrained, since v2 selects from v1's
   output; `previous` is enough to refresh the list if the hyperparameters
   don't need re-searching.
6. **Model training** (`src/Training/04_Model-Development/<model_folder>/02_training_<model_folder>.ipynb`) — trains a
   `lgb.LGBMClassifier` (GOSS boosting) on `model_dataset_final.parquet`
   using the tuned (or default) hyperparameters, wraps it as
   `LightGBMProductionWrapper`, registers it to the MLflow model registry
   under `HomeCredit_Scorecard_Model`, and saves `test_predictions.parquet`
   (`SK_ID_CURR`, `TARGET`, `predicted_proba` for the held-out test set)
   that both of the next two stages read.

   **Hyperparameters** (`HYPERPARAMS_SOURCE`, Section 1): `'tuned'` (default)
   loads the version's `*_resolved_best.json` from step 5; `'defaults'` uses
   the library's built-in defaults, a baseline showing what tuning adds
   (features still come from the tuning notebook's `selected_features.json`,
   so run tuning — `previous` is fine — at least once). `'defaults'` is for
   manual runs: set the `HOMECREDIT_HYPERPARAMS` env var or edit
   `HYPERPARAMS_SOURCE` in the notebook's first code cell. Under Airflow
   (`04_model_pipeline`) training always uses `'tuned'`. The MLflow run is
   tagged `hyperparameters_source`.

   **Training** — data is split 80 / 20 into train / test, then the steps
   below. The split is **fixed**: every tuning and training notebook, for
   every version, uses `RANDOM_STATE = 1994`, `TEST_SIZE = 0.20`, stratified
   on `TARGET`, so all versions are graded on the same 61,503 test
   applicants and are directly comparable. Nothing may be chosen using that
   test set — including the next version's features (hence out-of-fold
   SHAP, see 6.c below); a different split per version wouldn't avoid this,
   since a fresh random 20% still overlaps ~20% with the previous test set.
   1. **5-fold cross-validation on the 80%** (`CV_FOLDS`, Section 3), with
      early stopping on AUC in each fold (`N_ESTIMATORS_MAX` = 5000 is only
      a ceiling). The folds' average best iteration is the **number of
      trees**; their out-of-fold predictions (each applicant scored by a
      fold model that didn't train on them) give the **classification
      threshold** (F1-maximising, Section 4).
   2. **One final model** trained on the full 80% with that number of trees.
   3. **Evaluation on the 20% test set**, which is never used for any choice,
      so every reported metric (`test_*` in MLflow: AUC with a bootstrap 95%
      CI, Gini, KS, calibration, precision/recall/F1) is unbiased.

   Also logged: `cv_auc_mean` / `cv_auc_std` / `oof_auc` / `oof_f1_score` /
   `threshold` from the cross-validation, `best_iteration` (the tree count),
   and `train_*` metrics (on the full 80%) to show overfitting. History: a
   separate validation set (70 / 10 / 20) was used until 2026-10-18;
   cross-validation has replaced it so the final model trains on more data.
   (From 2026-10-19 to 2026-10-20 the cross-validation briefly ran in the
   tuning notebooks instead; it was moved back to keep tuning about
   hyperparameters only.)
   Section 6 of every training notebook (LightGBM v1–v4, XGB_v1),
   **Feature Importance & Interpretability**, covers 6.a gain-based
   importance, 6.b SHAP and 6.c the combined table. **SHAP** (6.b): exact
   TreeSHAP values for 2,000
   test-set applicants, checked to reproduce the model's raw output, saved
   to `ModelOutput/<model_folder>/` and logged to the MLflow run —
   `plots/SHAP_Summary.png` (beeswarm), `plots/SHAP_Waterfall_Example.png`
   (one high-risk applicant), `metadata/shap_feature_importance.xlsx`
   (mean |SHAP| per feature) and `metadata/shap_values_sample.parquet`
   (each sampled applicant's SHAP values, base value and raw margin, indexed
   by `SK_ID_CURR`). The same TreeSHAP decomposition produces the
   per-attribute points and reason codes at inference time
   (`src/Inference/explainability.py`).
   **Feature importance** (6.a, `lgbm_feature_importance.xlsx`) is **total gain** — LightGBM models are
   built with `importance_type='gain'`, XGBoost uses gain by default — not
   split count, which over-rewards features with many distinct values.
   Feature selection per version (the rule lives in the tuning notebook,
   which saves the list to `selected_features.json`; training reads it, and
   logs it to MLflow as `metadata/selected_features.json`):
   - **v1** — all features in `model_dataset_final.parquet`;
   - **v2** — v1's top features by out-of-fold SHAP: `cumulative_shap_share`
     ≤ `SHAP_CUTOFF` (90%) in v1's `feature_importance_combined.xlsx`
     (`summary` sheet) — 57 of 186 in the 2026-10-22 v1 run;
   - **v3** (and XGB_v1) — the fixed list in
     `src/Training/04_Model-Development/LightGBM_v3/selected_features.xlsx`;
   - **v4** — v3's top features by cumulative gain share
     (`IMPORTANCE_CUTOFF` = 90%) in v3's `lgbm_feature_importance.xlsx`.

   The **combined table** (6.c,
   `feature_importance_combined.xlsx`, logged under `metadata/`) has two
   sheets. `summary`, sorted by out-of-fold SHAP (each CV fold's model
   explaining applicants it didn't train on), reads left to right: `rank`,
   `feature`, `shap_share`, `cumulative_shap_share`, `gain_rank`,
   `gain_share`, `cumulative_gain_share` (counted in gain order — the
   `cumulative_importance` the next version selects on), `split_rank`,
   `rank_gap` (gain rank − SHAP rank) and `test_shap_rank` (reporting only);
   shares are formatted as percentages and the header/feature columns are
   frozen. `raw_values` holds the unscaled measures. Pick features for the
   next version from the SHAP or gain share columns, never from test-set
   SHAP — that would bias the test metrics.
   **Promotion to the `prod`
   alias is a manual step** — done in the MLflow UI after reviewing the run
   (MLflow uses aliases, not the deprecated stage system).
7. **Segmentation analysis** (`src/Training/05_Segmentation-Analysis/Segmentation_Analysis.ipynb`) —
   joins `test_predictions.parquet` back to `model_dataset_final.parquet`'s
   demographic columns by `SK_ID_CURR` and checks bad rate, mean PD, AUC,
   and KS *within* each segment (gender, education, family/housing/income
   type, occupation, organization, age band, and whether the applicant has
   any Home Credit bureau history at all) — catches a segment the overall
   AUC/KS hides, flags it, and writes charts + an Excel summary to
   `data/Model-Training/ModelOutput/<model_folder>/SegmentationAnalysis/`.
   Purely diagnostic: never changes what scoring computes.
8. **Scoring/scorecard** (`src/Training/06_Scoring/Credit_Scoring.ipynb`) —
   **final model only**: converts predicted probability of default to a PDO
   (Points-to-Double-the-Odds) credit score and A–E risk band, and produces
   calibration curves, KS statistic, and PSI (population stability index)
   diagnostics; writes to `data/Model-Training/ScoreOutput/<model_folder>/`.

Steps 5–7 run as one linear Airflow DAG per version (`04_model_pipeline` —
see §3); its `tuning` / `run_training` / `run_segmentation` params let you
skip or stop at any step. Step 8 is its own DAG (`05_scoring`), run once
for the model you've chosen.

## 5. Inference pipeline — architecture

Everything production-facing lives in `src/Inference/`:

| File | Role |
|---|---|
| `config.py` | Central config — data paths, MLflow URI/model name/alias, PDO constants, risk-band thresholds. Pinned to match `02_training_LightGBM_v3.ipynb` / `Credit_Scoring.ipynb`; keep in sync manually if those change. |
| `features/pipeline.py` | `assemble_features(raw_tables)` — the **one place** raw CSVs become the model's feature matrix. Calls each table's `features/<table>.py::build_features()`, then `merge_feature_frames()` (join on `SK_ID_CURR`, validate against `selected_features.xlsx`, downcast dtypes to match training). Used by batch, real-time, and the Airflow per-table tasks alike, so all three paths can't silently diverge. |
| `features/application.py`, `bureau.py`, `previous_application.py`, `pos_cash.py`, `installments.py` | Per-table feature engineering, mirroring the Preprocess notebooks' logic. `credit_card.py` exists as a TODO stub (not currently wired into `assemble_features`). |
| `model.py` | `load_model()` (cached MLflow pyfunc load via `models:/{name}@{alias}`), `predict_proba(feature_df)`. |
| `pdo_scorecard.py` | `proba_to_score()`, `score_to_risk_band()` — same PDO/band logic as `Credit_Scoring.ipynb`. |
| `score_model_input.py` | Shared `score_applicants(feature_df)` (also called by `realtime_api.py`) + a CLI (`main(application_date)`) that loads raw tables, assembles features, scores, and writes batch output. Runs identically standalone or from Airflow. |
| `realtime_api.py` | FastAPI app (`POST /score`) for single-applicant scoring — validates via `schemas.py`, calls `assemble_features` + `score_applicants`. |
| `schemas.py` | Pydantic `ApplicationRequest` / `ScoreResponse` models for the API. |
| `batch_feature_tasks.py` | Per-table Airflow task callables used by `dags/inference_batch_dag.py` — same feature logic as `score_model_input.py`, split into one task per table + `join_features` + `score_features`, writing intermediate parquets to `data/Inference/BatchFeatures/<date>/`. |
| `verify_feature_parity.py` | Compares `assemble_features()` output against the training ground truth (`model_dataset_final.parquet`) — structural (feature list) and value-level (row match rate) checks. |
| `Batch_Inference.ipynb` | Notebook version of batch scoring, for interactive/manual runs; reuses `score_model_input.load_raw_tables()`. |

Batch and real-time scoring both funnel through the same
`assemble_features()` → `score_applicants()` call chain, so they can't
silently compute different results for the same applicant.

## 6. Running it

### 6.1 Build and start the containers

```bash
# Build + start both containers (pipeline + mlflow), rebuilding the image
docker compose up --build

# Same, but detached (returns control of the terminal)
docker compose up --build -d

# Build the image without starting anything (e.g. to check the build succeeds)
docker build -t homecredit-pipeline:latest .

# Point at a different local data/src directory instead of ./data, ./src
DATA_PATH=/your/path SRC_PATH=/your/src docker compose up --build
```

This starts:
- `homecredit_pipeline` — Airflow standalone (webserver + scheduler +
  triggerer) at **http://localhost:8080** (login `admin` / `admin`).
- `homecredit_mlflow` — MLflow tracking server (UI + API) at
  **http://localhost:5000**, storing records in `data/mlflow.db` and files
  in `data/mlruns`. The pipeline container reaches it as
  `http://mlflow:5000` (set via `MLFLOW_TRACKING_URI` in
  `docker-compose.yml`). See §6.10.

Python packages in the `homecredit_pipeline` image come from
[`requirements.txt`](requirements.txt) (shared with local runs, §6.9) plus
the Docker-only extras in the `Dockerfile` (Airflow, papermill, kaggle).
After changing either file, rebuild: `docker compose up -d --build`.

If port 8080 is already in use on Windows, a common culprit is another local
service (e.g. MTAgentService) bound to it — stop that service or change the
host port mapping in `docker-compose.yml`.

Other useful compose commands:

```bash
# Tail logs from both containers (Ctrl+C to stop watching, containers keep running)
docker compose logs -f

# Tail logs from just one service
docker compose logs -f pipeline
docker compose logs -f mlflow

# Check container status
docker compose ps

# Stop the containers (keeps the airflow_db volume + your data/ and src/ bind mounts)
docker compose down

# Stop and also remove the named volumes (Airflow's SQLite DB, notebook_outputs, mlruns)
docker compose down -v

# Restart just one service without rebuilding
docker compose restart pipeline
```

### 6.2 Open the UIs

Both UIs are plain web pages served by the running containers — there's no
separate "open" command, just point a browser at the URL once
`docker compose up` is running:

- **Airflow UI**: http://localhost:8080 (login `admin` / `admin`)
- **MLflow UI**: http://localhost:5000

To open them straight from the terminal on Windows:

```powershell
# PowerShell
Start-Process "http://localhost:8080"
Start-Process "http://localhost:5000"
```

```cmd
:: cmd.exe
start http://localhost:8080
start http://localhost:5000
```

### 6.3 Drive Airflow from the CLI (alternative to the UI)

Everything you can do by clicking around the Airflow UI can also be run via
`airflow` CLI commands inside the running `homecredit_pipeline` container —
useful for scripting a run or checking status without a browser:

```bash
# Open a shell inside the running container
docker compose exec pipeline bash

# ...or run a single airflow command directly without an interactive shell
docker compose exec pipeline airflow dags list

# List all DAGs and whether they're paused
docker compose exec pipeline airflow dags list

# Un-pause a DAG (new DAGs can start paused depending on config)
docker compose exec pipeline airflow dags unpause 01_preprocessing

# Trigger a DAG run right now
docker compose exec pipeline airflow dags trigger 01_preprocessing

# Trigger with a config param (e.g. batch inference for a specific date)
docker compose exec pipeline airflow dags trigger home_credit_batch_inference \
  --conf '{"application_date": "2026-09-04"}'

# Trigger the create-mds DAG with hyperopt enabled
docker compose exec pipeline airflow dags trigger 03_mds_creation_and_multicollinearity_test \
  --conf '{"run_hyperopt": true}'

# List recent runs of a DAG and their state
docker compose exec pipeline airflow dags list-runs -d 01_preprocessing

# List task states for a specific run
docker compose exec pipeline airflow tasks states-for-dag-run \
  01_preprocessing <run_id>

# Tail a specific task's logs
docker compose exec pipeline airflow tasks logs 01_preprocessing \
  task_application <run_id>

# Re-run a specific task (e.g. after fixing a bug) by clearing its state
docker compose exec pipeline airflow tasks clear 01_preprocessing \
  -t task_application -s <run_id> -e <run_id>
```

(`docker compose exec` requires the containers to already be running via
`docker compose up`; use `docker compose run --rm pipeline airflow ...`
instead if you want a one-off command without the stack already up.)

### 6.4 Run the training pipeline

Via the Airflow UI (`localhost:8080`) or the equivalent `airflow dags
trigger` commands from §6.3:

1. **Trigger `01_preprocessing`** — runs all Preprocess notebooks. Outputs
   land in `data/Model-Training/Staging/`. By default, once all tasks
   succeed it automatically triggers `02_variable_selection`; pass
   `--conf '{"trigger_variable_selection": false}'` (or uncheck the param
   on the trigger screen) to stop after preprocessing instead.
   ```bash
   docker compose exec pipeline airflow dags trigger 01_preprocessing
   ```
2. **`02_variable_selection`** — runs `select_vars.ipynb`, scoring every
   engineered feature and writing `numeric_selection_metrics_full_df.xlsx`.
   Triggered automatically by step 1 unless disabled; can also be run
   standalone:
   ```bash
   docker compose exec pipeline airflow dags trigger 02_variable_selection
   ```
   By default, once it finishes it automatically triggers
   `03_mds_creation_and_multicollinearity_test` in turn; pass
   `--conf '{"trigger_mds_creation": false}'` (or uncheck the param on the
   trigger screen) to stop after variable selection instead.
3. **Trigger `03_mds_creation_and_multicollinearity_test`** — runs
   `model_dataset_creation.ipynb`, then `01b_multicollinearity_reduction.ipynb`
   (VIF-based feature pruning against `model_dataset_initial.parquet`,
   protecting features with strong predictive scores from step 2's output;
   writes `model_dataset_final.parquet`). It then triggers
   `04_model_pipeline` for LightGBM_v1: with a full hyperparameter search
   if the `run_hyperopt` param is set (checkbox on the trigger screen, or
   `--conf '{"run_hyperopt": true}'`), otherwise with `tuning='previous'`.
4. **`04_model_pipeline`** — builds and evaluates one model version: tuning →
   training (registers the model to MLflow) → segmentation analysis.
   Choose the version with `model_folder`; the DAG finds the
   two notebooks in that version's folder. Typical triggers:
   ```bash
   # First time for a version (no saved hyperparameters yet): full search, then train
   docker compose exec pipeline airflow dags trigger 04_model_pipeline \
     --conf '{"model_folder": "LightGBM_v3", "tuning": "search"}'

   # Retrain with the saved hyperparameters (refreshes the feature list): minutes
   docker compose exec pipeline airflow dags trigger 04_model_pipeline \
     --conf '{"model_folder": "LightGBM_v2", "tuning": "previous"}'

   # Retrain without any tuning step (saved hyperparameters + feature list must exist)
   docker compose exec pipeline airflow dags trigger 04_model_pipeline \
     --conf '{"model_folder": "LightGBM_v2", "tuning": "skip"}'

   # Only tune, don't train
   docker compose exec pipeline airflow dags trigger 04_model_pipeline \
     --conf '{"model_folder": "XGB_v1", "tuning": "search", "run_training": false}'
   ```
   or set the same params on the trigger screen in the UI. Order across
   versions: v1 before v2 (v2 selects from v1's output) and v3 before v4.
   Pass `--conf '{"run_segmentation": false}'` to stop after training. Running the notebooks by hand
   instead: set `HPO_HYPERPARAMS_SOURCE` (tuning: `search` / `previous`) and
   `HOMECREDIT_HYPERPARAMS` (training: `tuned` / `defaults`) before starting
   Jupyter/VS Code, or edit `HYPERPARAMS_SOURCE` in each notebook's first
   code cell.
5. **Compare versions and pick the final model** — test-set metrics in
   MLflow, and each version's
   `data/Model-Training/ModelOutput/<model_folder>/SegmentationAnalysis/segment_summary.xlsx`
   (any flagged segments).
6. **`05_scoring`** — builds the scorecard for the chosen model only (no
   retraining; reads its saved `test_predictions.parquet`) and writes to
   `data/Model-Training/ScoreOutput/<model_folder>/` (score distribution,
   bands, KS):
   ```bash
   docker compose exec pipeline airflow dags trigger 05_scoring \
     --conf '{"model_folder": "LightGBM_v3"}'
   ```
   Review the scorecard before promoting. Segmentation and scoring can also
   be run by hand against an already-trained `model_folder` — set
   `HOMECREDIT_SCORING_MODEL_FOLDER` and open either notebook directly.
7. **Promote the model** — in the MLflow UI (`localhost:5000`), open
   `HomeCredit_Scorecard_Model`, pick the version you want to serve, and
   assign it the **`prod`** alias. Inference always loads
   `models:/HomeCredit_Scorecard_Model@prod`.

   To do it from Python instead (with the MLflow server running):
   ```python
   from mlflow.tracking import MlflowClient
   MlflowClient("http://localhost:5000").set_registered_model_alias(
       "HomeCredit_Scorecard_Model", "prod", <version>)
   ```

You can also run any of these notebooks directly (outside Airflow/Docker) as
long as `HOMECREDIT_DATA_ROOT` / `HOMECREDIT_SRC_DIR` point at the right
paths — each notebook self-configures from those env vars. Training
notebooks also need the MLflow server running for their MLflow section —
see §6.9 for the full local setup. From a local Python environment (e.g.
`project_env`):

```powershell
# PowerShell: activate the local venv, then open Jupyter to run notebooks interactively
project_env\Scripts\Activate.ps1
jupyter notebook src\Training\01_Preprocess\Application_data.ipynb

# Or execute a notebook headlessly with papermill, same as the DAGs do
papermill src\Training\01_Preprocess\Application_data.ipynb `
  data\Model-Training\notebook_outputs\application__manual.ipynb
```

### 6.5 Run batch inference

Prerequisites: a model version has the `prod` alias assigned, and
`src/Inference/selected_features.xlsx` (or whichever file `HOME_CREDIT_SELECTED_FEATURES` points to,
per `config.SELECTED_FEATURES_PATH`) exists.

**Data layout expected** (per application date, e.g. `2026-09-04`):
- `data/Inference/BatchInput/<date>/application.csv` — the new applicant
  batch (date-partitioned).
- `data/Input/{bureau,previous_application,POS_CASH_balance,installments_payments}.csv`
  — shared credit history, static (not date-partitioned).

**Option A — command line / cron:**
```powershell
# PowerShell, from a local environment (e.g. project_env activated)
project_env\Scripts\Activate.ps1
python src\Inference\score_model_input.py --application-date 2026-09-04
```
```bash
# Or from inside the running container
docker compose exec pipeline python /src/Inference/score_model_input.py --application-date 2026-09-04
```

**Option B — Airflow:** trigger `home_credit_batch_inference` in the UI, or
from the CLI:
```bash
docker compose exec pipeline airflow dags trigger home_credit_batch_inference \
  --conf '{"application_date": "2026-09-04"}'
```
Leave the `application_date` param/`--conf` blank to use the DAG run's
scheduled date, or set it explicitly to score an arbitrary past/ad-hoc batch.

**Option C — notebook:** run `src/Inference/Batch_Inference.ipynb`
interactively.

Output for all three: `data/Inference/BatchOutput/<date>/scored_applicants.parquet`
and `band_summary.csv` (applicant count / mean score / mean PD per risk band).

### 6.6 Run real-time scoring

```bash
cd src/Inference
pip install -r requirements.txt   # first time only — fastapi, uvicorn, pydantic, etc.
uvicorn realtime_api:app --host 0.0.0.0 --port 8000

# Or with auto-reload while developing:
uvicorn realtime_api:app --host 0.0.0.0 --port 8000 --reload
```

The model loads once at startup (`@app.on_event("startup")`) and is cached
in-process — no MLflow round trip per request.

```bash
curl -X POST http://localhost:8000/score \
  -H "Content-Type: application/json" \
  -d '{ "SK_ID_CURR": 100001, ... }'   # see schemas.ApplicationRequest for the full field list

# Health check
curl http://localhost:8000/health
```

Returns `{ SK_ID_CURR, PROBA_DEFAULT, CREDIT_SCORE, RISK_BAND }`. A 422 means
required features were missing/unassemblable for that request; check the
server log for which ones.

### 6.7 Verify feature parity (training vs. inference)

Before trusting inference output, confirm `assemble_features()` reproduces
the same feature values the model was trained on:

```bash
python src/Inference/verify_feature_parity.py
```

Compares against `data/Model-Training/ModelDataset/model_dataset_final.parquet`
structurally (does the feature list match) and at the value level (row-by-row
match rate on a sample of known applicants).

### 6.8 Run the inference test suite

```bash
cd src/Inference
pip install -r requirements.txt
pytest
```

`test_inference_smoke.py` checks the plumbing (join logic, feature-contract
enforcement, PDO math, A–E band edges) without needing a real MLflow model
or real data. `test_history_store.py` currently fails on import — it
exercises a `history_store` module that doesn't exist in this package yet
(see §7).

### 6.9 Run the notebooks on your own machine (no Docker)

Docker is only needed for Airflow. To run the notebooks interactively (e.g.
cell by cell in VS Code) on Windows, macOS or Linux:

**One-time setup on a new machine**

1. Clone the repo and put the Kaggle CSVs in `data/Input/`. To keep existing
   MLflow history and models, copy the whole `data/` folder instead
   (`data/` is git-ignored).
2. Create a Python 3.12 environment and install the packages — the same
   versions the Docker image uses:
   ```bash
   pip install -r requirements.txt
   ```
3. Tell the code where the project lives (the defaults point at the
   original author's `A:\Dominic\...` paths):
   ```powershell
   setx HOMECREDIT_DATA_ROOT "C:\path\to\Home Credit\data"
   setx HOMECREDIT_SRC_DIR   "C:\path\to\Home Credit\src"
   ```
   Open a new terminal / restart VS Code afterwards so they take effect.

**Every work session**

Training notebooks start and stop the server themselves: when run locally
(tracking URI on `localhost`), the first cell of Section 8 starts
`scripts/start_mlflow_server.py` in the background if it isn't already
running, and the cell at the end of Section 8 stops it again (only if that
notebook started it). Allow ~10 seconds after the start cell before the
MLflow cells. If the kernel is restarted in between, the stop cell no longer
knows the process — end it from Task Manager (the `python.exe` processes
running MLflow). Under Airflow/Docker both cells do nothing. To run
the server yourself instead (e.g. to browse the UI between notebooks):

1. Start the MLflow server in its own terminal, with the project environment
   active, and leave it open:
   ```bash
   python scripts/start_mlflow_server.py
   ```
   It prints the Python it is using — that should be your project
   environment, not a base/system Python (a base-conda MLflow without
   `email-validator` fails to start). Wait for `Uvicorn running on
   http://127.0.0.1:5000`; the UI is then at http://localhost:5000.
2. Run notebooks as usual. Training notebooks print
   `MLflow tracking URI : http://localhost:5000` in their MLflow section; if
   one prints `sqlite:///...`, its kernel still holds an old setting —
   restart the kernel.
3. Stop the server with Ctrl+C when done.

Only the MLflow cells at the end of a training notebook need the server;
if you forget to start it, those cells fail with a connection error — start
the server and re-run just those cells. Don't run this script and the Docker
`mlflow` service at the same time (both use port 5000).

### 6.10 How MLflow is set up

**One server, two kinds of storage.** The MLflow server (Docker
`homecredit_mlflow`, or `scripts/start_mlflow_server.py` locally) is the only
thing that reads or writes MLflow storage:

| Stored | Where |
|---|---|
| Run info, params, metrics, tags, model versions, aliases | `data/mlflow.db` (SQLite) |
| Run files (JSON, plots, Excel tables, reports) | `data/mlruns/<run_id>/artifacts/` |
| Model files | `data/mlruns/models/m-<id>/artifacts/` |

Clients connect by URL — `MLFLOW_TRACKING_URI`, default
`http://localhost:5000`; `http://mlflow:5000` inside the pipeline container
— and upload/download files over HTTP. Because no client ever opens those
folders itself, nothing machine-specific is written into MLflow's records:
the experiment's artifact location is `mlflow-artifacts:/` (meaning "through
the server"), and new model versions point at `models:/m-<id>`. The same
runs and models therefore work from Windows, from Docker and from any other
machine that can reach the server.

**What happens when a training notebook logs a run**

1. `set_tracking_uri` / `set_experiment` — connect to the server and look
   up `Home_Credit_Default_Risk`.
2. `start_run` — the server creates the run in `mlflow.db`; its file
   location becomes `mlflow-artifacts:/<run_id>/artifacts`.
3. `log_params` / `log_param` / `set_tag` / `log_metric` — sent to the
   server, stored as rows in `mlflow.db`.
4. `log_dict`, `log_figure`, `log_text`, `log_artifact` — the file is
   written to a temp location, uploaded (HTTP `PUT`), and saved by the
   server under `data/mlruns/<run_id>/artifacts/` — `plots/` (ROC/PR,
   calibration, KS, SHAP summary + waterfall) and `metadata/`
   (hyperparameters, classification report, binning/KS tables, feature and
   SHAP importance, the combined importance table, per-applicant SHAP values).
5. `pyfunc.log_model` — the server creates a logged model `m-<id>`; the
   notebook pickles `LightGBMProductionWrapper` into a temp folder with
   `MLmodel`, `requirements.txt`, `conda.yaml`, `python_env.yaml` and a
   3-row input example (categorical columns cast to plain text so they
   match the signature), validates the example against the signature,
   uploads everything to `data/mlruns/models/m-<id>/artifacts/`, and
   registers a new `HomeCredit_Scorecard_Model` version pointing at
   `models:/m-<id>`.
6. End of the `with mlflow.start_run()` block — the server marks the run
   `FINISHED` (or `FAILED` if any step raised).

Loading is the reverse: `load_model("models:/HomeCredit_Scorecard_Model@prod")`
asks the server which version holds `@prod`, downloads its files and
unpickles the model.

**Running in Docker** follows the same steps; the differences are that
Airflow runs the notebook via papermill inside `homecredit_pipeline`, it
reaches the server as `http://mlflow:5000` over Docker's network (allowed
by `--allowed-hosts` in `docker-compose.yml`), and the server writes to
`/data/...` — the same `data/` folder through the `./data:/data` mount.
Runs from VS Code and from Airflow land in the same experiment.

**Why it's set up this way.** Inside Docker the data folder is called
`/data`; on Windows it's `A:\...\data`. MLflow used to record files by
absolute folder path, which can only be right for one of the two. Runs
logged from Windows (June–August 2026) recorded Windows paths; on
2026-08-06 those records were rewritten to `/data/...` so Airflow could
load the model, which silently broke logging from Windows; the 2026-09-16
SQLite migration kept the Docker path. Routing everything through the
server removes paths from the records altogether.
`scripts/point_experiment_at_mlflow_server.py` was the one-time fix that
switched the existing experiment from `file:/data/mlruns` to
`mlflow-artifacts:/` (MLflow has no command to change an experiment's
artifact location). It has already been applied to this `mlflow.db` and
is not needed on new machines — a fresh server creates experiments
correctly by itself.

## 7. Known gaps / manual steps

- **Model promotion is manual**: nothing auto-assigns the `prod` alias after
  training — do it in the MLflow UI after reviewing the run's metrics.
- **`04_model_pipeline` defaults to `LightGBM_v1`**, not the production
  `LightGBM_v3` model — pick `model_folder` explicitly (see §6.4) when
  (re)training the production model. Its `tuning` param defaults to
  `previous`, which needs a previous search for that version; `check_inputs`
  stops the run with a clear message if it doesn't exist yet.
- **`credit_card.py` feature module is a TODO stub** — credit card features
  are not currently included in `assemble_features()`'s output, even though
  `CreditCard_data.ipynb` produces them for training. If the model's
  selected-feature list includes `CREDITCARD_*` columns, inference will fail
  the missing-features check in `merge_feature_frames()` until this is
  filled in.
- **`.env`-based config** (instead of hardcoded default paths in `config.py`
  / notebooks) has been discussed but not implemented — paths currently
  resolve via `HOMECREDIT_DATA_ROOT` / `HOMECREDIT_SRC_DIR` env vars with a
  hardcoded local Windows path as the fallback default.
- **MLflow model versions 1–11 only load inside Docker**: all MLflow access
  now goes through the tracking server (§6.10), so runs and model versions
  registered from 2026-10-15 on (v12+) load anywhere the server is
  reachable. Versions 1–11 (including the current `@prod`, v11) predate
  that and still record an absolute `file:///data/mlruns/...` source, so
  they load only where `/data` is mounted at that path; promoting a newly
  trained version removes the limitation. Running a training notebook's
  MLflow section, or inference, needs the server up —
  `python scripts/start_mlflow_server.py` locally (§6.9) or
  `docker compose up -d mlflow`.
