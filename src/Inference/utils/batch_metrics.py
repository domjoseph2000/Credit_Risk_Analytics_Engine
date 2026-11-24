"""
Numbers for the batch monitoring dashboard (batch_dashboard_api.py),
calculated from batches that have already been scored
(DATA_ROOT/Inference/BatchOutput/<date>/).

Only reads those output files -- no model or MLflow needed.
"""
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from utils import config

SCORED_FILENAME = "scored_applicants.parquet"
BAND_SUMMARY_FILENAME = "band_summary.csv"
BAND_LETTERS = ["E", "D", "C", "B", "A"]  # worst to best, same order as config.BAND_LABELS


def list_batch_runs(output_root: Optional[Path] = None) -> list[str]:
    """
    Dates of the batches that have finished scoring (BatchOutput/<date>/
    folders containing scored_applicants.parquet), oldest first.
    """
    root = Path(output_root or config.SCORES_OUTPUT_DIR)
    if not root.exists():
        return []
    dates = [p.name for p in root.iterdir() if p.is_dir() and (p / SCORED_FILENAME).exists()]
    return sorted(dates)


def load_run(date: str, output_root: Optional[Path] = None) -> dict:
    """Read one batch's scored applicants: {"date": ..., "scored_df": DataFrame}."""
    run_dir = Path(output_root or config.SCORES_OUTPUT_DIR) / date
    scored_df = pd.read_parquet(run_dir / SCORED_FILENAME)
    return {"date": date, "scored_df": scored_df}


def _band_letter(label: str) -> str:
    """
    "E — Very High Risk" -> "E". Only the first letter is used, because older
    batches wrote the labels with "-" instead of "—".
    """
    return (label or "").strip()[:1]


def snapshot_metrics(scored_df: pd.DataFrame) -> dict:
    """
    Headline numbers for one batch: applicant count, mean and median score,
    mean PD, and the share of applicants in each risk band (A–E).
    """
    total = len(scored_df)
    band_counts = scored_df["Risk_Band"].astype(str).map(_band_letter).value_counts()
    band_pct = {letter: (band_counts.get(letter, 0) / total if total else 0.0) for letter in BAND_LETTERS}
    return {
        "applicant_count": total,
        "mean_score": float(scored_df["score"].mean()) if total else None,
        "median_score": float(scored_df["score"].median()) if total else None,
        "mean_pd": float(scored_df["PROBA_DEFAULT"].mean()) if total else None,
        "band_pct": band_pct,
    }


def calculate_psi(expected, actual, n_bins: int = 10, min_bin_edge: float = 0.0, max_bin_edge: float = 1.0):
    """
    Population Stability Index (PSI): how much a distribution has shifted
    between `expected` (the earlier batch) and `actual` (this batch).

    How: split the expected values into 10 equal-sized groups, then compare
    what share of each batch falls in each group:
        PSI = sum over groups of (actual% - expected%) × ln(actual% / expected%)
    0 means identical; see psi_verdict() for the thresholds. Empty groups get
    a tiny 0.01% so the log never divides by zero.

    Same formula and thresholds as the training notebooks' train-vs-test PSI
    check (section 7.d of 02_training_LightGBM_v4.ipynb), applied here to the
    predicted PDs of two batches.
    """
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)

    _, bin_edges = pd.qcut(expected, n_bins, retbins=True, duplicates="drop")
    bin_edges[0] = min_bin_edge
    bin_edges[-1] = max_bin_edge

    expected_counts = np.histogram(expected, bins=bin_edges)[0]
    actual_counts = np.histogram(actual, bins=bin_edges)[0]

    expected_pct = expected_counts / len(expected)
    actual_pct = actual_counts / len(actual)
    expected_pct = np.where(expected_pct == 0, 1e-4, expected_pct)
    actual_pct = np.where(actual_pct == 0, 1e-4, actual_pct)

    bin_psi = (actual_pct - expected_pct) * np.log(actual_pct / expected_pct)
    return float(np.sum(bin_psi))


def psi_verdict(psi: float) -> str:
    """The usual rule of thumb: below 0.10 stable, 0.10–0.25 a moderate shift, 0.25+ a significant shift."""
    if psi < 0.10:
        return "stable"
    if psi < 0.25:
        return "moderate_shift"
    return "significant_shift"


def compare_runs(current: dict, previous: Optional[dict]) -> Optional[dict]:
    """
    How this batch differs from the previous one: change in applicant count,
    mean score, mean PD and band shares, plus the PSI of the predicted PDs.

    current/previous: load_run() results. Returns None if there's no previous batch.
    """
    if previous is None:
        return None
    cur = snapshot_metrics(current["scored_df"])
    prev = snapshot_metrics(previous["scored_df"])
    psi = calculate_psi(previous["scored_df"]["PROBA_DEFAULT"], current["scored_df"]["PROBA_DEFAULT"])
    return {
        "previous_date": previous["date"],
        "deltas": {
            "applicant_count": cur["applicant_count"] - prev["applicant_count"],
            "mean_score": (cur["mean_score"] - prev["mean_score"]) if cur["mean_score"] is not None and prev["mean_score"] is not None else None,
            "mean_pd": (cur["mean_pd"] - prev["mean_pd"]) if cur["mean_pd"] is not None and prev["mean_pd"] is not None else None,
            "band_pct": {k: cur["band_pct"][k] - prev["band_pct"][k] for k in BAND_LETTERS},
        },
        "psi_proba_default": psi,
        "psi_verdict": psi_verdict(psi),
    }
