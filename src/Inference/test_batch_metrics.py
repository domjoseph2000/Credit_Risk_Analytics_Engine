"""
Unit tests for batch_metrics.py -- pure functions over already-scored
batch output, no real data/model needed. Run with: pytest
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import batch_metrics  # noqa: E402


def make_scored_df(n, band_counts):
    """band_counts: dict like {'E': 3, 'D': 2, ...} -- score/PD are just
    placeholder values, only Risk_Band composition and the given score/pd
    columns matter for these tests."""
    rows = []
    for letter, count in band_counts.items():
        for _ in range(count):
            rows.append({"SK_ID_CURR": len(rows), "PROBA_DEFAULT": np.random.uniform(0, 1), "score": 500, "Risk_Band": f"{letter} - Some Risk"})
    return pd.DataFrame(rows)


def test_calculate_psi_zero_for_identical_distributions():
    rng = np.random.default_rng(0)
    values = rng.uniform(0, 1, size=2000)
    psi = batch_metrics.calculate_psi(values, values, n_bins=10)
    assert psi == pytest.approx(0.0, abs=1e-9)


def test_calculate_psi_flags_significant_shift():
    rng = np.random.default_rng(0)
    expected = rng.uniform(0, 0.3, size=2000)  # mostly low-risk baseline
    actual = rng.uniform(0.7, 1.0, size=2000)  # mostly high-risk now -- a real shift
    psi = batch_metrics.calculate_psi(expected, actual, n_bins=10)
    assert psi > 0.25
    assert batch_metrics.psi_verdict(psi) == "significant_shift"


def test_band_letter_handles_hyphen_and_em_dash():
    assert batch_metrics._band_letter("E - Very High Risk") == "E"
    assert batch_metrics._band_letter("E — Very High Risk") == "E"


def test_snapshot_metrics_band_pct_sums_to_one():
    df = make_scored_df(10, {"E": 4, "D": 3, "C": 2, "B": 1})
    metrics = batch_metrics.snapshot_metrics(df)
    assert metrics["applicant_count"] == 10
    assert metrics["band_pct"]["E"] == pytest.approx(0.4)
    assert metrics["band_pct"]["A"] == 0.0
    assert sum(metrics["band_pct"].values()) == pytest.approx(1.0)


def test_compare_runs_returns_none_without_previous():
    current = {"date": "2026-08-31", "scored_df": make_scored_df(5, {"E": 5})}
    assert batch_metrics.compare_runs(current, None) is None


def test_compare_runs_computes_deltas_and_psi():
    previous = {"date": "2026-07-31", "scored_df": make_scored_df(10, {"E": 5, "D": 5})}
    current = {"date": "2026-08-31", "scored_df": make_scored_df(10, {"E": 2, "D": 8})}
    cmp = batch_metrics.compare_runs(current, previous)
    assert cmp["previous_date"] == "2026-07-31"
    assert cmp["deltas"]["applicant_count"] == 0
    assert cmp["deltas"]["band_pct"]["E"] == pytest.approx(-0.3)
    assert cmp["deltas"]["band_pct"]["D"] == pytest.approx(0.3)
    assert "psi_proba_default" in cmp and "psi_verdict" in cmp


def test_list_batch_runs_only_includes_dirs_with_scored_parquet(tmp_path):
    (tmp_path / "2026-07-31").mkdir()
    make_scored_df(3, {"E": 3}).to_parquet(tmp_path / "2026-07-31" / "scored_applicants.parquet")
    (tmp_path / "2026-08-15").mkdir()  # no scored_applicants.parquet -- e.g. a failed/in-progress run
    (tmp_path / "2026-08-31").mkdir()
    make_scored_df(2, {"D": 2}).to_parquet(tmp_path / "2026-08-31" / "scored_applicants.parquet")

    assert batch_metrics.list_batch_runs(tmp_path) == ["2026-07-31", "2026-08-31"]
