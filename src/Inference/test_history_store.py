"""
Tests for history_store: synthetic on-disk CSVs standing in for the real
bureau/previous/POS/credit-card/installments tables, checking that
per-applicant lookups filter correctly and the bureau -> bureau_balance
join (via SK_ID_BUREAU) works.
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config  # noqa: E402
import history_store  # noqa: E402


@pytest.fixture
def history_dir(tmp_path):
    bureau = pd.DataFrame(
        {
            "SK_ID_CURR": [1, 1, 2],
            "SK_ID_BUREAU": [900, 901, 902],
            "AMT_CREDIT_SUM": [10000, 5000, 20000],
        }
    )
    bureau_balance = pd.DataFrame(
        {
            "SK_ID_BUREAU": [900, 900, 902],
            "MONTHS_BALANCE": [-1, -2, -1],
            "STATUS": ["C", "C", "0"],
        }
    )
    previous_application = pd.DataFrame(
        {"SK_ID_CURR": [1, 3], "SK_ID_PREV": [500, 501], "AMT_CREDIT": [15000, 8000]}
    )

    bureau.to_csv(tmp_path / "bureau.csv", index=False)
    bureau_balance.to_csv(tmp_path / "bureau_balance.csv", index=False)
    previous_application.to_csv(tmp_path / "previous_application.csv", index=False)
    # pos_cash / credit_card / installments intentionally omitted to also
    # cover the "table file doesn't exist" path.

    config.HISTORY_DATA_DIR = tmp_path
    history_store._load_indexed_tables.cache_clear()
    return tmp_path


def test_applicant_with_bureau_and_previous_history(history_dir):
    result = history_store.get_applicant_history(1)

    assert len(result["bureau"]) == 2
    assert set(result["bureau"]["SK_ID_BUREAU"]) == {900, 901}
    assert len(result["previous_application"]) == 1
    # bureau_balance joined via this applicant's own SK_ID_BUREAU values
    assert len(result["bureau_balance"]) == 2
    assert set(result["bureau_balance"]["SK_ID_BUREAU"]) == {900}


def test_applicant_with_no_history_returns_empty_dict(history_dir):
    result = history_store.get_applicant_history(999)
    assert result == {}


def test_applicant_with_only_previous_application_history(history_dir):
    result = history_store.get_applicant_history(3)
    assert "bureau" not in result
    assert "bureau_balance" not in result
    assert len(result["previous_application"]) == 1


def test_missing_table_file_is_skipped_not_an_error(history_dir):
    # pos_cash/credit_card/installments files don't exist in this fixture
    result = history_store.get_applicant_history(1)
    assert "pos_cash" not in result
    assert "credit_card" not in result
    assert "installments" not in result
