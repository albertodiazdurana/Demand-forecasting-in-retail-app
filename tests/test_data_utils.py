"""Tests for data_utils module."""

import pytest
import pandas as pd

from data.data_utils import (
    get_stores,
    get_items_for_store,
    get_history,
    generate_forecast_dates,
)


@pytest.fixture
def sample_lookup_df():
    """Create a sample lookup DataFrame for testing."""
    return pd.DataFrame(
        {
            "store_nbr": [1, 1, 2, 2, 3],
            "item_nbr": [100, 101, 100, 102, 100],
            "family": ["GROCERY", "DAIRY", "GROCERY", "BEVERAGES", "GROCERY"],
            "avg_sales": [10.5, 5.2, 8.3, 12.1, 6.7],
        }
    )


@pytest.fixture
def sample_sales_df():
    """Create a sample sales DataFrame for testing."""
    dates = pd.date_range("2024-01-01", periods=60, freq="D")
    return pd.DataFrame(
        {
            "date": list(dates) * 2,
            "store_nbr": [1] * 60 + [2] * 60,
            "item_nbr": [100] * 60 + [100] * 60,
            "unit_sales": list(range(1, 61)) + list(range(10, 70)),
        }
    )


class TestGetStores:
    """Tests for get_stores function."""

    def test_returns_sorted_unique_stores(self, sample_lookup_df):
        """Should return sorted list of unique store numbers."""
        stores = get_stores(sample_lookup_df)
        assert stores == [1, 2, 3]

    def test_returns_list(self, sample_lookup_df):
        """Should return a list, not a numpy array or Series."""
        stores = get_stores(sample_lookup_df)
        assert isinstance(stores, list)

    def test_empty_dataframe(self):
        """Should handle empty DataFrame."""
        empty_df = pd.DataFrame({"store_nbr": []})
        stores = get_stores(empty_df)
        assert stores == []


class TestGetItemsForStore:
    """Tests for get_items_for_store function."""

    def test_returns_items_for_specific_store(self, sample_lookup_df):
        """Should return only items for the specified store."""
        items = get_items_for_store(sample_lookup_df, 1)
        assert len(items) == 2
        assert set(items["item_nbr"].tolist()) == {100, 101}

    def test_returns_dataframe_copy(self, sample_lookup_df):
        """Should return a copy, not a view."""
        items = get_items_for_store(sample_lookup_df, 1)
        items["new_col"] = "test"
        assert "new_col" not in sample_lookup_df.columns

    def test_nonexistent_store(self, sample_lookup_df):
        """Should return empty DataFrame for nonexistent store."""
        items = get_items_for_store(sample_lookup_df, 999)
        assert len(items) == 0


class TestGetHistory:
    """Tests for get_history function."""

    def test_returns_history_for_store_item(self, sample_sales_df):
        """Should return history for specific store-item pair."""
        history = get_history(sample_sales_df, 1, 100)
        assert len(history) == 60
        assert all(history["store_nbr"] == 1)
        assert all(history["item_nbr"] == 100)

    def test_respects_end_date(self, sample_sales_df):
        """Should filter by end_date when provided."""
        history = get_history(sample_sales_df, 1, 100, end_date="2024-01-15")
        assert len(history) == 15
        assert history["date"].max() <= pd.Timestamp("2024-01-15")

    def test_respects_days_limit(self, sample_sales_df):
        """Should limit to specified number of days."""
        history = get_history(sample_sales_df, 1, 100, days=30)
        assert len(history) == 30

    def test_returns_sorted_by_date(self, sample_sales_df):
        """Should return data sorted by date."""
        history = get_history(sample_sales_df, 1, 100)
        dates = history["date"].tolist()
        assert dates == sorted(dates)

    def test_nonexistent_combination(self, sample_sales_df):
        """Should return empty DataFrame for nonexistent store-item."""
        history = get_history(sample_sales_df, 999, 999)
        assert len(history) == 0


class TestGenerateForecastDates:
    """Tests for generate_forecast_dates function."""

    def test_generates_correct_number_of_dates(self):
        """Should generate the requested number of dates."""
        dates = generate_forecast_dates("2024-01-01", 7)
        assert len(dates) == 7

    def test_dates_are_consecutive(self):
        """Should generate consecutive daily dates."""
        dates = generate_forecast_dates("2024-01-01", 3)
        assert dates[0] == pd.Timestamp("2024-01-01")
        assert dates[1] == pd.Timestamp("2024-01-02")
        assert dates[2] == pd.Timestamp("2024-01-03")

    def test_returns_list(self):
        """Should return a list of timestamps."""
        dates = generate_forecast_dates("2024-01-01", 5)
        assert isinstance(dates, list)

    def test_single_day(self):
        """Should handle single day forecast."""
        dates = generate_forecast_dates("2024-06-15", 1)
        assert len(dates) == 1
        assert dates[0] == pd.Timestamp("2024-06-15")
