"""Data loading and feature engineering utilities."""

import pandas as pd


def load_sample_data(data_path):
    """Load sample forecast data.

    Args:
        data_path: Path to sample_forecast_data.pkl

    Returns:
        DataFrame with historical data
    """
    return pd.read_pickle(data_path)


def load_lookup_table(lookup_path):
    """Load store-item lookup table.

    Args:
        lookup_path: Path to store_item_lookup.csv

    Returns:
        DataFrame with store-item pairs
    """
    return pd.read_csv(lookup_path)


def get_stores(lookup_df):
    """Get list of available stores."""
    return sorted(lookup_df["store_nbr"].unique().tolist())


def get_items_for_store(lookup_df, store_nbr):
    """Get items available for a specific store.

    Args:
        lookup_df: Lookup DataFrame
        store_nbr: Store number

    Returns:
        DataFrame with items for that store
    """
    return lookup_df[lookup_df["store_nbr"] == store_nbr].copy()


def get_history(df, store_nbr, item_nbr, end_date=None, days=180):
    """Get historical sales for a store-item pair.

    Args:
        df: Sample data DataFrame
        store_nbr: Store number
        item_nbr: Item number
        end_date: End date for history (default: latest)
        days: Number of days of history to return

    Returns:
        DataFrame with historical data
    """
    mask = (df["store_nbr"] == store_nbr) & (df["item_nbr"] == item_nbr)
    history = df[mask].copy()
    history = history.sort_values("date")

    if end_date is not None:
        history = history[history["date"] <= pd.Timestamp(end_date)]

    # Return last N days
    if len(history) > days:
        history = history.tail(days)

    return history


def generate_forecast_dates(start_date, n_days):
    """Generate list of forecast dates.

    Args:
        start_date: First forecast date
        n_days: Number of days to forecast

    Returns:
        List of dates
    """
    dates = pd.date_range(start=start_date, periods=n_days, freq="D")
    return dates.tolist()
