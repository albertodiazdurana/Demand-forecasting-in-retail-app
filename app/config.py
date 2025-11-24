"""Configuration for Demand Forecasting App."""
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent.parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"

# Model files (update after FULL_02 completes)
MODEL_PATH = ARTIFACTS_DIR / "model.keras"  # or .pkl for XGBoost
SCALER_PATH = ARTIFACTS_DIR / "scaler.pkl"
FEATURES_PATH = ARTIFACTS_DIR / "feature_columns.json"

# Forecast settings
FORECAST_START = "2014-01-01"
FORECAST_END = "2014-03-31"
MAX_FORECAST_DAYS = 30

# Guayas stores
STORES = [24, 26, 27, 28, 29, 30, 32, 34, 35, 36, 51]
