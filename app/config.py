"""Configuration for Demand Forecasting App."""
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent.parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"
DATA_DIR = BASE_DIR / "data"

# Model files
MODEL_PATH = ARTIFACTS_DIR / "xgboost_model_full.pkl"
SCALER_PATH = ARTIFACTS_DIR / "scaler_full.pkl"
FEATURES_PATH = ARTIFACTS_DIR / "feature_columns.json"
CONFIG_PATH = ARTIFACTS_DIR / "model_config_full.json"

# Data files
SAMPLE_DATA_PATH = DATA_DIR / "sample_forecast_data.pkl"
LOOKUP_PATH = DATA_DIR / "store_item_lookup.csv"

# Forecast settings
FORECAST_START = "2014-01-01"
FORECAST_END = "2014-03-31"
MAX_FORECAST_DAYS = 30
HISTORY_DAYS = 180

# Feature columns (33 features per DEC-014)
FEATURE_COLUMNS = [
    # Temporal (8)
    'unit_sales_lag1', 'unit_sales_lag7', 'unit_sales_lag14', 'unit_sales_lag30',
    'unit_sales_7d_avg', 'unit_sales_14d_avg', 'unit_sales_30d_avg',
    'unit_sales_lag1_7d_corr',
    
    # Calendar (7)
    'year', 'month', 'day', 'dayofweek', 'dayofyear', 'weekofyear', 'quarter',
    
    # Holiday (4)
    'holiday_proximity', 'is_holiday', 'holiday_period', 'days_to_next_holiday',
    
    # Promotion (2)
    'onpromotion', 'promo_item_interaction',
    
    # Store/Item (7)
    'cluster', 'store_avg_sales', 'item_avg_sales', 'item_store_avg',
    'cluster_avg_sales', 'family_avg_sales', 'city_avg_sales',
    
    # Derived (5)
    'perishable', 'weekend', 'month_start', 'month_end', 'is_payday'
]
