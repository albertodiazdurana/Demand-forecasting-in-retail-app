# Demand Forecasting in Retail - Web App

Interactive demand forecasting application for Corporación Favorita grocery stores in Guayas, Ecuador.

## Live Demo

🚀 **[Launch App](https://demand-forecasting-in-retail-app.streamlit.app)** (LIVE)

## Screenshot

![Streamlit App](docs/demand-forecasting-in-retail-app.streamlit.app_.png)

## Overview

This Streamlit application provides sales forecasts for grocery products using a production XGBoost model trained on 3.8M historical transactions.

**Related Repository:** [Demand-forecasting-in-retail](https://github.com/albertodiazdurana/Demand-forecasting-in-retail) - Full analysis and model development.

## Features

- ✅ Store and product selection (10 stores, 20 items)
- ✅ Single day / Multi-day forecasts (up to 30 days)
- ✅ Historical sales + forecast visualization
- ✅ Autoregressive forecasting (updates lag features)
- ✅ Download forecast as CSV
- ✅ Help popover with usage instructions

## Model Performance

| Metric | Value |
|--------|-------|
| Model | XGBoost |
| RMSE | 6.4008 |
| MAE | 1.7480 |
| Training Data | 3.8M rows (Oct 2013 - Feb 2014) |
| Test Data | 818K rows (March 2014) |

## Project Structure
```
Demand-forecasting-in-retail-app/
├── app/
│   ├── main.py          # Streamlit UI
│   ├── config.py        # Configuration
│   └── __init__.py
├── model/
│   ├── model_utils.py   # Model loading
│   └── __init__.py
├── data/
│   ├── data_utils.py    # Data processing
│   ├── sample_forecast_data.pkl
│   ├── store_item_lookup.csv
│   └── __init__.py
├── artifacts/           # Model files (2.1 MB)
│   ├── xgboost_model_full.pkl
│   ├── scaler_full.pkl
│   ├── feature_columns.json
│   └── model_config_full.json
├── tests/               # Unit tests
│   ├── test_data_utils.py
│   └── test_model_utils.py
├── docs/
│   └── demand-forecasting-in-retail-app.streamlit.app_.png
├── requirements.txt
├── .gitignore
└── README.md
```

## Local Development

### Prerequisites
- Python 3.11
- Virtual environment

### Setup
```bash
# Clone repository
git clone https://github.com/albertodiazdurana/Demand-forecasting-in-retail-app.git
cd Demand-forecasting-in-retail-app

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run app
streamlit run app/main.py
```

App will open at http://localhost:8501

### Running Tests
```bash
# Run all tests
python -m pytest tests/ -v

# Run with coverage
python -m pytest tests/ -v --cov=data --cov=model
```

## How to Use

1. **Select Store** - Choose from 10 Guayas stores
2. **Select Item** - Pick a product (filtered by store)
3. **Set Forecast Date** - Choose cutoff point
4. **Choose Mode** - Single Day or Multi-Day (1-30 days)
5. **Generate Forecast** - Click button to run prediction
6. **Download CSV** - Export results for planning

## Model Details

**Training Configuration:**
- Period: Oct 1, 2013 - Feb 21, 2014
- Gap: 7 days (DEC-013)
- Features: 33 (DEC-014)
- Hyperparameters: max_depth=6, n_estimators=500

**Top 5 Features:**
1. unit_sales_7d_avg (6.43)
2. unit_sales_lag1_7d_corr (1.96)
3. unit_sales_lag1 (1.64)
4. item_avg_sales (0.30)
5. unit_sales_14d_avg (0.23)

## License

MIT License

## Acknowledgments

- Data: [Kaggle Corporación Favorita Competition](https://www.kaggle.com/c/favorita-grocery-sales-forecasting)
- Course: Time Series Forecasting
- Model: Developed in [main analysis repository](https://github.com/albertodiazdurana/Demand-forecasting-in-retail)

## Author

Alberto Diaz Durana  
[GitHub](https://github.com/albertodiazdurana) | [LinkedIn](https://www.linkedin.com/in/albertodiazdurana/)
