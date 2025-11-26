# Demand Forecasting in Retail - Web App

Interactive demand forecasting application for Corporación Favorita grocery stores in Guayas, Ecuador.

## Live Demo

🚀 **[Launch App](https://demand-forecasting-in-retail-app.streamlit.app)** *(LIVE)*

## Screenshot

![Streamlit App](docs/demand-forecasting-in-retail-app.streamlit.app_.png)

## Overview

This Streamlit application provides sales forecasts for grocery products using a production XGBoost model trained on 4.8M historical transactions.

**Related Repository:** [Demand-forecasting-in-retail](https://github.com/albertodiazdurana/Demand-forecasting-in-retail) - Full analysis and model development.

## Features

- ✅ Model loaded and ready
- 🚧 Store and product selection (coming soon)
- 🚧 Single day / N-day forecasts (coming soon)
- 🚧 Historical sales + forecast visualization (coming soon)
- 🚧 Download forecast as CSV (coming soon)

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
│   └── __init__.py
├── artifacts/           # Model files (2.1 MB)
│   ├── xgboost_model_full.pkl
│   ├── scaler_full.pkl
│   ├── feature_columns.json
│   └── model_config_full.json
├── docs/
│   └── streamlit_app_screenshot.png
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

## Configuration

Edit `app/config.py` to customize:
- Model paths
- Store list (10 Guayas stores)
- Forecast date range (Jan-Mar 2014)
- Feature columns (33 features)

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

## Deployment (Coming in Week 4 Day 3)

App will be deployed to Streamlit Community Cloud.

## License

MIT License

## Acknowledgments

- Data: [Kaggle Corporación Favorita Competition](https://www.kaggle.com/c/favorita-grocery-sales-forecasting)
- Course: Time Series Forecasting
- Model: Developed in [main analysis repository](https://github.com/albertodiazdurana/Demand-forecasting-in-retail)

## Author

Alberto Diaz Durana  
[GitHub](https://github.com/albertodiazdurana) | [LinkedIn](https://www.linkedin.com/in/albertodiazdurana/)
