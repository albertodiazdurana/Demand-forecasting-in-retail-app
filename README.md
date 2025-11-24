# Demand Forecasting in Retail - Web App

Interactive demand forecasting application for Corporación Favorita grocery stores in Guayas, Ecuador.

## Live Demo

🚀 **[Launch App](https://[deployment-url].streamlit.app)** *(link pending deployment)*

## Overview

This Streamlit application provides sales forecasts for grocery products using a machine learning model trained on 4.8M historical transactions.

**Related Repository:** [Demand-forecasting-in-retail](https://github.com/albertodiazdurana/Demand-forecasting-in-retail) - Full analysis and model development.

## Features

- Select store and product family
- Single day or N-day forecasts (up to 30 days)
- Historical sales + forecast visualization
- Download forecast as CSV

## Model Performance

| Metric | Value |
|--------|-------|
| Model | TBD (XGBoost or LSTM) |
| RMSE | TBD |
| Training Data | 3.8M rows (Oct 2013 - Feb 2014) |
| Test Data | 818K rows (March 2014) |

## Project Structure
```
Demand-forecasting-in-retail-app/
├── app/
│   ├── main.py          # Streamlit UI
│   └── config.py        # Configuration
├── model/
│   └── model_utils.py   # Model loading
├── data/
│   └── data_utils.py    # Data processing
├── artifacts/           # Model files
├── requirements.txt
└── README.md
```

## Local Development
```bash
# Clone repository
git clone https://github.com/albertodiazdurana/Demand-forecasting-in-retail-app.git
cd Demand-forecasting-in-retail-app

# Install dependencies
pip install -r requirements.txt

# Run app
streamlit run app/main.py
```

## Configuration

Edit `app/config.py` to update:
- Model paths
- Store list
- Forecast date range

## Screenshots

*(Add after deployment)*

## License

MIT License

## Acknowledgments

- Data: [Kaggle Corporación Favorita Competition](https://www.kaggle.com/c/favorita-grocery-sales-forecasting)
- Course: Time Series Forecasting
