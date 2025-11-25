"""Streamlit app for demand forecasting."""
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from app.config import *
from model.model_utils import load_model, load_scaler, load_config

st.set_page_config(
    page_title="Demand Forecast - Corporación Favorita",
    page_icon="🛒",
    layout="wide"
)

# Load model and artifacts
@st.cache_resource
def load_artifacts():
    """Load model, scaler, and config (cached)."""
    try:
        model = load_model(MODEL_PATH)
        scaler = load_scaler(SCALER_PATH)
        config = load_config(CONFIG_PATH)
        return model, scaler, config
    except Exception as e:
        st.error(f"Error loading artifacts: {e}")
        return None, None, None

# Main app
st.title("🛒 Demand Forecasting")
st.subheader("Corporación Favorita - Guayas Region")

# Load artifacts
model, scaler, config = load_artifacts()

if model is None:
    st.error("Failed to load model artifacts. Please check file paths.")
    st.stop()

# Sidebar - Model Info
st.sidebar.header("📊 Model Information")
st.sidebar.write(f"**Model Type:** {config['model_type'].upper()}")
st.sidebar.write(f"**RMSE:** {config['metrics']['rmse']:.4f}")
st.sidebar.write(f"**MAE:** {config['metrics']['mae']:.4f}")
st.sidebar.write(f"**Training Samples:** {config['training_samples']:,}")

st.sidebar.markdown("---")
st.sidebar.header("ℹ️ About")
st.sidebar.write("This app provides sales forecasts for Guayas stores using a production XGBoost model trained on 3.8M transactions.")
st.sidebar.write(f"**Features:** {config['n_features']}")
st.sidebar.write(f"**Training Period:** {config['training_period']['start']} to {config['training_period']['end']}")

# Main content
st.success("✅ Model loaded successfully!")

st.markdown("---")
st.info("🚧 **Interactive forecasting features coming soon:**")
st.write("- Store and product selection")
st.write("- Single day / N-day forecast mode")
st.write("- Historical + forecast visualization")
st.write("- CSV download")

st.markdown("---")

# Show model details
with st.expander("🔍 Model Details"):
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Performance Metrics:**")
        st.write(f"- RMSE: {config['metrics']['rmse']:.4f}")
        st.write(f"- MAE: {config['metrics']['mae']:.4f}")
        st.write(f"- Bias: {config['metrics']['bias']:.4f}")
        st.write(f"- MAPE (non-zero): {config['metrics']['mape_nonzero']:.2f}%")
    
    with col2:
        st.write("**Training Configuration:**")
        st.write(f"- Samples: {config['training_samples']:,}")
        st.write(f"- Features: {config['n_features']}")
        st.write(f"- Gap: {config['training_period']['gap_days']} days")
        st.write(f"- Max Depth: {config['hyperparameters']['max_depth']}")

with st.expander("📋 Feature List (33 features)"):
    st.write(", ".join(FEATURE_COLUMNS))

st.markdown("---")
st.caption("Demand Forecasting in Retail | [GitHub](https://github.com/albertodiazdurana)")
