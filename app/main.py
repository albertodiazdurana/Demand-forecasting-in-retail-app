"""Streamlit app for demand forecasting."""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from app.config import (
    MODEL_PATH, SCALER_PATH, CONFIG_PATH, FEATURE_COLUMNS,
    SAMPLE_DATA_PATH, LOOKUP_PATH, MAX_FORECAST_DAYS, HISTORY_DAYS
)
from model.model_utils import load_model, load_scaler, load_config
from data.data_utils import (
    load_sample_data, load_lookup_table, get_stores,
    get_items_for_store, get_history, prepare_features_for_prediction
)

st.set_page_config(
    page_title="Demand Forecast - Corporación Favorita",
    page_icon="🛒",
    layout="wide"
)

# Load artifacts (cached)
@st.cache_resource
def load_artifacts():
    """Load model, scaler, and config (cached)."""
    try:
        model = load_model(MODEL_PATH)
        scaler = load_scaler(SCALER_PATH)
        config = load_config(CONFIG_PATH)
        return model, scaler, config
    except Exception as e:
        st.error(f"Error loading model artifacts: {e}")
        return None, None, None

@st.cache_data
def load_data():
    """Load sample data and lookup table (cached)."""
    try:
        df = load_sample_data(SAMPLE_DATA_PATH)
        lookup = load_lookup_table(LOOKUP_PATH)
        return df, lookup
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None

def autoregressive_forecast(model, history, feature_columns, n_days):
    """
    Generate multi-day forecast with autoregressive updates.
    
    Each prediction updates lag features for the next prediction:
    - lag1 becomes the new prediction
    - lag7 shifts by 1 day
    - Rolling averages update with new value
    """
    predictions = []
    
    # Get initial feature values from latest history row
    latest = history.tail(1).copy()
    X = latest[feature_columns].values.flatten().astype(np.float32)
    
    # Create feature index map for easy updates
    feat_idx = {col: i for i, col in enumerate(feature_columns)}
    
    # Track recent predictions for rolling calculations
    recent_sales = history['unit_sales'].tail(30).tolist()
    
    for day in range(n_days):
        # Predict
        pred = model.predict(X.reshape(1, -1))[0]
        pred = max(0, float(pred))  # No negative sales
        predictions.append(pred)
        
        # Update features for next prediction (autoregressive)
        if day < n_days - 1:
            # Add prediction to recent sales
            recent_sales.append(pred)
            if len(recent_sales) > 30:
                recent_sales.pop(0)
            
            # Update lag features
            if 'unit_sales_lag1' in feat_idx:
                X[feat_idx['unit_sales_lag1']] = pred
            
            # Update rolling averages
            if 'unit_sales_7d_avg' in feat_idx and len(recent_sales) >= 7:
                X[feat_idx['unit_sales_7d_avg']] = np.mean(recent_sales[-7:])
            if 'unit_sales_14d_avg' in feat_idx and len(recent_sales) >= 14:
                X[feat_idx['unit_sales_14d_avg']] = np.mean(recent_sales[-14:])
            if 'unit_sales_30d_avg' in feat_idx and len(recent_sales) >= 30:
                X[feat_idx['unit_sales_30d_avg']] = np.mean(recent_sales[-30:])
            
            # Update calendar features
            next_date = pd.Timestamp(history['date'].max()) + timedelta(days=day+1)
            if 'dayofweek' in feat_idx:
                X[feat_idx['dayofweek']] = next_date.dayofweek
            if 'day' in feat_idx:
                X[feat_idx['day']] = next_date.day
            if 'month' in feat_idx:
                X[feat_idx['month']] = next_date.month
            if 'weekend' in feat_idx:
                X[feat_idx['weekend']] = 1 if next_date.dayofweek >= 5 else 0
    
    return predictions

# Main app
st.title("🛒 Demand Forecasting")
st.subheader("Corporación Favorita - Guayas Region")

# Help button
with st.popover("ℹ️ How to Use"):
    st.markdown("""
    **Quick Start:**
    1. Select a **Store** and **Item** in the sidebar
    2. Choose a **Forecast Date** (cutoff point)
    3. Pick **Single Day** or **Multi-Day** mode
    4. Click **Generate Forecast**
    
    **Features:**
    - 📈 View historical sales + forecast chart
    - 📋 See detailed prediction table
    - 📥 Download forecast as CSV
    
    **Note:** Forecasts use an XGBoost model trained on 3.8M transactions from Guayas stores.
    """)

# Load everything
model, scaler, config = load_artifacts()
df, lookup = load_data()

if model is None or df is None:
    st.error("Failed to load required files. Please check configuration.")
    st.stop()

# Sidebar - Model Info
st.sidebar.header("📊 Model Information")
st.sidebar.write(f"**Model:** {config['model_type'].upper()}")
st.sidebar.write(f"**RMSE:** {config['metrics']['rmse']:.4f}")
st.sidebar.write(f"**MAE:** {config['metrics']['mae']:.4f}")
st.sidebar.write(f"**Training:** {config['training_samples']:,} samples")

st.sidebar.markdown("---")

# Sidebar - Store/Item Selection
st.sidebar.header("🏪 Selection")

# Store dropdown
stores = get_stores(lookup)
selected_store = st.sidebar.selectbox("Store", stores, format_func=lambda x: f"Store {x}")

# Item dropdown (filtered by store)
items_df = get_items_for_store(lookup, selected_store)
item_options = items_df['item_nbr'].tolist()
item_labels = {row['item_nbr']: f"{row['item_nbr']} ({row['family']})" 
               for _, row in items_df.iterrows()}

selected_item = st.sidebar.selectbox(
    "Item", 
    item_options,
    format_func=lambda x: item_labels.get(x, str(x))
)

# Show selected item info
item_info = items_df[items_df['item_nbr'] == selected_item].iloc[0]
st.sidebar.write(f"**Family:** {item_info['family']}")
st.sidebar.write(f"**Avg Sales:** {item_info['avg_sales']:.1f} units/day")

st.sidebar.markdown("---")

# Forecast Configuration
st.sidebar.header("📅 Forecast Settings")

# Date range from data
min_date = df['date'].min().date()
max_date = df['date'].max().date()

# Forecast date (must have enough history)
forecast_date = st.sidebar.date_input(
    "Forecast Start Date",
    value=max_date,
    min_value=min_date + timedelta(days=30),  # Need 30 days for lag features
    max_value=max_date
)

# Forecast mode
forecast_mode = st.sidebar.radio("Forecast Mode", ["Single Day", "Multi-Day"])

if forecast_mode == "Multi-Day":
    n_days = st.sidebar.slider("Days to Forecast", 1, MAX_FORECAST_DAYS, 7)
else:
    n_days = 1

# Generate Forecast button
generate_forecast = st.sidebar.button("🔮 Generate Forecast", type="primary")

# Main content
st.markdown("---")

# Get history for display
history = get_history(df, selected_store, selected_item, end_date=forecast_date, days=HISTORY_DAYS)

if len(history) == 0:
    st.warning("No historical data available for this store-item combination.")
    st.stop()

# Display current selection
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Store", selected_store)
with col2:
    st.metric("Item", selected_item)
with col3:
    st.metric("Family", item_info['family'])
with col4:
    st.metric("History Days", len(history))

st.markdown("---")

# Generate forecast when button clicked
if generate_forecast:
    with st.spinner("Generating forecast..."):
        # Generate predictions using autoregressive method
        predictions = autoregressive_forecast(model, history, FEATURE_COLUMNS, n_days)
        
        # Create dates for forecast
        dates = [pd.Timestamp(forecast_date) + timedelta(days=i+1) for i in range(n_days)]
        
        # Create forecast DataFrame
        forecast_df = pd.DataFrame({
            'date': dates,
            'predicted_sales': predictions
        })
        
        # Display results
        st.subheader("📈 Forecast Results")
        
        # Plot
        fig, ax = plt.subplots(figsize=(12, 5))
        
        # History
        ax.plot(history['date'], history['unit_sales'], 
               label='Historical Sales', color='#3498db', alpha=0.7)
        
        # Forecast
        ax.plot(forecast_df['date'], forecast_df['predicted_sales'],
               marker='o', label='Forecast', color='#e74c3c', linewidth=2)
        
        # Vertical line at forecast start
        ax.axvline(x=pd.Timestamp(forecast_date), color='gray', 
                  linestyle='--', alpha=0.5, label='Forecast Start')
        
        ax.set_xlabel('Date')
        ax.set_ylabel('Unit Sales')
        ax.set_title(f'Sales Forecast - Store {selected_store}, Item {selected_item}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Forecast table
        st.subheader("📋 Forecast Details")
        
        display_df = forecast_df.copy()
        display_df['date'] = display_df['date'].dt.strftime('%Y-%m-%d')
        display_df['predicted_sales'] = display_df['predicted_sales'].round(2)
        display_df.columns = ['Date', 'Predicted Sales']
        
        st.dataframe(display_df, use_container_width=True, hide_index=True)
        
        # Summary stats
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Forecast", f"{forecast_df['predicted_sales'].sum():.1f} units")
        with col2:
            st.metric("Daily Average", f"{forecast_df['predicted_sales'].mean():.1f} units")
        with col3:
            hist_avg = history['unit_sales'].mean()
            st.metric("Historical Average", f"{hist_avg:.1f} units")
        
        # CSV Download
        csv = forecast_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Forecast CSV",
            data=csv,
            file_name=f"forecast_store{selected_store}_item{selected_item}_{forecast_date}.csv",
            mime="text/csv"
        )

else:
    # Show historical data when no forecast generated
    st.subheader("�� Historical Sales")
    
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(history['date'], history['unit_sales'], color='#3498db')
    ax.set_xlabel('Date')
    ax.set_ylabel('Unit Sales')
    ax.set_title(f'Historical Sales - Store {selected_store}, Item {selected_item}')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)
    
    # Stats
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Average", f"{history['unit_sales'].mean():.1f}")
    with col2:
        st.metric("Max", f"{history['unit_sales'].max():.1f}")
    with col3:
        st.metric("Min", f"{history['unit_sales'].min():.1f}")
    with col4:
        st.metric("Std Dev", f"{history['unit_sales'].std():.1f}")
    
    st.info("👈 Configure settings in the sidebar and click **Generate Forecast** to create predictions.")

# Footer
st.markdown("---")
st.caption("Demand Forecasting in Retail | [GitHub](https://github.com/albertodiazdurana) | Model: XGBoost (RMSE 6.4008)")
