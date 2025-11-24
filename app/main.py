"""Streamlit app for demand forecasting."""
import streamlit as st

st.set_page_config(
    page_title="Demand Forecast - Corporación Favorita",
    page_icon="🛒",
    layout="wide"
)

st.title("🛒 Demand Forecasting")
st.subheader("Corporación Favorita - Guayas Region")

st.info("App under development. Model artifacts pending from FULL_02.")

# Placeholder UI
st.sidebar.header("Configuration")
store = st.sidebar.selectbox("Store", [24, 26, 27, 28, 29, 30, 32, 34, 35, 36, 51])
st.sidebar.write(f"Selected store: {store}")

st.write("---")
st.write("**Next steps:**")
st.write("1. Load production model from artifacts")
st.write("2. Add date picker and forecast generation")
st.write("3. Display forecast plot and CSV download")
