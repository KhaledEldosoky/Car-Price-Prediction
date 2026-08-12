import streamlit as st
import pandas as pd
import numpy as np
import joblib

@st.cache_resource
def load_model():
    return joblib.load('car_price_model.pkl')

pipeline = load_model()

st.title("🚗 AI Car Price Prediction Engine")

# Unique categories extracted from the dataset (for UI population)
manufacturers = ['LEXUS', 'CHEVROLET', 'HONDA', 'FORD', 'HYUNDAI', 'TOYOTA', 'MERCEDES-BENZ', 'BMW', 'VOLKSWAGEN', 'AUDI'] # Truncated for example
models = ['RX 450', 'Equinox', 'FIT', 'Escape', 'Santa FE', 'Prius', 'Camry', 'E 350', 'X5'] # Truncated for example
colors = ['Silver', 'Black', 'White', 'Grey', 'Blue', 'Red']
categories = ['Jeep', 'Hatchback', 'Sedan', 'Microbus', 'Goods wagon', 'Universal', 'Coupe', 'Minivan', 'Cabriolet', 'Limousine', 'Pickup']

st.set_page_config(page_title="Car Price Predictor", layout="wide")

st.markdown("Enter the specifications of the vehicle below to get a real-time price estimation.")

# Layout: Split into Categorical and Numerical Sections
col1, col2 = st.columns(2)

with col1:
    st.subheader("Categorical Features")
    st.caption("Use the searchable multi-selects for high cardinality fields.")
    
    # High Cardinality -> Searchable Multiselect (restricted to 1 to mimic exact selection)
    manufacturer = st.multiselect("Manufacturer", manufacturers, max_selections=1, default=["TOYOTA"])
    car_model = st.multiselect("Model", models, max_selections=1, default=["Camry"])
    color = st.multiselect("Color", colors, max_selections=1, default=["Black"])
    
    # Low Cardinality
    category = st.selectbox("Category", categories)
    fuel_type = st.selectbox("Fuel Type", ['Petrol', 'Diesel', 'Hybrid', 'LPG', 'CNG', 'Plug-in Hybrid', 'Hydrogen'])
    gear_box = st.selectbox("Gear Box Type", ['Automatic', 'Tiptronic', 'Manual', 'Variator'])
    drive_wheels = st.selectbox("Drive Wheels", ['4x4', 'Front', 'Rear'])
    doors = st.selectbox("Doors", ['4-5', '2-3', '>5'])
    wheel = st.selectbox("Wheel", ['Left wheel', 'Right-hand drive'])
    leather = st.checkbox("Leather Interior")

with col2:
    st.subheader("Numerical Features")
    st.caption("Enter the exact digits for the vehicle's specs.")
    
    prod_year = st.number_input("Production Year", min_value=1900, max_value=2026, value=2015, step=1)
    mileage = st.number_input("Mileage (km)", min_value=0, value=100000, step=1000)
    engine_vol = st.number_input("Engine Volume (L)", min_value=0.0, value=2.0, step=0.1)
    turbo = st.checkbox("Has Turbo?")
    cylinders = st.number_input("Cylinders", min_value=1, max_value=16, value=4, step=1)
    airbags = st.number_input("Airbags", min_value=0, max_value=20, value=4, step=1)
    levy = st.number_input("Levy (Tax/Import Fee)", min_value=0.0, value=1000.0, step=50.0)

st.markdown("---")

# Predict Button
if st.button("🔮 Predict Price", type="primary", use_container_width=True):
    # Construct input payload matching training data schema
    input_data = pd.DataFrame({
        'Manufacturer': [manufacturer[0] if manufacturer else 'TOYOTA'],
        'Model': [car_model[0] if car_model else 'Camry'],
        'Color': [color[0] if color else 'Black'],
        'Category': [category],
        'Fuel type': [fuel_type],
        'Gear box type': [gear_box],
        'Drive wheels': [drive_wheels],
        'Doors': [doors],
        'Wheel': [wheel],
        'Leather interior': ['Yes' if leather else 'No'],
        'Prod. year': [prod_year],
        'Mileage': [mileage],
        'Engine volume': [engine_vol],
        'Turbo': [1 if turbo else 0],
        'Cylinders': [cylinders],
        'Airbags': [airbags],
        'Levy': [levy]
    })
    
    with st.spinner("Processing through ML Pipeline..."):
        try:
            # Inference through the exact same Pipeline
            prediction = pipeline.predict(input_data)[0]
            
            st.success("Prediction Successful!")
            st.markdown(f"### 💰 Estimated Market Price: **${prediction:,.2f}**")
            
        except Exception as e:
            st.error(f"An error occurred during prediction: {str(e)}")
