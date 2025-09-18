import streamlit as st
import joblib
import pandas as pd

try:
    model = joblib.load('best_model.pkl')
    scaler = joblib.load('scaler.pkl')
    st.success("Model and scaler loaded successfully!")
except FileNotFoundError:
    st.error("Error loading model or scaler. Make sure 'best_model.pkl' and 'scaler.pkl' are in the correct directory.")
    st.stop()

# Load the training columns:
try:
    # Attempt to load saved training columns
    training_columns = joblib.load('training_columns.pkl')
    st.success("Training columns loaded successfully!")
except FileNotFoundError:
    st.warning("Could not find 'training_columns.pkl'. Using a placeholder list. Predictions may be inaccurate.")
    # In a real application, you MUST load the actual training columns.
    # This is a placeholder for demonstration purposes.
    training_columns = ['year', 'kilometerdriven', 'ownernumber', 'isc24assured', 'benefits', 'discountprice', 'created_year', 'created_month', 'created_dayofweek', 'car_age', 'month_num'] # Add your actual columns here


st.title('Car Price Prediction App')

st.write("""
This app predicts the price of a car based on its features.
Please enter the car's specifications below:
""")

year = st.number_input('Year', min_value=1990, max_value=2023, value=2022)
kilometerdriven = st.number_input('Kilometer Driven', min_value=0, value=50000)
ownernumber = st.selectbox('Owner Number', [1, 2, 3])
isc24assured = st.checkbox('Is C24 Assured?')
benefits = st.number_input('Benefits', min_value=0, value=10000)
discountprice = st.number_input('Discount Price', min_value=0, value=5000)


created_year = 2023 # Example
created_month = 1 # Example
created_dayofweek = 0 # Example (Monday)
car_age = created_year - year # Example
month_num = created_month # Example


# Create a DataFrame from the input
input_data = pd.DataFrame([[
    year, kilometerdriven, ownernumber, isc24assured, benefits, discountprice,
    created_year, created_month, created_dayofweek, car_age, month_num
    # Add values for all other features (including dummy variables) here
]], columns=['year', 'kilometerdriven', 'ownernumber', 'isc24assured', 'benefits', 'discountprice', 'created_year', 'created_month', 'created_dayofweek', 'car_age', 'month_num']) # Add your actual columns here

# Example of aligning columns (assuming you loaded 'training_columns.pkl')
try:
    input_data = input_data.reindex(columns=training_columns, fill_value=0)
except NameError:
    st.error("Training columns not loaded. Cannot align input data. Please ensure 'training_columns.pkl' exists.")
    st.stop()


# Scale the input data
input_data_scaled = scaler.transform(input_data)

# Make prediction
if st.button('Predict Price'):
    prediction = model.predict(input_data_scaled)
    st.success(f'Predicted Car Price: ₹{prediction[0]:,.2f}')
