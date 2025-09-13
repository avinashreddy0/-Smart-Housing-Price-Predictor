import pandas as pd
import numpy as np
import joblib
import streamlit as st

st.title("🔮 Smart Housing Price Predictor:")
st.header("🏠 House Price Prediction")

mode_selection = st.sidebar.selectbox(
    "Choose Model",
    ("LinearRegression", "RandomForest", "GradientBoosting")
)

# --- Function to create feature bar chart ---
def show_feature_chart(user_input):
    features = ["Income", "House Age", "Rooms", "Bedrooms",
                "Avg Occupancy", "Population", "Latitude", "Longitude"]
    chart_data = pd.DataFrame(user_input, columns=features)
    st.subheader("📊 Feature Values You Entered")
    st.bar_chart(chart_data.T)

# --- User Inputs ---
income = st.number_input("Median Income", min_value=0.0, max_value=100000.0)
house_age = st.number_input("House Age", min_value=0.0, max_value=25.0)
rooms = st.number_input("Average Rooms", min_value=0.0, max_value=1000.0)
bedrooms = st.number_input("Average Bedrooms", min_value=0.0, max_value=1000.0)
Avg_Occupanc = st.number_input("Average Occupancy", min_value=0.0, max_value=1000.0)
population = st.number_input("Population", min_value=0.0, max_value=100.0)
latitude = st.number_input("Latitude", min_value=0.0, max_value=1000.0)
longitude = st.number_input("Longitude", min_value=0.0, max_value=1000.0)

user_input_array = np.array([[income, house_age, rooms, bedrooms, Avg_Occupanc,
                              population, latitude, longitude]])

# --- Prediction ---
if st.button("Predict Price"):
    if mode_selection == "LinearRegression":
        model = joblib.load("linearregression.pkl")
    elif mode_selection == "RandomForest":
        model = joblib.load("randomforest.pkl")
    else:  # GradientBoosting
        model = joblib.load("gradientboosting.pkl")

    prediction = model.predict(user_input_array)

    # Convert to INR
    predicted_value = prediction[0] * 100000
    in_inr = predicted_value * 83

    st.success(f"💰 Predicted House Price: ₹{in_inr:,.2f}")

    # --- Show feature bar chart ---
    show_feature_chart(user_input_array)
