import streamlit as st
import pickle
import pandas as pd
import numpy as np
from datetime import datetime

# Page Config
st.set_page_config(
    page_title="SolarQuest",
    page_icon="☀️",
    layout="wide"
)

# Custom CSS Styling
def load_css():
    with open("assets/style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

# Sidebar Navigation
st.sidebar.title("☀️ Solar AI Platform")
page = st.sidebar.radio(
    "Navigate",
    [
        "🏠 Dashboard",
        "🌞 Radiation Prediction",
        "📊 Uncertainty Quantification",
        "⚙️ Predictive Maintenance",
        "🤖 Solar AI Chatbot"
    ]
)

# Load Models
@st.cache_resource
def load_models():
    radiation_model = pickle.load(open("models/xgboost_solar_model.pkl", "rb"))
    quantile_model = pickle.load(open("models/xgb_q90.pkl", "rb"))
    maintenance_model = pickle.load(open("NLP/maintenance_classifier.pkl", "rb"))
    return radiation_model, quantile_model, maintenance_model

radiation_model, quantile_model, maintenance_model = load_models()

# ---------------- DASHBOARD ---------------- #

if page == "🏠 Dashboard":

    st.title("SolarQuest")
    st.markdown("AI-powered Solar Monitoring & Predictive Intelligence Platform")

    col1, col2, col3 = st.columns(3)

    col1.metric("Today's Radiation (kW/m²)", "5.4", "+2%")
    col2.metric("Predicted Energy (kWh)", "480", "+4.3%")
    col3.metric("System Health Score", "92%", "Stable")

    st.divider()

    st.subheader("System Overview")

    sample_data = pd.DataFrame({
        "Hour": range(24),
        "Radiation": np.random.uniform(0, 800, 24)
    })

    st.line_chart(sample_data.set_index("Hour"))

# ---------------- RADIATION PREDICTION ---------------- #

elif page == "🌞 Radiation Prediction":

    st.title("Solar Radiation Prediction")

    temp = st.number_input("Temperature (°C)")
    pressure = st.number_input("Pressure")
    humidity = st.number_input("Humidity")
    wind_speed = st.number_input("Wind Speed")

    if st.button("Predict Radiation"):

        input_data = np.array([[temp, pressure, humidity, wind_speed]])
        prediction = radiation_model.predict(input_data)

        st.success(f"Predicted Solar Radiation: {prediction[0]:.2f} W/m²")

# ---------------- UNCERTAINTY QUANTIFICATION ---------------- #

elif page == "📊 Uncertainty Quantification":

    st.title("Radiation Uncertainty Quantification (Q10, Q50, Q90)")

    temp = st.number_input("Temperature")
    pressure = st.number_input("Pressure")
    humidity = st.number_input("Humidity")
    wind_speed = st.number_input("Wind Speed")

    if st.button("Predict Quantiles"):

        input_data = np.array([[temp, pressure, humidity, wind_speed]])
        q_preds = quantile_model.predict(input_data)

        st.info(f"Q10: {q_preds[0][0]:.2f}")
        st.success(f"Q50 (Median): {q_preds[0][1]:.2f}")
        st.warning(f"Q90: {q_preds[0][2]:.2f}")

# ---------------- PREDICTIVE MAINTENANCE ---------------- #

elif page == "⚙️ Predictive Maintenance":

    st.title("Solar Panel Predictive Maintenance (MLP)")

    panel_temp = st.number_input("Panel Temperature")
    voltage = st.number_input("Voltage")
    current = st.number_input("Current")
    dust = st.number_input("Dust Level")
    humidity = st.number_input("Humidity")

    if st.button("Check System Health"):

        input_data = np.array([[panel_temp, voltage, current, dust, humidity]])
        result = maintenance_model.predict(input_data)

        if result[0] == 1:
            st.error("⚠️ Maintenance Required!")
        else:
            st.success("✅ System Operating Normally")

# ---------------- CHATBOT ---------------- #

elif page == "🤖 Solar AI Chatbot":

    st.title("Solar AI Assistant")

    from chatbot.chatbot_logic import get_chatbot_response

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    user_input = st.text_input("Ask anything about your solar system:")

    if st.button("Send"):

        response = get_chatbot_response(user_input)
        st.session_state.chat_history.append(("You", user_input))
        st.session_state.chat_history.append(("Bot", response))

    for role, message in st.session_state.chat_history:
        st.markdown(f"**{role}:** {message}")
