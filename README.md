SolarQuest - Solar AI Analytics Platform

An end-to-end AI-powered Solar Energy Intelligence Platform built using Machine Learning, Deep Learning, Uncertainty Quantification, and Generative AI, deployed with an interactive Streamlit multi-page dashboard.

This platform enables solar radiation forecasting, uncertainty estimation, predictive maintenance, and intelligent conversational insights — all in one unified application.

🚀 Features
🌞 1. Solar Radiation Prediction

ML-based regression model for predicting solar radiation.

Takes environmental inputs such as:

Temperature

Pressure

Humidity

Wind Speed

Outputs predicted radiation in W/m².

📊 2. Uncertainty Quantification (Q10, Q50, Q90)

Quantile regression model.

Provides:

Q10 → Lower bound prediction

Q50 → Median estimate

Q90 → Upper bound prediction

Helps assess prediction reliability and risk margin.

⚙️ 3. Predictive Maintenance (MLP Model)

Multi-layer Perceptron model to detect system health.

Uses inputs like:

Panel Temperature

Voltage

Current

Dust Level

Humidity

Outputs:

✅ Normal Operation

⚠️ Maintenance Required

🤖 4. Solar AI Chatbot

Powered by GPT API.

Answers:

Solar performance queries

Energy generation insights

Maintenance explanations

App-related data questions

Maintains contextual conversation memory within session.

📈 5. Interactive Dashboard

Multi-page professional UI using Streamlit.

KPI cards

Real-time model inference

Clean, presentation-ready layout.


🛠️ Tech Stack

Python

Streamlit

Scikit-Learn

MLP (Neural Network)

Quantile Regression

OpenAI GPT API

Pandas / NumPy

Pickle (Model Serialization)
