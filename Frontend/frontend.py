import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Solar Forecast Demo", layout="centered")

st.title("🌞 Solar Energy Forecasting – Uncertainty Demo")
st.write(
    "This is a **demo app** to understand how uncertainty-aware predictions work "
    "using lower, median, and upper bounds."
)

# ---------------------------
# 1️⃣ Create FAKE input data
# ---------------------------
st.subheader("🔧 Synthetic Input Features")

hours = np.arange(6, 18)  # 6 AM to 6 PM

data = pd.DataFrame({
    "hour": hours,
    "irradiation": np.clip(
        800 * np.sin((hours - 6) / 12 * np.pi), 0, None
    ),
    "temperature": 25 + 5 * np.sin((hours - 6) / 12 * np.pi),
})

st.dataframe(data)

# ---------------------------
# 2️⃣ Fake model predictions
# ---------------------------
# (Pretending these came from q10, q50, q90 models)

median_pred = data["irradiation"] * 0.5
noise = np.random.uniform(40, 80, size=len(median_pred))

lower_pred = median_pred - noise
upper_pred = median_pred + noise

# Ensure no negative power
lower_pred = np.clip(lower_pred, 0, None)

# ---------------------------
# 3️⃣ Show numerical output
# ---------------------------
st.subheader("🔮 Forecast Output")

st.metric(
    "Expected Power (kW)",
    f"{median_pred.mean():.1f}",
    f"± {(upper_pred.mean() - lower_pred.mean()) / 2:.1f}"
)

st.write(
    f"Typical Prediction Range: **{lower_pred.mean():.1f} – {upper_pred.mean():.1f} kW**"
)

# ---------------------------
# 4️⃣ Plot uncertainty band
# ---------------------------
st.subheader("📊 Prediction with Uncertainty Band")

fig, ax = plt.subplots(figsize=(9, 4))

ax.plot(hours, median_pred, label="Prediction (q50)", linewidth=2)
ax.fill_between(
    hours,
    lower_pred,
    upper_pred,
    alpha=0.3,
    label="Uncertainty Band (q10–q90)"
)

ax.set_xlabel("Hour of Day")
ax.set_ylabel("Solar Power (kW)")
ax.set_title("Solar Power Forecast with Uncertainty")
ax.legend()

st.pyplot(fig)

# ---------------------------
# 5️⃣ Explanation (key part)
# ---------------------------
st.subheader("🧠 What is happening here?")

st.markdown(
"""
- **Blue line (q50)** → Most likely solar power prediction  
- **Shaded region (q10–q90)** → Possible range due to uncertainty  
- Wider band = **less confidence**
- Narrow band = **high confidence**

In real projects:
- These curves come from **three trained ML models**
- Here we simulated them to explain the concept
"""
)

st.success("✅ This demo shows how uncertainty-aware forecasting works")

