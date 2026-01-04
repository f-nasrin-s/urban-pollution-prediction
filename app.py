# ==============================
# Urban Pollution Prediction App
# ==============================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import requests

from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score
from streamlit_autorefresh import st_autorefresh

# -------------------------------
# AUTO REFRESH (every 60 seconds)
# -------------------------------
st_autorefresh(interval=60 * 1000, key="refresh")

# -------------------------------
# PAGE TITLE
# -------------------------------
st.set_page_config(page_title="Urban AQI Prediction", layout="wide")
st.title("Urban Pollution Prediction 🚦")
st.write("ML-powered AQI prediction with live data, explainability & alerts")

# -------------------------------
# LOAD DATASET
# -------------------------------
df = pd.read_csv("TRAQID.csv")

# -------------------------------
# FIND AQI COLUMN
# -------------------------------
aqi_col = [c for c in df.columns if "aqi" in c.lower()][0]

# -------------------------------
# CITY SELECTION (OPTIONAL)
# -------------------------------
if "City" in df.columns:
    city = st.selectbox("Select City", df["City"].unique())
    df = df[df["City"] == city]
else:
    city = "Default City"

# -------------------------------
# PREPARE FEATURES
# -------------------------------
drop_cols = ["Image", "created_at", "Sequence", "aqi_cat", aqi_col]
X = df.drop(columns=[c for c in drop_cols if c in df.columns])
y = df[aqi_col]

label_encoders = {}
for col in X.select_dtypes(include="object").columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le

# -------------------------------
# TRAIN TEST SPLIT
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# MODEL TRAINING
# -------------------------------
model = XGBRegressor(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=5,
    random_state=42
)
model.fit(X_train, y_train)

# -------------------------------
# LIVE INPUT SECTION
# -------------------------------
st.subheader("🔮 Live AQI Prediction")

input_data = {}
for col in X.columns:
    if col in label_encoders:
        option = st.selectbox(col, label_encoders[col].classes_)
        input_data[col] = label_encoders[col].transform([option])[0]
    else:
        input_data[col] = st.slider(
            col,
            float(X[col].min()),
            float(X[col].max()),
            float(X[col].mean())
        )

input_df = pd.DataFrame([input_data])
prediction = model.predict(input_df)[0]

# -------------------------------
# AQI CATEGORY + ALERT
# -------------------------------
def aqi_label(val):
    if val <= 50:
        return "Good", "🟢 Safe for outdoor activities"
    elif val <= 100:
        return "Moderate", "🟡 Sensitive people be cautious"
    elif val <= 150:
        return "Unhealthy (Sensitive)", "🟠 Avoid long exposure"
    elif val <= 200:
        return "Unhealthy", "🔴 Stay indoors"
    elif val <= 300:
        return "Very Unhealthy", "🟣 Health warning"
    else:
        return "Hazardous", "⚫ Emergency condition"

label, alert = aqi_label(prediction)

st.metric("Predicted AQI", f"{prediction:.2f}")
st.warning(f"{label} — {alert}")

# -------------------------------
# LIVE AQI FROM API
# -------------------------------
st.subheader("🌍 Live AQI (API)")

API_KEY = st.secrets.get("OPENWEATHER_API_KEY", "")

city_coords = {
    "Delhi": (28.61, 77.20),
    "Mumbai": (19.07, 72.87),
    "Chennai": (13.08, 80.27),
    "Bangalore": (12.97, 77.59)
}

if city in city_coords and API_KEY:
    lat, lon = city_coords[city]
    url = f"https://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={API_KEY}"
    res = requests.get(url).json()
    live_aqi = res["list"][0]["main"]["aqi"] * 50
    st.info(f"Live AQI: {live_aqi}")

# -------------------------------
# LIVE vs PREDICTED
# -------------------------------
if "live_aqi" in locals():
    compare = pd.DataFrame({
        "AQI": ["Live AQI", "Predicted AQI"],
        "Value": [live_aqi, prediction]
    })
    st.bar_chart(compare.set_index("AQI"))

# -------------------------------
# MODEL PERFORMANCE
# -------------------------------
y_pred = model.predict(X_test)
st.subheader("📊 Model Performance")
st.metric("MAE", f"{mean_absolute_error(y_test, y_pred):.2f}")
st.metric("R² Score", f"{r2_score(y_test, y_pred):.2f}")

# -------------------------------
# SHAP EXPLAINABILITY
# -------------------------------
st.subheader("🧠 Feature Importance (SHAP)")
explainer = shap.Explainer(model)
shap_values = explainer(X_test)

fig, ax = plt.subplots()
shap.summary_plot(shap_values, X_test, show=False)
st.pyplot(fig)

# -------------------------------
# MAP VIEW
# -------------------------------
if city in city_coords:
    map_df = pd.DataFrame(
        {"lat": [city_coords[city][0]], "lon": [city_coords[city][1]]}
    )
    st.subheader("🗺️ Pollution Location")
    st.map(map_df)

# -------------------------------
# IMPACT STATEMENT
# -------------------------------
st.success(
    "Impact: Helps citizens plan outdoor activities and enables authorities "
    "to take early pollution-control measures."
)


