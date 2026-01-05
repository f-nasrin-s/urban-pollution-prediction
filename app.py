# ============================================================
# SMART CITY AI – URBAN POLLUTION COMMAND CENTER
# Domain: Smart Cities & Urban Intelligence
# Ultimate Hackathon Version (Safe + Robust)
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import requests
import folium
import pytz

from datetime import datetime
from streamlit_folium import st_folium
from folium.plugins import HeatMap

from xgboost import XGBRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# ------------------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------------------
st.set_page_config(page_title="Smart City Pollution AI", layout="wide")

st.title("🏙️ Smart City Pollution AI – Command Center")
st.caption("AI Prediction • Factor Attribution • Health Risk • City Intelligence")

ist = pytz.timezone("Asia/Kolkata")
st.caption(f"⏱️ System Time: {datetime.now(ist).strftime('%d %b %Y | %H:%M:%S IST')}")

# ------------------------------------------------------------
# MODEL TRAINING (CACHED)
# ------------------------------------------------------------
@st.cache_resource
def train_model():
    df = pd.read_csv("TRAQID.csv")

    aqi_col = [c for c in df.columns if "aqi" in c.lower()][0]
    drop_cols = ["Image", "created_at", "Sequence", "aqi_cat", aqi_col]

    X = df.drop(columns=[c for c in drop_cols if c in df.columns])
    y = df[aqi_col]

    encoders = {}
    for col in X.select_dtypes(include="object").columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        encoders[col] = le

    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = XGBRegressor(
        n_estimators=80,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=42
    )
    model.fit(X_train, y_train)

    return model, encoders, X.columns.tolist()

model, encoders, features = train_model()

# ------------------------------------------------------------
# LOCATION SELECTION (AUTO → MAP FALLBACK)
# ------------------------------------------------------------
st.subheader("📍 Location Selection")

def auto_location():
    try:
        res = requests.get("https://ipapi.co/json/", timeout=5).json()
        return float(res["latitude"]), float(res["longitude"])
    except:
        return None, None

mode = st.radio(
    "Choose location method:",
    ["📌 Auto Detect My Location", "🗺️ Select Location on Map"]
)

if mode == "📌 Auto Detect My Location":
    lat, lon = auto_location()

    if lat is None:
        st.warning("Auto detection unavailable — please select location on map.")
        mode = "🗺️ Select Location on Map"
    else:
        st.success(f"Detected → {lat:.4f}, {lon:.4f}")

if mode == "🗺️ Select Location on Map":
    base_map = folium.Map(location=[20.5937, 78.9629], zoom_start=5)
    map_data = st_folium(base_map, height=420)

    if not map_data or not map_data.get("last_clicked"):
        st.info("Click anywhere on the map to select location")
        st.stop()

    lat = float(map_data["last_clicked"]["lat"])
    lon = float(map_data["last_clicked"]["lng"])
    st.success(f"Selected → {lat:.4f}, {lon:.4f}")

# ------------------------------------------------------------
# LIVE AIR POLLUTION DATA
# ------------------------------------------------------------
API_KEY = st.secrets.get("OPENWEATHER_API_KEY", "")
if not API_KEY:
    st.error("OpenWeather API key missing")
    st.stop()

url = f"https://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={API_KEY}"
pollution = requests.get(url).json()
components = pollution["list"][0]["components"]

pm25 = float(components["pm2_5"])
pm10 = float(components["pm10"])

c1, c2 = st.columns(2)
c1.metric("PM2.5 (µg/m³)", pm25)
c2.metric("PM10 (µg/m³)", pm10)

# ------------------------------------------------------------
# AQI PREDICTION
# ------------------------------------------------------------
row = {}
for col in features:
    if col.lower() == "pm2.5":
        row[col] = pm25
    elif col.lower() == "pm10":
        row[col] = pm10
    else:
        row[col] = 0

predicted_aqi = float(model.predict(pd.DataFrame([row]))[0])

# ------------------------------------------------------------
# ADVANCED ANALYTICS
# ------------------------------------------------------------
risk_score = min(100, round(predicted_aqi / 3))

if pm25 > 1.3 * pm10:
    source = "🚗 Traffic & Combustion Emissions"
elif pm10 > 1.3 * pm25:
    source = "🏗️ Dust / Construction Activity"
else:
    source = "🏭 Mixed Urban Emissions"

health_risk = {
    "Children": "High" if predicted_aqi > 120 else "Moderate",
    "Elderly": "High" if predicted_aqi > 100 else "Moderate",
    "Asthma Patients": "Severe" if predicted_aqi > 90 else "Moderate"
}

# ------------------------------------------------------------
# DISPLAY RESULTS
# ------------------------------------------------------------
st.subheader("🔮 AQI Prediction")
st.metric("Predicted AQI", f"{predicted_aqi:.2f}")
st.metric("Urban Risk Index", f"{risk_score} / 100")

st.subheader("🧪 Factor Attribution")
st.write(f"**Dominant Pollutant:** {'PM2.5' if pm25 > pm10 else 'PM10'}")
st.write(f"**Likely Source:** {source}")

st.subheader("❤️ Health Impact")
for k, v in health_risk.items():
    st.write(f"- **{k}** → {v} risk")

# ------------------------------------------------------------
# HOTSPOT MAP
# ------------------------------------------------------------
st.subheader("🗺️ Pollution Hotspot Map")

m2 = folium.Map(location=[lat, lon], zoom_start=12)

folium.CircleMarker(
    [lat, lon],
    radius=20,
    fill=True,
    fill_color="red",
    fill_opacity=0.85,
    popup=f"AQI: {round(predicted_aqi,2)}"
).add_to(m2)

HeatMap([[lat, lon, predicted_aqi]], radius=35).add_to(m2)

st_folium(m2, height=420)

# ------------------------------------------------------------
# SMART CITY DECISION ENGINE
# ------------------------------------------------------------
st.subheader("🏛️ Smart City Action Engine")

if predicted_aqi > 180:
    st.error("""
🚨 SEVERE POLLUTION ALERT  
• Emergency public advisory  
• Heavy traffic restriction  
• Construction halt  
• Deploy air purification units
""")
elif predicted_aqi > 120:
    st.warning("""
⚠️ MODERATE–HIGH POLLUTION  
• Remote work advisory  
• Traffic congestion control  
• Continuous monitoring
""")
else:
    st.success("""
✅ LOW POLLUTION ZONE  
• Normal activities allowed  
• Promote green mobility
""")
