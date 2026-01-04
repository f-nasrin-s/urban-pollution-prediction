# app.py
import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import shap
import matplotlib.pyplot as plt
import requests
import io

# -----------------------
# Title
# -----------------------
st.title("Urban Pollution Prediction 🚦")
st.markdown(
    "Predict AQI using environmental data and visualize feature importance dynamically. "
    "Toggle between Global (overall) and Local (input-specific) SHAP explanations."
)

# -----------------------
# Load dataset
# -----------------------
df = pd.read_csv("TRAQID.csv")

st.subheader("Columns in dataset:")
st.write(df.columns.tolist())

# -----------------------
# Detect AQI column
# -----------------------
aqi_cols = [col for col in df.columns if "AQI" in col.upper()]
if not aqi_cols:
    st.error("AQI column not found in dataset!")
    st.stop()
AQI_column = aqi_cols[0]

# -----------------------
# City-based selection (if column exists)
# -----------------------
if "City" in df.columns:
    city = st.selectbox("Select City", df["City"].unique())
    city_data = df[df["City"] == city]
else:
    city = "All Cities"
    city_data = df

# -----------------------
# Prepare features and target
# -----------------------
drop_cols = ["Image", "created_at", "Sequence", "aqi_cat", AQI_column]
X = city_data.drop(columns=[col for col in drop_cols if col in city_data.columns])
y = city_data[AQI_column]

# Encode categorical features
label_encoders = {}
for col in X.select_dtypes(include='object').columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le

# -----------------------
# Train-test split
# -----------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------
# Train XGBoost model
# -----------------------
model = XGBRegressor(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=5,
    random_state=42
)
model.fit(X_train, y_train)

# -----------------------
# Live AQI prediction inputs
# -----------------------
st.subheader("Predict AQI for Custom Inputs (Live Update)")
input_data = {}
for col in X.columns:
    if col in label_encoders:
        options = label_encoders[col].classes_
        selected = st.selectbox(f"{col}", options, key=f"input_{col}")
        input_data[col] = label_encoders[col].transform([selected])[0]
    else:
        val = st.slider(f"{col}", float(X[col].min()), float(X[col].max()), float(X[col].mean()))
        input_data[col] = val

input_df = pd.DataFrame([input_data])
prediction = model.predict(input_df)[0]

# -----------------------
# Color-coded AQI category
# -----------------------
def aqi_category(aqi):
    if aqi <= 50:
        return "Good", "green"
    elif aqi <= 100:
        return "Moderate", "yellow"
    elif aqi <= 150:
        return "Unhealthy for Sensitive Groups", "orange"
    elif aqi <= 200:
        return "Unhealthy", "red"
    elif aqi <= 300:
        return "Very Unhealthy", "purple"
    else:
        return "Hazardous", "maroon"

category, color = aqi_category(prediction)
st.success(f"Predicted AQI: {prediction:.2f}")
st.markdown(f"**AQI Category:** <span style='color:{color};font-weight:bold'>{category}</span>", unsafe_allow_html=True)

# -----------------------
# Optional live AQI from API
# -----------------------
API_KEY = "83350e70e4de15a991533bdd03e028ab"  # replace with your OpenWeatherMap API key
city_coords = {"Delhi": (28.6139, 77.2090), "Mumbai": (19.0760, 72.8777)}  # extend as needed
if city in city_coords:
    lat, lon = city_coords[city]
    url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={API_KEY}"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            live_aqi = response.json()['list'][0]['main']['aqi']
            st.info(f"Live AQI in {city}: {live_aqi}")
    except:
        st.warning("Could not fetch live AQI")

# -----------------------
# Global vs Local SHAP toggle
# -----------------------
st.subheader("Feature Importance (SHAP)")
shap_option = st.radio("Select SHAP Type:", ["Local (for this input)", "Global (overall)"])
explainer = shap.Explainer(model)
shap_values = explainer(input_df) if shap_option=="Local (for this input)" else explainer(X_test)

fig, ax = plt.subplots(figsize=(10, 6))
if shap_option == "Local (for this input)":
    shap.plots.bar(shap_values, show=False)
else:
    shap.summary_plot(shap_values, X_test, show=False)
st.pyplot(fig, bbox_inches='tight')

# -----------------------
# Predicted vs Actual AQI
# -----------------------
st.subheader("Predicted vs Actual AQI")
y_pred = model.predict(X_test)
fig2, ax2 = plt.subplots(figsize=(8,5))
ax2.scatter(y_test, y_pred, color='blue', alpha=0.6)
ax2.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
ax2.set_xlabel("Actual AQI")
ax2.set_ylabel("Predicted AQI")
ax2.set_title("Predicted vs Actual AQI")
st.pyplot(fig2)

# -----------------------
# Environmental feature trends
# -----------------------
st.subheader("Environmental Feature Trends")
for col in ["PM2.5", "PM10", "Temperature", "Humidity"]:
    if col in df.columns:
        st.line_chart(df[col])

# -----------------------
# Download Prediction & SHAP
# -----------------------
st.subheader("Download Prediction & SHAP Values")
shap_df = pd.DataFrame({
    "Feature": X.columns,
    "Value": input_df.iloc[0].values,
    "SHAP_value": shap_values.values[0] if shap_option=="Local (for this input)" else np.nan
})
shap_df.loc[-1] = ["Predicted AQI", prediction, np.nan]
shap_df.index = shap_df.index + 1
shap_df = shap_df.sort_index()
csv_buffer = io.StringIO()
shap_df.to_csv(csv_buffer, index=False)
st.download_button(
    label="Download CSV",
    data=csv_buffer.getvalue(),
    file_name="aqi_prediction_shap.csv",
    mime="text/csv"
)

# -----------------------
# Highlight impact
# -----------------------
st.markdown(
    "💡 **Impact:** Our app can help 1M+ citizens make daily decisions about outdoor activities "
    "and enable city planners to take proactive measures against pollution."
)

