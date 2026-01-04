# app.py
import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import shap
import matplotlib.pyplot as plt

# -----------------------
# Title
# -----------------------
st.title("Urban Pollution Prediction 🚦")
st.markdown(
    "Predict AQI using environmental data and visualize feature importance with SHAP. "
    "Select filters below to get dynamic predictions."
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
# Identify city columns
# -----------------------
city_cols = [col for col in df.columns if col.startswith("City_")]
if city_cols:
    st.subheader("Select City")
    city_options = [col.replace("City_", "") for col in city_cols]
    selected_city = st.selectbox("City", city_options)
    # Filter data for selected city
    df = df[df[f"City_{selected_city}"] == 1].reset_index(drop=True)
else:
    selected_city = None

# -----------------------
# Prepare features and target
# -----------------------
drop_cols = ["Image", "created_at", "Sequence", "aqi_cat", AQI_column] + city_cols
X = df.drop(columns=[col for col in drop_cols if col in df.columns])
y = df[AQI_column]

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
# Interactive filters
# -----------------------
st.subheader("Predict AQI for Custom Inputs")

# Function to inverse transform encoded categories
def decode(col, val):
    le = label_encoders.get(col)
    if le:
        return le.inverse_transform([val])[0]
    return val

input_data = {}
for col in X.columns:
    if col in label_encoders:
        options = label_encoders[col].classes_
        selected = st.selectbox(f"{col}", options, key=f"input_{col}")
        input_data[col] = label_encoders[col].transform([selected])[0]
    else:
        val = st.number_input(f"{col}", float(X[col].min()), float(X[col].max()), float(X[col].mean()))
        input_data[col] = val

# Predict button
if st.button("Predict AQI"):
    input_df = pd.DataFrame([input_data])
    prediction = model.predict(input_df)[0]
    st.success(f"Predicted AQI: {prediction:.2f}")

# -----------------------
# Show sample predictions
# -----------------------
st.subheader(f"Sample Predictions{' for ' + selected_city if selected_city else ''}")
y_pred = model.predict(X_test)
st.write(pd.DataFrame(y_pred, columns=["Predicted AQI"]).head(10))

# -----------------------
# SHAP feature importance
# -----------------------
st.subheader(f"Feature Importance (SHAP){' for ' + selected_city if selected_city else ''}")

explainer = shap.Explainer(model)
shap_values = explainer(X_test)

fig, ax = plt.subplots(figsize=(10, 6))
shap.summary_plot(shap_values, X_test, show=False)
st.pyplot(fig, bbox_inches='tight')
