import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import shap
import matplotlib.pyplot as plt

st.title("Urban Pollution Prediction 🚦")

# Load dataset
df = pd.read_csv("TRAQID.csv")

# Show columns for debugging
st.write("Columns in dataset:")
st.write(df.columns.tolist())

# Use the correct AQI column name
AQI_column = "aqi"  # Replace with the exact name from Step 2

if AQI_column in df.columns:

    # Features and target
    X = df.drop(columns=[AQI_column])
    y = df[AQI_column]

    # Encode categorical features
    for col in X.select_dtypes(include='object').columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = XGBRegressor()
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)
    st.subheader("Sample Predictions")
    st.write(y_pred[:5])

    # SHAP explainability
    explainer = shap.Explainer(model)
    shap_values = explainer(X_test)

    st.subheader("Feature Importance (SHAP)")
    shap.summary_plot(shap_values, X_test, show=False)
    st.pyplot(bbox_inches='tight')

else:
    st.warning(f"{AQI_column} column not found in dataset!")
