"""Streamlit web app for CropWise — Crop Prediction."""
import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os

MODEL_PATH = "models/model.pkl"

st.set_page_config(page_title="CropWise", page_icon="🌾", layout="centered")

st.title("🌾 CropWise — Smart Crop Recommendation")
st.write("AI-powered crop prediction based on soil and weather data.")

# Ensure model exists
if not os.path.exists(MODEL_PATH):
    st.error("❌ Trained model not found! Please run `python src/pipeline.py` first.")
    st.stop()

model = joblib.load(MODEL_PATH)
st.success("✅ Model loaded successfully!")

# --- Input Mode ---
mode = st.radio("Choose input mode:", ["Manual Input", "Upload CSV"])

if mode == "Manual Input":
    st.subheader("Enter Conditions")

    # Create numeric inputs dynamically (example common features)
    n_features = st.number_input("How many features does your model expect?", min_value=1, max_value=20, value=5)
    features = []
    for i in range(n_features):
        val = st.number_input(f"Feature {i+1} value:", value=0.0)
        features.append(val)

    if st.button("Predict Crop"):
        try:
            pred = model.predict([features])[0]
            st.success(f"🌱 Recommended Crop: **{pred}**")
        except Exception as e:
            st.error(f"Prediction failed: {e}")

else:
    st.subheader("Upload CSV File")
    uploaded = st.file_uploader("Upload a CSV with same columns used in training", type=["csv"])
    if uploaded:
        try:
            df = pd.read_csv(uploaded)
            st.write("Preview of uploaded data:", df.head())
            preds = model.predict(df)
            df["Predicted_Crop"] = preds
            st.write("Predictions:", df.head())
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("Download Predictions", data=csv, file_name="predictions.csv", mime="text/csv")
        except Exception as e:
            st.error(f"Error reading or predicting: {e}")

st.markdown("---")
st.caption("Built with ❤️ using Streamlit and scikit-learn.")
