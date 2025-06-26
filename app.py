
import streamlit as st
import numpy as np
import joblib

# Load the trained Parkinson’s model
model = joblib.load("parkinsons_model.pkl")

# Web page title
st.set_page_config(page_title="Parkinson's Prediction", layout="centered")
st.title("🧠 Idiopathic Parkinson’s Disease Prediction")
st.markdown("Enter voice measurements to predict the presence of Parkinson’s Disease.")

# Input fields (adjust if your model uses different features)
fo = st.number_input("MDVP:Fo(Hz)", min_value=0.0, step=0.1)
fhi = st.number_input("MDVP:Fhi(Hz)", min_value=0.0, step=0.1)
flo = st.number_input("MDVP:Flo(Hz)", min_value=0.0, step=0.1)
jitter = st.number_input("MDVP:Jitter(%)", min_value=0.0, step=0.001)
shimmer = st.number_input("MDVP:Shimmer", min_value=0.0, step=0.001)
spread1 = st.number_input("spread1", min_value=-10.0, step=0.1)
d2 = st.number_input("D2", min_value=0.0, step=0.01)

# Predict button
if st.button("Predict"):
    try:
        input_data = np.array([[fo, fhi, flo, jitter, shimmer, spread1, d2]])
        prediction = model.predict(input_data)

        if prediction[0] == 1:
            st.error("⚠️ Prediction: High chance of Idiopathic Parkinson’s Disease.")
        else:
            st.success("✅ Prediction: Likely Healthy.")
    except Exception as e:
        st.warning(f"Prediction failed. Error: {e}")
