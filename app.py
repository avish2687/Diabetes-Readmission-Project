import streamlit as st
import joblib
import numpy as np

model = joblib.load("readmission_model.pkl")

st.title("🏥 Diabetes Readmission Risk Predictor")

st.write("Predict 30-day hospital readmission risk")

age = st.slider("Age", 20, 100, 50)
time_in_hospital = st.slider("Days in Hospital", 1, 14, 3)
num_medications = st.slider("Number of Medications", 1, 50, 10)
num_procedures = st.slider("Number of Procedures", 0, 10, 1)

if st.button("Predict Risk"):
    input_data = np.array([[age, time_in_hospital, num_medications, num_procedures]])
    risk = model.predict_proba(input_data)[0][1]

    if risk > 0.7:
        st.error(f"🔴 High Risk of Readmission ({risk:.2f})")
    elif risk > 0.4:
        st.warning(f"🟠 Medium Risk ({risk:.2f})")
    else:
        st.success(f"🟢 Low Risk ({risk:.2f})")
