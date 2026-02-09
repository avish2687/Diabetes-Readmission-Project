import streamlit as st
import joblib
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --------------------------------------------------
# Page setup
# --------------------------------------------------
st.set_page_config(
    page_title="Diabetes Readmission – Explainable AI",
    layout="centered"
)

st.title("🏥 Diabetes Readmission Risk Predictor")
st.write(
    "Clinical decision-support tool to predict **30-day hospital readmission risk** "
    "for diabetes patients with transparent explanations."
)

st.divider()

# --------------------------------------------------
# Load model & metadata
# --------------------------------------------------
model = joblib.load("readmission_model.pkl")
feature_names = joblib.load("feature_names.pkl")
explainer = shap.TreeExplainer(model)

# --------------------------------------------------
# Patient Inputs (UCI-aligned & doctor-friendly)
# --------------------------------------------------
st.subheader("🧾 Patient Information")

age = st.selectbox(
    "Age Group",
    [
        "[0-10)", "[10-20)", "[20-30)", "[30-40)",
        "[40-50)", "[50-60)", "[60-70)", "[70-80)",
        "[80-90)", "[90-100)"
    ]
)

time_in_hospital = st.slider("Days in Hospital", 1, 14, 3)
num_medications = st.slider("Number of Medications", 1, 50, 10)
num_lab_procedures = st.slider("Lab Procedures", 1, 100, 40)
number_inpatient = st.slider("Previous Inpatient Admissions", 0, 20, 1)

A1Cresult = st.selectbox("HbA1c Result", ["Norm", ">7", ">8"])
insulin = st.selectbox("Insulin Therapy", ["No", "Steady", "Up", "Down"])

# --------------------------------------------------
# Prepare input (match training features)
# --------------------------------------------------
input_dict = {
    "age": age,
    "time_in_hospital": time_in_hospital,
    "num_medications": num_medications,
    "num_lab_procedures": num_lab_procedures,
    "number_inpatient": number_inpatient,
    "A1Cresult": A1Cresult,
    "insulin": insulin
}

input_df = pd.DataFrame([input_dict])
input_df = pd.get_dummies(input_df)
input_df = input_df.reindex(columns=feature_names, fill_value=0)

st.divider()

# --------------------------------------------------
# Prediction
# --------------------------------------------------
if st.button("🔍 Predict Readmission Risk"):

    probability = model.predict_proba(input_df)[0][1]

    st.subheader("📊 Risk Assessment")

    if probability > 0.70:
        st.error(f"🔴 **High Risk of Readmission**\n\nProbability: **{probability:.2f}**")
    elif probability > 0.40:
        st.warning(f"🟠 **Moderate Risk of Readmission**\n\nProbability: **{probability:.2f}**")
    else:
        st.success(f"🟢 **Low Risk of Readmission**\n\nProbability: **{probability:.2f}**")

    # --------------------------------------------------
    # SHAP – Patient-level Explainability
    # --------------------------------------------------
    st.divider()
    st.subheader("🧠 Why this prediction? (Explainable AI)")

    shap_values = explainer.shap_values(input_df)

    st.write(
        "The chart below explains which clinical factors **increased** or **reduced** "
        "this patient’s readmission risk."
    )

    fig, ax = plt.subplots(figsize=(8, 4))
    shap.waterfall_plot(
        shap.Explanation(
            values=shap_values[1][0],
            base_values=explainer.expected_value[1],
            data=input_df.iloc[0],
            feature_names=input_df.columns
        ),
        show=False
    )

    st.pyplot(fig)

    # --------------------------------------------------
    # Bias & Fairness Note
    # --------------------------------------------------
    st.divider()
    st.subheader("⚖️ Fairness & Ethics")

    st.info(
        "This model was evaluated for fairness across age groups to ensure "
        "consistent recall and avoid under-identifying high-risk patients. "
        "Predictions are intended to **support**, not replace, clinical judgment."
    )

    st.caption(
        "⚠️ Clinical decision-support only. Final medical decisions must be made by healthcare professionals."
    )
