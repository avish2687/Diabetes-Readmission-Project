import pandas as pd
import shap
import joblib
import matplotlib.pyplot as plt
from Preprocess import preprocess_data

# -----------------------------
# Load model and data
# -----------------------------
model = joblib.load("readmission_model.pkl")

df = pd.read_csv("data/diabetic_data.csv")
df = preprocess_data(df)

X = df.drop("readmitted", axis=1)

# -----------------------------
# Create SHAP Explainer
# -----------------------------
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

# -----------------------------
# GLOBAL EXPLANATION
# -----------------------------
print("Generating global SHAP summary plot...")
shap.summary_plot(
    shap_values[1],
    X,
    plot_type="bar",
    show=True
)

# -----------------------------
# LOCAL (PATIENT-LEVEL) EXPLANATION
# -----------------------------
patient_index = 5  # example patient
patient_data = X.iloc[patient_index]

print(f"Explaining prediction for patient index {patient_index}")

shap.waterfall_plot(
    shap.Explanation(
        values=shap_values[1][patient_index],
        base_values=explainer.expected_value[1],
        data=patient_data,
        feature_names=X.columns
    ),
    show=True
)
