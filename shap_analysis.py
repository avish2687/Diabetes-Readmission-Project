import shap
import joblib
import pandas as pd
from Preprocess import preprocess_data

model = joblib.load("readmission_model.pkl")

df = pd.read_csv("data/diabetic_data.csv")
df = preprocess_data(df)

X = df.drop("readmitted", axis=1)

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

# Global feature importance
shap.summary_plot(shap_values[1], X, plot_type="bar")

# Local explanation (single patient)
shap.force_plot(
    explainer.expected_value[1],
    shap_values[1][0],
    X.iloc[0],
    matplotlib=True
)
