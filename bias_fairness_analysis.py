import pandas as pd
import joblib
from sklearn.metrics import recall_score, confusion_matrix
from Preprocess import preprocess_data

# -----------------------------
# Load model and data
# -----------------------------
model = joblib.load("readmission_model.pkl")

df = pd.read_csv("data/diabetic_data.csv")
df = preprocess_data(df)

X = df.drop("readmitted", axis=1)
y_true = df["readmitted"]
y_pred = model.predict(X)

df["y_true"] = y_true
df["y_pred"] = y_pred

# -----------------------------
# FAIRNESS FUNCTION
# -----------------------------
def fairness_by_group(df, group_col):
    results = []

    for group in df[group_col].unique():
        subset = df[df[group_col] == group]

        tn, fp, fn, tp = confusion_matrix(
            subset["y_true"],
            subset["y_pred"]
        ).ravel()

        recall = recall_score(subset["y_true"], subset["y_pred"])
        false_negative_rate = fn / (fn + tp)

        results.append({
            "Group": group,
            "Recall": round(recall, 3),
            "False Negative Rate": round(false_negative_rate, 3),
            "Samples": len(subset)
        })

    return pd.DataFrame(results)

# -----------------------------
# AGE FAIRNESS CHECK
# -----------------------------
age_fairness = fairness_by_group(df, "age")
print("\n=== FAIRNESS CHECK: AGE GROUPS ===")
print(age_fairness)

# -----------------------------
# GENDER FAIRNESS CHECK (if available)
# -----------------------------
if "gender" in df.columns:
    gender_fairness = fairness_by_group(df, "gender")
    print("\n=== FAIRNESS CHECK: GENDER ===")
    print(gender_fairness)
