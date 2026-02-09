import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from Preprocess import preprocess_data

df = pd.read_csv("data/diabetic_data.csv")
df = preprocess_data(df)

X = df.drop("readmitted", axis=1)
y = df["readmitted"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# -----------------------------
# Bias-aware model
# -----------------------------
model = RandomForestClassifier(
    n_estimators=300,
    class_weight="balanced",  # critical for fairness
    max_depth=12,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(classification_report(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]))

joblib.dump(model, "readmission_model.pkl")
joblib.dump(X.columns.tolist(), "feature_names.pkl")
