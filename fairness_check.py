import pandas as pd
import joblib
from sklearn.metrics import recall_score
from Preprocess import preprocess_data

model = joblib.load("readmission_model.pkl")

df = pd.read_csv("data/diabetic_data.csv")
df = preprocess_data(df)

X = df.drop("readmitted", axis=1)
y_true = df["readmitted"]
y_pred = model.predict(X)

df["y_true"] = y_true
df["y_pred"] = y_pred

def recall_by_group(df, group_col):
    print(f"\nRecall by {group_col}:")
    for g in df[group_col].unique():
        subset = df[df[group_col] == g]
        print(g, recall_score(subset["y_true"], subset["y_pred"]))

# Age fairness
recall_by_group(df, "age")
