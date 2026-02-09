import pandas as pd
from sklearn.preprocessing import LabelEncoder

def preprocess_data(df):
    df = df.replace('?', pd.NA)
    df = df.dropna()

    df['readmitted'] = df['readmitted'].apply(
        lambda x: 1 if x == '<30' else 0
    )

    categorical_cols = df.select_dtypes(include='object').columns
    le = LabelEncoder()

    for col in categorical_cols:
        df[col] = le.fit_transform(df[col])

    return df
