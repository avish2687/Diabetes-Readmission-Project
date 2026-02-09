import pandas as pd
from sklearn.preprocessing import LabelEncoder

def preprocess_data(df):
    # Replace missing values
    df = df.replace('?', pd.NA)

    # Drop columns with too many missing values
    drop_cols = ['weight', 'payer_code', 'medical_specialty']
    df = df.drop(columns=drop_cols)

    # Drop remaining missing rows
    df = df.dropna()

    # Binary target
    df['readmitted'] = df['readmitted'].apply(
        lambda x: 1 if x == '<30' else 0
    )

    # Drop IDs
    df = df.drop(columns=['encounter_id', 'patient_nbr'])

    # Encode categorical features
    categorical_cols = df.select_dtypes(include='object').columns
    le = LabelEncoder()

    for col in categorical_cols:
        df[col] = le.fit_transform(df[col])

    return df
