import pandas as pd
from sklearn.preprocessing import LabelEncoder

def preprocess_data(df):
    
    df = df.replace('?', pd.NA)

    # Drop high-missing columns
    df = df.drop(columns=['weight', 'payer_code', 'medical_specialty'])

    # Drop remaining missing rows
    df = df.dropna()

    # Target variable
    df['readmitted'] = df['readmitted'].apply(
        lambda x: 1 if x == '<30' else 0
    )

    # Drop IDs
    df = df.drop(columns=['encounter_id', 'patient_nbr'])

    # Encode categorical columns
    categorical_cols = df.select_dtypes(include='object').columns
    le = LabelEncoder()

    for col in categorical_cols:
        df[col] = le.fit_transform(df[col])

    return df

