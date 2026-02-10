from fastapi import FastAPI
import joblib
import pandas as pd
from src.preprocess import preprocess_diabetes_data

app = FastAPI()

model = joblib.load("models/readmission_model.pkl")
