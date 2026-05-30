from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
import os

app = FastAPI(title="Fraud Detection API")

base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
preprocessor_path = os.path.join(base_dir, 'data', 'processed', 'preprocessor.joblib')
model_path = os.path.join(base_dir, 'models', 'isolation_forest_optimized.joblib')

try:
    preprocessor = joblib.load(preprocessor_path)
    saved_data = joblib.load(model_path)
    model = saved_data['model']
    optimal_threshold = saved_data['threshold']
    print("Models and threshold loaded successfully.")
except Exception as e:
    print(f"Error loading models: {e}")


class Transaction(BaseModel):
    amount: float
    transaction_hour: int
    merchant_category: str
    foreign_transaction: int
    location_mismatch: int
    device_trust_score: int
    velocity_last_24h: int
    cardholder_age: int


@app.post("/predict")
def predict_fraud(transaction: Transaction):
    try:
        data = pd.DataFrame([transaction.model_dump()])
        processed_data = preprocessor.transform(data)

        score = -model.decision_function(processed_data)[0]

        is_fraud = 1 if score > optimal_threshold else 0

        return {
            "is_fraud": is_fraud,
            "anomaly_score": float(score)
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))