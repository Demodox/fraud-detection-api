import os
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel,Field, field_validator
import pickle
import logging
import time
import numpy as np
from xgboost import XGBClassifier


#LOGGING
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


# GLOBAL Variables
THRESHOLD = 0.0891 



# Load model & scaler 
try:
    # 1. Load XGBoost Model
    model_path = "models/fraud_model.json"
    if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
    
    model = XGBClassifier()
    model.load_model(model_path)
    logger.info(f"Model loaded successfully from {model_path}") 

    # 2. Load RobustScaler
    scaler_path = "models/scaler.pkl"
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler file not found at {scaler_path}")
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    logger.info("Scaler loaded successfully.")

except Exception as e:
    logger.error(f"Failed to load startup artifacts: {e}")
    raise RuntimeError("Application failed to initialize model artifacts.")



# App
app = FastAPI(
    title       = "Real-time Fraud Detection API",
    description = "Detects fraudulent credit card transactions using XGBoost",
    version     = "1.0.0"
    
)

# Request schema 
class Transaction(BaseModel):
    transaction_id: str   = Field(..., min_length=1, max_length=100, example="txn_001")
    amount: float = Field(..., example=-0.146)   
    time: float = Field(..., example=-0.276)   
    v1:  float = Field(..., example=-1.5488); v2:  float = Field(..., example= 1.8087)
    v3:  float = Field(..., example=-0.9535); v4:  float = Field(..., example= 2.2131)
    v5:  float = Field(..., example=-2.0157); v6:  float = Field(..., example=-0.9135)
    v7:  float = Field(..., example=-2.3560); v8:  float = Field(..., example= 1.1972)
    v9:  float = Field(..., example=-1.6784); v10: float = Field(..., example=-3.5387)
    v11: float = Field(..., example= 3.1021); v12: float = Field(..., example=-3.9934)
    v13: float = Field(..., example=-1.9374); v14: float = Field(..., example=-3.8229)
    v15: float = Field(..., example= 0.8310); v16: float = Field(..., example=-2.4754)
    v17: float = Field(..., example=-5.2119); v18: float = Field(..., example=-0.4139)
    v19: float = Field(..., example= 0.9333); v20: float = Field(..., example= 0.3908)
    v21: float = Field(..., example= 0.8551); v22: float = Field(..., example= 0.7747)
    v23: float = Field(..., example= 0.0590); v24: float = Field(..., example= 0.3432)
    v25: float = Field(..., example=-0.4689); v26: float = Field(..., example=-0.2783)
    v27: float = Field(..., example= 0.6259); v28: float = Field(..., example= 0.3956)

    @field_validator("transaction_id")
    @classmethod
    def id_not_blank(cls, v):
        if not v.strip():
            raise ValueError("transaction_id cannot be blank")
        return v.strip()

#Routes
@app.get("/",tags=["general"])
def home():
    return {
        "message": "Fraud Detection API is running!",
        "docs"   : "Visit /docs to test the API in browser",
        "health" : "Visit /health to check model status"
    }

@app.get("/health",tags=["general"])
def health():
    if model is None or scaler is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model or Scaler not initialized"
        )
    return {"status": "Healthy", "model_version": "XGBoost_v1", "threshold": THRESHOLD}


@app.post("/predict",tags=["ML Services"])
def predict(txn: Transaction):

    if model is None or scaler is None:
        raise HTTPException(
            status_code = 503,
            detail      = "Model not ready — server is still starting up"
        )
    
    try:
        # Step 1 — Scale amount and time
        scaled       = scaler.transform([[txn.amount, txn.time]])
        amt_scaled   = scaled[0][0]
        time_scaled  = scaled[0][1]

        # Step 2 — Build feature array 
        features = np.array([[
            
            time_scaled,
            txn.v1,  txn.v2,  txn.v3,  txn.v4,
            txn.v5,  txn.v6,  txn.v7,  txn.v8,
            txn.v9,  txn.v10, txn.v11, txn.v12,
            txn.v13, txn.v14, txn.v15, txn.v16,
            txn.v17, txn.v18, txn.v19, txn.v20,
            txn.v21, txn.v22, txn.v23, txn.v24,
            txn.v25, txn.v26, txn.v27, txn.v28,
            amt_scaled
        ]])

        # Step 3 — Predict
        start = time.time()
        fraud_score = float(model.predict_proba(features)[0][1])
        latency_ms = round((time.time() - start) * 1000, 2)
        is_fraud    = fraud_score >= THRESHOLD

        logger.info(
        f"{status} | txn={txn.transaction_id} | "
        f"score={fraud_score} | "
        f"amount=€{txn.amount} | "
        f"latency={latency_ms}ms"
       )

        # Step 4 — Return result
        return {
            "transaction_id": txn.transaction_id,
            "fraud_score":    round(fraud_score, 4),
            "is_fraud":       is_fraud,
            "threshold":      THRESHOLD,
            "latency_ms":     latency_ms
        }

    except Exception as e:
        logger.error(f"Prediction failed for {txn.transaction_id}: {e}")
        raise HTTPException(
            status_code = 500,
            detail= f"Prediction error: {str(e)}"
        )