# Real-time Fraud Detection API

A machine learning API that detects fraudulent credit card transactions in real-time using XGBoost, served via FastAPI, and containerized with Docker.

## Results

| Metric | Score |
|--------|-------|
| ROC-AUC | 0.9737 |
| PR-AUC | 0.8287 |
| Precision | 0.8478 |
| Recall | 0.8211 |
| F2 Score | 0.8263 |
| Threshold | 0.0891 |

## Tech Stack

- **ML** — XGBoost, Scikit-learn, Optuna, MLflow
- **API** — FastAPI, Pydantic, Uvicorn
- **DevOps** — Docker, GitHub Actions

## Project Structure

```
fraud-detection-api/
├── app/
│   ├── main.py       # FastApi Code 
├── models/
│   ├── fraud_model.json        
│   ├── scaler.pkl             
│   └── model_config.json        # threshold + metadata
├
├── requirements.txt
└── README.md
```

## Setup

### 1 — Place your model files
Copy the files downloaded from Colab into the `models/` folder:
```
models/fraud_model.json
models/scaler.pkl
```

### 2 — Create virtual environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### 3 — Install dependencies
```bash
pip install -r requirements.txt
```

### 4 — Run the API
```bash
uvicorn app.main:app --reload
```



## API Endpoints

| Method | Endpoint         | Description                    |
|--------|-----------------|--------------------------------|
| GET    | /               | Root — API status              |
| GET    | /health         | Model loaded + threshold info  |
| POST   | /predict        | Single transaction prediction  |
| GET    | /docs           | Swagger UI (auto-generated)    |

## Sample Request

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
      "transaction_id": "txn_fraud_002",
      "amount": 0.5823449889135255,
      "time": -0.09819604099973566,
      "v1": -3.0435, "v2": 4.5312,
      "v3": -3.9056, "v4": 4.3659,
      "v5": -3.0435, "v6": -1.3556,
      "v7": -3.0435, "v8": 0.4331,
      "v9": -0.3870, "v10": -3.0435,
      "v11": 1.9465, "v12": -5.1322,
      "v13": 0.8182, "v14": -6.0546,
      "v15": 0.1414, "v16": -2.4623,
      "v17": -8.6627, "v18": -0.8300,
      "v19": 0.0846, "v20": 0.1370,
      "v21": 0.5966, "v22": 1.4929,
      "v23": 0.6401, "v24": 0.1617,
      "v25": 0.2309, "v26": 0.6699,
      "v27": 0.1274, "v28": 0.0641
    }'
```

## Sample Response

```json
{
  "transaction_id": "txn_001",
  "fraud_score": 0.0234,
  "is_fraud": false,
  "threshold_used": 0.0891,
  "latency_ms": 4.21
}
```

## Docker Command
Start working  →  docker-compose up -d
Test API       →  http://localhost:8000/docs
Stop working   →  docker-compose down
