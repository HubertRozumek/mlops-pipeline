from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from typing import List
import logging
import time
from .models import PredictionRequest, PredictionResponse, HealthResponse
from .health import HealthChecker

# Metrics
PREDICTION_COUNTER = Counter('predictions_total', 'Total predictions made')
PREDICTION_LATENCY = Histogram('prediction_duration_seconds', 'Prediction latency')
ERROR_COUNTER = Counter('prediction_errors_total', 'Total prediction errors')

app = FastAPI(title="MLOps Pipeline API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
model = None
health_checker = HealthChecker()

@app.on_event("startup")
async def load_model():
    global model, model_version
    try:
        # Load latest model from MLflow
        client = mlflow.tracking.MlflowClient()
        model_name = "customer_churn_predictor"
        latest_version = client.get_latest_versions(model_name, stages=["Production"])
        
        if latest_version:
            model_uri = f"models:/{model_name}/{latest_version[0].version}"
            model = mlflow.sklearn.load_model(model_uri)
            model_version = latest_version[0].version  
            logging.info(f"Loaded model version {latest_version[0].version}")
        else:
            logging.warning("No production model found, using None model")
            
    except Exception as e:
        logging.error(f"Failed to load model: {e}")

@app.get("/health", response_model=HealthResponse)
async def health_check():
    return await health_checker.get_health_status()

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    start_time = time.time()
    
    try:
        if model is None:
            ERROR_COUNTER.inc()
            raise HTTPException(status_code=503, detail="Model not available")
        
        # Convert request to DataFrame
        input_data = pd.DataFrame([request.dict()])
        
        # Make prediction
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]
        
        PREDICTION_COUNTER.inc()
        PREDICTION_LATENCY.observe(time.time() - start_time)
        
        return PredictionResponse(
            prediction=int(prediction),
            probability=float(probability),
            model_version=model_version or "unknown"
        )
        
    except Exception as e:
        ERROR_COUNTER.inc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)