from pydantic import BaseModel
from typing import Optional

class PredictionRequest(BaseModel):
    age: int
    tenure: int
    monthly_charges: float
    total_charges: float

class PredictionResponse(BaseModel):
    prediction: int
    probability: float
    model_version: str

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    mlflow_connection: bool
    uptime_seconds: float
    memory_usage_mb: float