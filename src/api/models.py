from pydantic import BaseModel
from typing import Optional, Dict, Any

class PredictionRequest(BaseModel):
    age: int
    tenure: int
    monthly_charges: float
    total_charges: float

class PredictionResponse(BaseModel):
    prediction: int
    probability: float
    model_version: str

class ComponentHealth(BaseModel):
    status: str
    details: Dict[str, Any] = {}
    
class HealthResponse(BaseModel):
    status: str
    timestamp: str
    uptime_seconds: float
    checks: Dict[str, ComponentHealth]