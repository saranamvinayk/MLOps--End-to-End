import logging
import joblib
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from prometheus_fastapi_instrumentator import Instrumentator
import numpy as np
import os

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting API. Loading ML model into memory...")
    try:
        model_path = "models/model.pkl"
        if os.path.exists(model_path):
            ml_models["predictor"] = joblib.load(model_path)
            logger.info("Model loaded successfully.")
        else:
            logger.warning(f"Model not found at {model_path}. Using mock model.")
            ml_models["predictor"] = "mock_model_loaded"
    except Exception as e:
        logger.critical(f"Failed to load model: {e}")
        raise RuntimeError(f"Startup failed: {e}")
    
    yield
    
    logger.info("Shutting down. Clearing model memory...")
    ml_models.clear()

app = FastAPI(title="Real Estate ML Inference API", version="1.0.0", lifespan=lifespan)

instrumentator = Instrumentator().instrument(app)
instrumentator.expose(app, endpoint="/metrics", include_in_schema=False)

class PredictionRequest(BaseModel):
    sqft: float = Field(..., gt=0, description="Square footage")
    bedrooms: int = Field(..., gt=0, description="Number of bedrooms")
    bathrooms: float = Field(..., gt=0, description="Number of bathrooms")

class PredictionResponse(BaseModel):
    predicted_price: float
    model_version: str

@app.get("/health")
def health_check():
    if "predictor" not in ml_models:
        raise HTTPException(status_code=503, detail="Model is unavailable")
    return {"status": "healthy", "model_loaded": True}

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    try:
        features = np.array([[request.sqft, request.bedrooms, request.bathrooms]])
        if ml_models["predictor"] == "mock_model_loaded":
            prediction = 450000.0
        else:
            prediction = ml_models["predictor"].predict(features)[0]
        
        return PredictionResponse(predicted_price=prediction, model_version="v1.0.0")
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal inference error")
