from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import json
import random
import math
import pandas as pd
from dataset_generator import dataset_generator
from forecasting_algorithms import forecasting_algorithms

app = FastAPI(title="Energy Forecasting API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:5174", "http://127.0.0.1:5173", "http://127.0.0.1:5174"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class EnergyDataPoint(BaseModel):
    consumption: float
    hour: int
    dayofweek: int
    month: int
    dayofyear: int

class PredictionRequest(BaseModel):
    data: List[EnergyDataPoint]
    model_type: str = "both"

class PredictionResponse(BaseModel):
    gru_prediction: Optional[float] = None
    lstm_prediction: Optional[float] = None
    next_hour_forecast: Dict[str, Any]
    ensemble_prediction: Optional[Dict[str, Any]] = None
    algorithm_comparison: Optional[Dict[str, float]] = None

def generate_enhanced_sample_data(hours=168):
    """Generate enhanced realistic energy consumption data using dataset generator"""
    dataset_result = dataset_generator.generate_multiple_algorithms_data(hours)
    
    # Convert to the expected format
    sample_data = []
    for record in dataset_result['historical_data']:
        sample_data.append({
            'consumption': round(record['consumption'], 2),
            'hour': record['hour'],
            'dayofweek': record['day_of_week'],
            'month': record['month'],
            'dayofyear': record['day_of_year'],
            'temperature': round(record['temperature'], 1),
            'humidity': round(record['humidity'], 1),
            'season': record['season'],
            'is_holiday': record['is_holiday']
        })
    
    return {
        'data': sample_data,
        'statistics': dataset_result['statistics'],
        'correlations': dataset_result['correlations']
    }

def enhanced_prediction(data, model_type="ensemble"):
    """Enhanced predictions using multiple algorithms"""
    if len(data) < 10:
        raise ValueError("Need at least 10 data points for prediction")
    
    # Convert to DataFrame for advanced algorithms
    df_data = []
    for point in data:
        df_data.append({
            'consumption': point.consumption,
            'hour': point.hour,
            'day_of_week': point.dayofweek,
            'day_of_year': point.dayofyear,
            'month': point.month,
            'temperature': getattr(point, 'temperature', 20),  # Default temperature
            'humidity': getattr(point, 'humidity', 50)  # Default humidity
        })
    
    df = pd.DataFrame(df_data)
    
    # Get ensemble prediction
    ensemble_result = forecasting_algorithms.ensemble_forecast(df)
    
    return ensemble_result

@app.get("/")
async def root():
    return {"message": "Energy Forecasting API", "status": "running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "energy-forecasting"}

@app.get("/sample-data")
async def get_sample_data(hours: int = 168):
    """Get enhanced sample energy consumption data with statistics"""
    try:
        # Set seed for consistent data
        random.seed(42)
        enhanced_data = generate_enhanced_sample_data(hours)
        return {
            "data": enhanced_data['data'], 
            "count": len(enhanced_data['data']),
            "statistics": enhanced_data['statistics'],
            "correlations": enhanced_data['correlations']
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict", response_model=PredictionResponse)
async def predict_energy_consumption(request: PredictionRequest):
    """Predict next hour energy consumption using multiple algorithms"""
    try:
        if len(request.data) < 10:
            raise HTTPException(status_code=400, detail="Need at least 10 data points for prediction")
        
        # Get enhanced predictions
        ensemble_result = enhanced_prediction(request.data, request.model_type)
        
        # Get simple predictions for backward compatibility
        recent_values = [point.consumption for point in request.data[-10:]]
        avg_recent = sum(recent_values) / len(recent_values)
        
        response = PredictionResponse(
            gru_prediction=round(ensemble_result['individual_predictions']['holt_winters'], 2),
            lstm_prediction=round(ensemble_result['individual_predictions']['linear_regression'], 2),
            ensemble_prediction=ensemble_result,
            algorithm_comparison=ensemble_result['individual_predictions'],
            next_hour_forecast={
                "timestamp": "next_hour",
                "ensemble_prediction": round(ensemble_result['ensemble_prediction'], 2),
                "confidence_interval": ensemble_result['confidence_interval'],
                "prediction_variance": round(ensemble_result['prediction_variance'], 3),
                "confidence": round((1 - ensemble_result['prediction_variance'] / avg_recent) * 0.9, 2)
            }
        )
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")

@app.get("/model-info")
async def get_model_info():
    """Get information about the models"""
    return {
        "models": {
            "gru": {
                "input_dim": 5,
                "hidden_dim": 256,
                "output_dim": 1,
                "n_layers": 2,
                "window_size": 90,
                "status": "mock_ready"
            },
            "lstm": {
                "input_dim": 5,
                "hidden_dim": 256,
                "output_dim": 1,
                "n_layers": 2,
                "window_size": 90,
                "status": "mock_ready"
            }
        },
        "features": ["consumption", "hour", "dayofweek", "month", "dayofyear"],
        "note": "This is a mock API for demonstration. Replace with real models for production."
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
