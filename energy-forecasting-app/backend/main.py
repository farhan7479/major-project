from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import numpy as np
from models import forecasting_service

app = FastAPI(title="Energy Forecasting API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],  # Vite default port
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
    model_type: str = "both"  # "gru", "lstm", or "both"

class PredictionResponse(BaseModel):
    gru_prediction: Optional[float] = None
    lstm_prediction: Optional[float] = None
    next_hour_forecast: Dict[str, Any]

@app.get("/")
async def root():
    return {"message": "Energy Forecasting API", "status": "running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "energy-forecasting"}

@app.get("/sample-data")
async def get_sample_data(hours: int = 168):
    """Get sample energy consumption data"""
    try:
        sample_data = forecasting_service.generate_sample_data(hours)
        return {"data": sample_data, "count": len(sample_data)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict", response_model=PredictionResponse)
async def predict_energy_consumption(request: PredictionRequest):
    """Predict next hour energy consumption using LSTM/GRU models"""
    try:
        # Convert data to appropriate format
        data_array = []
        for point in request.data:
            data_array.append([
                point.consumption,
                point.hour,
                point.dayofweek,
                point.month,
                point.dayofyear
            ])
        
        # Preprocess data
        input_tensor = forecasting_service.preprocess_data(np.array(data_array))
        
        response = PredictionResponse(
            next_hour_forecast={
                "timestamp": "next_hour",
                "confidence": 0.85
            }
        )
        
        # Make predictions based on model type
        if request.model_type in ["gru", "both"]:
            try:
                gru_pred = forecasting_service.predict_gru(input_tensor)
                response.gru_prediction = float(gru_pred)
            except Exception as e:
                print(f"GRU prediction error: {e}")
                response.gru_prediction = None
        
        if request.model_type in ["lstm", "both"]:
            try:
                lstm_pred = forecasting_service.predict_lstm(input_tensor)
                response.lstm_prediction = float(lstm_pred)
            except Exception as e:
                print(f"LSTM prediction error: {e}")
                response.lstm_prediction = None
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")

@app.get("/model-info")
async def get_model_info():
    """Get information about the loaded models"""
    return {
        "models": {
            "gru": {
                "input_dim": forecasting_service.input_dim,
                "hidden_dim": forecasting_service.hidden_dim,
                "output_dim": forecasting_service.output_dim,
                "n_layers": forecasting_service.n_layers,
                "window_size": forecasting_service.window_size
            },
            "lstm": {
                "input_dim": forecasting_service.input_dim,
                "hidden_dim": forecasting_service.hidden_dim,
                "output_dim": forecasting_service.output_dim,
                "n_layers": forecasting_service.n_layers,
                "window_size": forecasting_service.window_size
            }
        },
        "features": ["consumption", "hour", "dayofweek", "month", "dayofyear"]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
