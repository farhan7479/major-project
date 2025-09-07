import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import math

class EnergyForecastingAlgorithms:
    """Collection of energy forecasting algorithms"""
    
    def __init__(self):
        self.scaler = StandardScaler()
    
    def moving_average(self, data: List[float], window: int = 24) -> float:
        """Simple moving average forecast"""
        if len(data) < window:
            return np.mean(data)
        return np.mean(data[-window:])
    
    def exponential_smoothing(self, data: List[float], alpha: float = 0.3) -> float:
        """Exponential smoothing forecast"""
        if not data:
            return 0
        
        result = data[0]
        for value in data[1:]:
            result = alpha * value + (1 - alpha) * result
        return result
    
    def holt_winters(self, data: List[float], season_length: int = 24, alpha: float = 0.3, 
                     beta: float = 0.1, gamma: float = 0.1) -> float:
        """Holt-Winters triple exponential smoothing"""
        if len(data) < season_length * 2:
            return self.exponential_smoothing(data, alpha)
        
        # Initialize components
        level = np.mean(data[:season_length])
        trend = (np.mean(data[season_length:2*season_length]) - np.mean(data[:season_length])) / season_length
        seasonal = [data[i] - level for i in range(season_length)]
        
        # Apply Holt-Winters
        for i in range(season_length, len(data)):
            prev_level = level
            level = alpha * (data[i] - seasonal[i % season_length]) + (1 - alpha) * (level + trend)
            trend = beta * (level - prev_level) + (1 - beta) * trend
            seasonal[i % season_length] = gamma * (data[i] - level) + (1 - gamma) * seasonal[i % season_length]
        
        return level + trend + seasonal[0]
    
    def linear_regression_forecast(self, df: pd.DataFrame) -> Dict:
        """Linear regression with multiple features"""
        if len(df) < 10:
            return {'prediction': df['consumption'].mean(), 'confidence': 0.5}
        
        # Prepare features
        features = ['hour', 'day_of_week', 'day_of_year', 'temperature', 'humidity']
        X = df[features].fillna(df[features].mean())
        y = df['consumption']
        
        # Fit model
        model = LinearRegression()
        model.fit(X, y)
        
        # Predict next hour
        last_row = X.iloc[-1:].copy()
        last_row['hour'] = (last_row['hour'] + 1) % 24
        
        prediction = model.predict(last_row)[0]
        
        # Calculate R² as confidence measure
        confidence = model.score(X, y)
        
        return {
            'prediction': max(0, prediction),
            'confidence': confidence,
            'feature_importance': dict(zip(features, model.coef_))
        }
    
    def seasonal_decomposition_forecast(self, data: List[float], period: int = 24) -> Dict:
        """Seasonal decomposition forecast"""
        if len(data) < period * 3:
            return {'prediction': np.mean(data), 'trend': 0, 'seasonal': 0}
        
        data_array = np.array(data)
        
        # Calculate trend using moving average
        trend = []
        for i in range(len(data_array)):
            start = max(0, i - period//2)
            end = min(len(data_array), i + period//2 + 1)
            trend.append(np.mean(data_array[start:end]))
        
        trend = np.array(trend)
        
        # Remove trend to get seasonal + noise
        detrended = data_array - trend
        
        # Calculate seasonal component
        seasonal = np.zeros(period)
        for i in range(period):
            seasonal_values = detrended[i::period]
            seasonal[i] = np.mean(seasonal_values) if len(seasonal_values) > 0 else 0
        
        # Forecast
        next_trend = trend[-1] + (trend[-1] - trend[-2]) if len(trend) > 1 else trend[-1]
        next_seasonal = seasonal[len(data) % period]
        
        prediction = next_trend + next_seasonal
        
        return {
            'prediction': max(0, prediction),
            'trend': next_trend,
            'seasonal': next_seasonal,
            'trend_direction': 'increasing' if len(trend) > 1 and trend[-1] > trend[-2] else 'decreasing'
        }
    
    def arima_simple(self, data: List[float], p: int = 1, d: int = 1, q: int = 1) -> float:
        """Simplified ARIMA implementation"""
        if len(data) < max(p, q) + d:
            return np.mean(data)
        
        # Differencing
        diff_data = data
        for _ in range(d):
            diff_data = np.diff(diff_data)
        
        if len(diff_data) < max(p, q):
            return data[-1]
        
        # Simple AR component
        ar_prediction = 0
        for i in range(1, min(p + 1, len(diff_data) + 1)):
            ar_prediction += 0.5 ** i * diff_data[-i]
        
        # Add back differences
        prediction = data[-1] + ar_prediction
        
        return max(0, prediction)
    
    def ensemble_forecast(self, df: pd.DataFrame) -> Dict:
        """Ensemble of multiple algorithms"""
        consumption_data = df['consumption'].tolist()
        
        # Get predictions from different algorithms
        predictions = {}
        
        predictions['moving_average'] = self.moving_average(consumption_data)
        predictions['exponential_smoothing'] = self.exponential_smoothing(consumption_data)
        predictions['holt_winters'] = self.holt_winters(consumption_data)
        predictions['arima'] = self.arima_simple(consumption_data)
        
        # Linear regression
        lr_result = self.linear_regression_forecast(df)
        predictions['linear_regression'] = lr_result['prediction']
        
        # Seasonal decomposition
        seasonal_result = self.seasonal_decomposition_forecast(consumption_data)
        predictions['seasonal_decomposition'] = seasonal_result['prediction']
        
        # Calculate weights based on recent performance (simplified)
        weights = {
            'moving_average': 0.15,
            'exponential_smoothing': 0.20,
            'holt_winters': 0.25,
            'linear_regression': 0.20,
            'seasonal_decomposition': 0.15,
            'arima': 0.05
        }
        
        # Ensemble prediction
        ensemble_prediction = sum(predictions[method] * weights[method] for method in predictions)
        
        # Calculate prediction intervals
        pred_values = list(predictions.values())
        prediction_std = np.std(pred_values)
        
        return {
            'ensemble_prediction': ensemble_prediction,
            'individual_predictions': predictions,
            'confidence_interval': {
                'lower': ensemble_prediction - 1.96 * prediction_std,
                'upper': ensemble_prediction + 1.96 * prediction_std
            },
            'prediction_variance': prediction_std,
            'algorithm_weights': weights,
            'linear_regression_details': lr_result,
            'seasonal_analysis': seasonal_result
        }

# Instance for use in main app
forecasting_algorithms = EnergyForecastingAlgorithms()
