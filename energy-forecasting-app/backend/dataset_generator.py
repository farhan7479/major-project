import pandas as pd
import numpy as np
import datetime
from typing import List, Dict

class EnergyDatasetGenerator:
    """Generate realistic energy consumption datasets based on real patterns"""
    
    def __init__(self):
        self.seasonal_patterns = {
            'winter': {'base': 100, 'amplitude': 40, 'peak_hours': [7, 8, 18, 19, 20]},
            'spring': {'base': 70, 'amplitude': 25, 'peak_hours': [7, 8, 18, 19]},
            'summer': {'base': 85, 'amplitude': 35, 'peak_hours': [12, 13, 14, 15, 16]},
            'autumn': {'base': 75, 'amplitude': 30, 'peak_hours': [7, 8, 18, 19]}
        }
    
    def get_season(self, month: int) -> str:
        """Determine season based on month"""
        if month in [12, 1, 2]:
            return 'winter'
        elif month in [3, 4, 5]:
            return 'spring'
        elif month in [6, 7, 8]:
            return 'summer'
        else:
            return 'autumn'
    
    def generate_hourly_data(self, start_date: str, days: int = 30) -> pd.DataFrame:
        """Generate hourly energy consumption data"""
        start = datetime.datetime.strptime(start_date, '%Y-%m-%d')
        dates = [start + datetime.timedelta(hours=i) for i in range(days * 24)]
        
        data = []
        for dt in dates:
            season = self.get_season(dt.month)
            pattern = self.seasonal_patterns[season]
            
            # Base consumption
            consumption = pattern['base']
            
            # Daily pattern
            if dt.hour in pattern['peak_hours']:
                consumption += pattern['amplitude'] * 0.8
            elif dt.hour in [22, 23, 0, 1, 2, 3, 4, 5]:
                consumption *= 0.6  # Night reduction
            
            # Weekly pattern
            if dt.weekday() >= 5:  # Weekend
                consumption *= 0.75
            
            # Add random noise
            consumption += np.random.normal(0, consumption * 0.1)
            
            # Weather effect simulation
            if season == 'summer' and dt.hour in [12, 13, 14, 15]:
                consumption += np.random.normal(20, 5)  # AC usage
            elif season == 'winter' and dt.hour in [6, 7, 8, 17, 18, 19]:
                consumption += np.random.normal(15, 3)  # Heating
            
            data.append({
                'timestamp': dt,
                'consumption': max(0, consumption),
                'hour': dt.hour,
                'day_of_week': dt.weekday(),
                'day_of_year': dt.timetuple().tm_yday,
                'month': dt.month,
                'season': season,
                'temperature': self.simulate_temperature(dt),
                'humidity': np.random.uniform(30, 80),
                'is_holiday': self.is_holiday(dt)
            })
        
        return pd.DataFrame(data)
    
    def simulate_temperature(self, dt: datetime.datetime) -> float:
        """Simulate temperature based on season and time"""
        season_temps = {
            'winter': {'base': 5, 'range': 10},
            'spring': {'base': 15, 'range': 8},
            'summer': {'base': 25, 'range': 10},
            'autumn': {'base': 12, 'range': 8}
        }
        season = self.get_season(dt.month)
        temp_config = season_temps[season]
        
        # Daily temperature variation
        daily_variation = 5 * np.sin(2 * np.pi * (dt.hour - 6) / 24)
        base_temp = temp_config['base'] + np.random.uniform(-temp_config['range']/2, temp_config['range']/2)
        
        return base_temp + daily_variation
    
    def is_holiday(self, dt: datetime.datetime) -> bool:
        """Simple holiday detection"""
        # Major holidays (simplified)
        holidays = [
            (1, 1),   # New Year
            (7, 4),   # Independence Day
            (12, 25), # Christmas
        ]
        return (dt.month, dt.day) in holidays
    
    def generate_multiple_algorithms_data(self, hours: int = 168) -> Dict:
        """Generate data suitable for multiple forecasting algorithms"""
        end_date = datetime.datetime.now()
        start_date = end_date - datetime.timedelta(hours=hours)
        
        df = self.generate_hourly_data(start_date.strftime('%Y-%m-%d'), days=hours//24 + 1)
        df = df.head(hours)  # Ensure exact hours
        
        # Prepare data for different algorithms
        return {
            'historical_data': df.to_dict('records'),
            'statistics': {
                'mean_consumption': df['consumption'].mean(),
                'std_consumption': df['consumption'].std(),
                'min_consumption': df['consumption'].min(),
                'max_consumption': df['consumption'].max(),
                'peak_hours': df.groupby('hour')['consumption'].mean().nlargest(5).index.tolist(),
                'seasonal_averages': df.groupby('season')['consumption'].mean().to_dict()
            },
            'correlations': {
                'temperature_correlation': df['consumption'].corr(df['temperature']),
                'humidity_correlation': df['consumption'].corr(df['humidity'])
            }
        }

# Instance for use in main app
dataset_generator = EnergyDatasetGenerator()
