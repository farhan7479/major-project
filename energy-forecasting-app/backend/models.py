import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import MinMaxScaler

class GRUNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, drop_prob=0.2):
        super(GRUNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.gru = nn.GRU(
            input_dim, hidden_dim, n_layers, batch_first=True, dropout=drop_prob
        )
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x, h):
        out, h = self.gru(x, h)
        out = self.fc(self.relu(out[:, -1]))
        return out, h

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = weight.new(self.n_layers, batch_size, self.hidden_dim).zero_()
        return hidden

class LSTMNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, drop_prob=0.2):
        super(LSTMNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.lstm = nn.LSTM(
            input_dim, hidden_dim, n_layers, batch_first=True, dropout=drop_prob
        )
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x, h):
        out, h = self.lstm(x, h)
        out = self.fc(self.relu(out[:, -1]))
        return out, h

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = (
            weight.new(self.n_layers, batch_size, self.hidden_dim).zero_(),
            weight.new(self.n_layers, batch_size, self.hidden_dim).zero_(),
        )
        return hidden

class EnergyForecastingService:
    def __init__(self):
        self.input_dim = 5
        self.hidden_dim = 256
        self.output_dim = 1
        self.n_layers = 2
        self.window_size = 90
        
        # Initialize models
        self.gru_model = GRUNet(self.input_dim, self.hidden_dim, self.output_dim, self.n_layers)
        self.lstm_model = LSTMNet(self.input_dim, self.hidden_dim, self.output_dim, self.n_layers)
        
        # Initialize scalers
        self.scaler = MinMaxScaler()
        self.label_scaler = MinMaxScaler()
        
    def load_models(self, gru_path=None, lstm_path=None):
        """Load trained models from file paths"""
        try:
            if gru_path:
                self.gru_model.load_state_dict(torch.load(gru_path, map_location='cpu'))
                self.gru_model.eval()
            if lstm_path:
                self.lstm_model.load_state_dict(torch.load(lstm_path, map_location='cpu'))
                self.lstm_model.eval()
        except Exception as e:
            print(f"Error loading models: {e}")
    
    def preprocess_data(self, data):
        """Preprocess input data for prediction"""
        # Assuming data is a list of dictionaries with features
        # Convert to numpy array and scale
        if isinstance(data, list):
            data = np.array(data)
        
        # Scale the data
        scaled_data = self.scaler.fit_transform(data.reshape(-1, self.input_dim))
        
        # Create sequences for prediction
        if len(scaled_data) >= self.window_size:
            sequence = scaled_data[-self.window_size:].reshape(1, self.window_size, self.input_dim)
            return torch.FloatTensor(sequence)
        else:
            raise ValueError(f"Need at least {self.window_size} data points for prediction")
    
    def predict_gru(self, input_data):
        """Make prediction using GRU model"""
        with torch.no_grad():
            h = self.gru_model.init_hidden(1)
            output, _ = self.gru_model(input_data, h)
            return output.item()
    
    def predict_lstm(self, input_data):
        """Make prediction using LSTM model"""
        with torch.no_grad():
            h = self.lstm_model.init_hidden(1)
            output, _ = self.lstm_model(input_data, h)
            return output.item()
    
    def generate_sample_data(self, hours=168):
        """Generate sample data for demonstration"""
        np.random.seed(42)
        
        sample_data = []
        for i in range(hours):
            # Simulate energy consumption with daily and weekly patterns
            hour_of_day = i % 24
            day_of_week = (i // 24) % 7
            
            # Base consumption with daily pattern
            base_consumption = 50 + 30 * np.sin(2 * np.pi * hour_of_day / 24) + np.random.normal(0, 5)
            
            # Weekend adjustment
            if day_of_week >= 5:  # Weekend
                base_consumption *= 0.8
            
            sample_data.append({
                'consumption': max(0, base_consumption),
                'hour': hour_of_day,
                'dayofweek': day_of_week,
                'month': 6,  # June
                'dayofyear': 150 + i // 24
            })
        
        return sample_data

# Global service instance
forecasting_service = EnergyForecastingService()
