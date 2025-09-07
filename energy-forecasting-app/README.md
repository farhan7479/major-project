# Energy Forecasting Application

A full-stack web application for energy consumption forecasting using LSTM and GRU neural networks.

## Architecture

- **Backend**: FastAPI with PyTorch models
- **Frontend**: React + Vite + Tailwind CSS + Chart.js
- **Models**: LSTM and GRU networks for time series prediction

## Features

- 🔮 Real-time energy consumption forecasting
- 📊 Interactive charts with Chart.js
- 🎛️ Model selection (LSTM, GRU, or both)
- 📱 Responsive design with Tailwind CSS
- ⚡ Fast API with automatic documentation
- 🔄 Real-time predictions and data visualization

## Quick Start

### Backend Setup

1. Navigate to the backend directory:
```bash
cd backend
```

2. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Start the FastAPI server:
```bash
python main.py
```

The API will be available at `http://localhost:8000`
API Documentation: `http://localhost:8000/docs`

### Frontend Setup

1. Navigate to the frontend directory:
```bash
cd frontend
```

2. Install dependencies:
```bash
npm install
```

3. Start the development server:
```bash
npm run dev
```

The application will be available at `http://localhost:5173`

## API Endpoints

- `GET /` - API status
- `GET /health` - Health check
- `GET /sample-data?hours=168` - Get sample energy data
- `POST /predict` - Make energy consumption predictions
- `GET /model-info` - Get model information

## Usage

1. **Load Data**: The app automatically loads sample energy consumption data
2. **Configure**: Select data range (24h, 1 week, 1 month) and model type
3. **Predict**: Click "Make Prediction" to forecast next hour consumption
4. **Visualize**: View results on interactive charts with historical data and predictions

## Model Details

- **Input Features**: consumption, hour, day of week, month, day of year
- **Window Size**: 90 time steps (hours)
- **Architecture**: 2-layer LSTM/GRU with 256 hidden units
- **Output**: Next hour energy consumption prediction

## Development

### Adding New Models

1. Implement the model class in `backend/models.py`
2. Add prediction method to `EnergyForecastingService`
3. Update API endpoints in `main.py`
4. Update frontend components for new model type

### Customizing UI

- Modify components in `frontend/src/components/`
- Update styling with Tailwind CSS classes
- Add new charts or visualizations

## Technologies Used

### Backend
- FastAPI
- PyTorch
- NumPy
- Pandas
- Scikit-learn
- Uvicorn

### Frontend
- React 18
- Vite
- Tailwind CSS
- Chart.js / React Chart.js 2
- Axios

## License

MIT License
