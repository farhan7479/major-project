# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Repository overview
This repo contains two mostly independent projects:

- `energy-forecasting-app/`: the primary full-stack app (FastAPI backend + React/Vite frontend) that serves forecasts and visualizes them.
- `energy_consumption_prediction-master/`: a notebook-based prototype/training project (Jupyter) for LSTM/GRU energy consumption prediction.

## Common commands
Most day-to-day development happens inside `energy-forecasting-app/`.

### Full-stack (recommended)
From `energy-forecasting-app/`:
- Start both backend and frontend (bootstraps venv + installs deps as needed):
  - `npm run start`
  - or `./start.sh`

### Backend (FastAPI)
From `energy-forecasting-app/backend/`:
- Create/activate venv:
  - `python3 -m venv venv`
  - `source venv/bin/activate`
- Install deps:
  - `pip install -r requirements.txt`
- Run API server:
  - `python main.py`

Notes:
- There is also `simple_main.py`, which returns richer responses (statistics/correlations and an ensemble payload) used by the current frontend UI.
  - If you run `simple_main.py`, you may need to add missing Python deps (for example `scipy`) since `backend/forecasting_algorithms.py` imports it but it is not listed in `backend/requirements.txt`.

### Frontend (React + Vite)
From `energy-forecasting-app/frontend/`:
- Install deps:
  - `npm install`
- Dev server:
  - `npm run dev`
- Lint:
  - `npm run lint`
- Production build:
  - `npm run build`
- Preview production build:
  - `npm run preview`

### Tests
This repo does not currently define an automated test runner (no `pytest` test suite and no frontend unit test scripts like Jest/Vitest).

## Architecture notes (big picture)
### `energy-forecasting-app/` runtime flow
- Frontend runs on Vite (default `5173`) and calls the backend on `http://localhost:8000`.
  - API client: `energy-forecasting-app/frontend/src/services/api.js`
- Backend is a FastAPI app exposing:
  - `GET /sample-data`: produces the time-series input the UI charts
  - `POST /predict`: produces one-step-ahead forecast results
  - `GET /model-info`: exposes model/config metadata
  - Entrypoints:
    - `energy-forecasting-app/backend/main.py`: uses a PyTorch-based `EnergyForecastingService` from `backend/models.py`.
    - `energy-forecasting-app/backend/simple_main.py`: uses `dataset_generator.py` + `forecasting_algorithms.py` to produce an “ensemble” response and additional analytics.

### Backend modules
- `energy-forecasting-app/backend/models.py`
  - Defines `GRUNet` and `LSTMNet` PyTorch modules.
  - Defines `EnergyForecastingService` (global `forecasting_service`) used by `backend/main.py` for preprocessing and inference.
- `energy-forecasting-app/backend/forecasting_algorithms.py`
  - Implements a small suite of classical forecasting approaches (moving average, exponential smoothing, Holt-Winters, simplified ARIMA, linear regression, seasonal decomposition) and an `ensemble_forecast` aggregator.
- `energy-forecasting-app/backend/dataset_generator.py`
  - Generates synthetic but “realistic” hourly consumption data with seasonality, weather effects, and basic holiday flags.

### Frontend modules
- `energy-forecasting-app/frontend/src/App.jsx` is the page-level orchestrator:
  - loads sample data via `api.getSampleData(hours)`
  - triggers predictions via `api.predictEnergyConsumption(data, modelType)`
  - conditionally renders the chart components based on response shape
- UI visualization components (Chart.js):
  - `EnergyChart.jsx`: historical series + point predictions
  - `AlgorithmComparisonChart.jsx`: bar chart of per-algorithm predictions
  - `ConfidenceIntervalChart.jsx`: last-24-hours + next-hour prediction band
  - `SeasonalPatternChart.jsx`: hourly/weekly patterns + summary stats

## Notebook prototype project
`energy_consumption_prediction-master/` is centered around the Jupyter notebook `Energy consumption prediction using LSTM-GRU in PyTorch.ipynb` and its `requirements.txt`.
- Typical workflow: install deps into a Python env and run `jupyter notebook` from that directory.
