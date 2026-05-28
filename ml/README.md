# ML pipeline

Training code for the load forecasting models that the FastAPI backend serves.

## Layout

```
ml/
  data/
    raw/         # BDG2 dumps as downloaded
    processed/   # engineered features, train/val/test splits
  notebooks/     # EDA and one-off exploration
  scripts/       # reproducible pipeline (feature build, training, eval)
  models/
    checkpoints/ # trained model weights
    results/     # metrics tables, plots
```

## Setup

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r ml/requirements.txt
```

Deep-learning deps (`torch`, `pytorch-forecasting`, `xgboost`, `statsmodels`) are added in later weeks as the corresponding models come online — keeps the cold-install fast.

## Pipeline

1. `scripts/download_bdg2.py` — fetch BDG2 release CSVs to `data/raw/`
2. `scripts/select_buildings.py` — choose 1–3 buildings with low missingness
3. `scripts/build_features.py` — lag/rolling/cyclical/weather features → `data/processed/features.parquet`
4. `scripts/train_<model>.py` — one per model (arima, xgboost, lstm, bilstm, cnn_lstm, tft)
5. `scripts/evaluate.py` — produce results table + plots in `models/results/`
