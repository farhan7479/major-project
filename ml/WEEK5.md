# Week 5 — Real model serving

The FastAPI backend now loads the seven trained checkpoints from `ml/models/checkpoints/` at startup and serves real predictions on real building data. The old `simple_main.py` (which faked LSTM outputs by relabeling Holt-Winters and linear regression) is gone.

## Updated leaderboard (CORRECTED)

The honest numbers, with the cross-process MPS bug now fixed:

| Rank | Model | MAE | RMSE | MAPE | R² | Peak F1 |
|---|---|---|---|---|---|---|
| 1 | Random Forest | **2.64** | 4.02 | **3.51%** | **0.976** | **0.774** |
| 2 | XGBoost | 2.72 | **4.00** | 3.56% | **0.976** | 0.764 |
| 3 | Transformer | 3.83 | 5.62 | 5.29% | 0.951 | 0.697 |
| 4 | ARIMA | 5.03 | 8.49 | 7.26% | 0.894 | 0.718 |
| 5 | Naive (last) | 5.54 | 9.87 | 7.35% | 0.858 | 0.703 |
| 6 | CNN-LSTM | 6.94 | 9.03 | 9.32% | 0.877 | 0.667 |
| 7 | Naive (24h) | 7.60 | 13.67 | 10.21% | 0.648 | 0.490 |
| 8 | LSTM | 9.42 | 12.30 | 12.57% | 0.767 | 0.338 |
| 9 | BiLSTM | 12.80 | 16.06 | 17.53% | 0.602 | 0.526 |

**LSTM and BiLSTM are now WORSE than the naive baseline.** Only Transformer beats naive among the deep learning models. Tree models still dominate.

WEEK3.md's previous LSTM/BiLSTM/CNN-LSTM numbers (5.14 / 5.97 / 4.54 MAE) were artificially good because of a PyTorch MPS bug — see "The MPS bug" section below.

## What got built

### `ml/serve/` package
- **`feature_pipeline.py`** — turns a `(building_id, target_timestamp)` request into either a tabular feature row (for tree models) or a (1, 168, 17) normalized sequence (for DL models). Mirrors the training-time engineering exactly so the API row a model sees matches what it saw during training.
- **`model_registry.py`** — single class that loads all trained models at startup and exposes one `.predict(model_name, building_id, target_ts, weather_override=None)` call. Dispatches to the right backend (sklearn `.predict`, PyTorch forward, statsmodels SARIMAX `.predict`).
- **`dl_models.py`** — copies of the four neural net classes (LSTM, BiLSTM, CNN-LSTM, Transformer) so the registry can re-instantiate from saved state dicts.
- **`artifacts/`** — `context.parquet` (full hourly history per building), `scalers.json` (per-building target z-score + global feature z-score), `manifest.json` (buildings list, test period, feature column order, seq_len).

### `ml/scripts/save_inference_artifacts.py`
Materializes the three artifact files from `features.parquet` + the same scaler logic used in `torch_data._fit_scalers`. Re-run whenever features are rebuilt.

### `ml/scripts/verify_consistency.py`
Sanity check that runs after any retrain — for each model, samples 30 rows from `predictions/<model>.parquet`, calls the registry to predict the same hours, and asserts max diff < 0.5 kWh. Should always print "all consistent" after `aggregate_results.py`.

### `ml/scripts/smoke_test_api.py`
Hits every endpoint on a running uvicorn server, verifies API output matches the stored test predictions within FP tolerance. Run after starting the backend.

### `energy-forecasting-app/backend/main.py`
Brand new FastAPI app. Endpoints:
- `GET /health` — `{"status":"ok", "models_loaded": 7}`
- `GET /models` — list of model names + display names + kind (tabular / sequence / arima)
- `GET /buildings` — list of buildings + test period start/end
- `POST /forecast` — single model prediction
- `POST /batch-forecast` — multiple models in one request

Loads the registry once in the FastAPI lifespan, so per-request latency is fast (<50 ms).

## How to run the system

```bash
# 1. Make sure artifacts are up to date (only needed after features rebuild)
source .venv/bin/activate
python ml/scripts/save_inference_artifacts.py

# 2. Start the API
cd energy-forecasting-app/backend
~/dev/major-project/.venv/bin/uvicorn main:app --port 8000

# 3. (Optional) Smoke test from another shell
python ml/scripts/smoke_test_api.py
```

## Example API calls

```bash
# Health
curl http://localhost:8000/health

# Predict 2 PM consumption at Sydney with XGBoost
curl -X POST http://localhost:8000/forecast \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "xgboost",
    "building_id": "Hog_office_Sydney",
    "target_datetime": "2017-11-15T14:00:00"
  }'
# → {"model":"xgboost", "building_id":"Hog_office_Sydney",
#    "target_datetime":"2017-11-15T14:00:00", "prediction_kwh":177.4, "actual_kwh":174.56}

# Compare all 7 models on the same hour
curl -X POST http://localhost:8000/batch-forecast \
  -H 'Content-Type: application/json' \
  -d '{
    "models": ["random_forest","xgboost","arima","lstm","bilstm","cnn_lstm","transformer"],
    "building_id": "Hog_office_Sydney",
    "target_datetime": "2017-11-15T14:00:00"
  }'

# Predict with a weather override (only tabular models react to it)
curl -X POST http://localhost:8000/forecast \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "xgboost",
    "building_id": "Hog_office_Sydney",
    "target_datetime": "2017-11-15T14:00:00",
    "weather_override": {"airTemperature": 32.0}
  }'
```

## The MPS bug (worth understanding for viva)

While building model serving I noticed that the LSTM/BiLSTM/CNN-LSTM checkpoints on disk produced different predictions than what `train_model` reported during training — by 10-30 kWh. Sometimes a model with reported test MAE 5.14 actually had ~9.8 MAE when reloaded.

**Cause.** Apple's MPS (Metal Performance Shaders) backend for PyTorch's `nn.LSTM` is non-deterministic across Python processes. Two processes that load the same `.pt` file and run inference on the same input get different numerical results. The bug only affects LSTM (and modules built on it: BiLSTM, CNN-LSTM); Transformer and tree models are fine.

**How the symptom looked.**
- Training (process 1) computes test predictions on MPS → MAE 5.14 → saves to `predictions/lstm.parquet`
- API/verify (process 2) loads same `.pt`, predicts on MPS → MAE 9.8 (different)
- API/verify on CPU also → MAE 9.8 (CPU is deterministic; the MPS-trained file genuinely gives 9.8 on CPU)
- The "5.14" was a process-1-only MPS artifact

**Fix in `torch_train.py`.** After saving best state to disk, reload into a fresh model **on CPU** for the final test eval. The numbers written to `predictions.parquet` are now CPU-evaluated, deterministic across processes, and match what the API serves.

**Fix in `model_registry.py`.** Inference always runs on CPU regardless of MPS availability. ~10 ms per request — fast enough.

**Why I'm flagging this for viva.** An examiner who notices that `WEEK3.md` says LSTM MAE = 5.14 but the current leaderboard says 9.42 will ask why. The answer: a real software engineering bug, found and fixed by a verify-consistency harness that compares API output against training-time predictions on every retrain. This is the kind of thing that bites real ML systems and is part of the project's contribution.

## What this means for the report

The corrected story for the report's Results chapter:

> Random Forest and XGBoost (test MAE 2.64–2.72 kWh) clearly outperform every deep learning model on this dataset of 3 office buildings × 2 years of hourly readings. Among the deep models, only the Transformer (3.83 MAE) beats the naive last-value baseline (5.54 MAE). LSTM (9.42) and BiLSTM (12.80) fail to beat naive, indicating they are under-trained at this data scale relative to their parameter count. CNN-LSTM (6.94) sits between the two camps. These findings echo the M5 forecasting competition where gradient-boosting methods dominated deep architectures on similarly-scaled tabular forecasting.

This is more honest (and more defensible at viva) than the current draft, which claims deep models dominate.

## Status check after 5 weeks

- ✅ W1: Data + features
- ✅ W2: Classical baselines
- ✅ W3: Deep learning models (numbers since corrected)
- ✅ W4: Transformer + final figures
- ✅ W5: Live FastAPI service backed by real models, consistent with training-time metrics
- ⏳ W6: React frontend (Dashboard page, model selector matching the new model list, real numbers in the UI) + rewrite report's Results chapter with corrected leaderboard

## Next session: Week 6

Frontend rebuild + report rewrite. The frontend currently calls the (now removed) `simple_main.py` and hardcodes Holt-Winters/Linear-Regression model names. It needs:
- React Router (Forecast page + Dashboard page, matching the report's two-page claim)
- Model selector populated from `GET /models` (the 7 real models)
- Building selector populated from `GET /buildings`
- DateTime picker constrained to the test period
- Results table showing prediction + actual + which model

Then update WEEK3.md, the report's Chapter 5 (Results), and Appendix D (repo tree, tech stack) to match what we actually built.
