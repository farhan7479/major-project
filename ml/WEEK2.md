# Week 2 — Baselines and tree-based models

What got built, real numbers, and what they mean for the report.

## Leaderboard (test set, macro across 3 buildings)

| Model | MAE | RMSE | MAPE | R² | Peak F1 |
|---|---|---|---|---|---|
| Random Forest | **2.64** | 4.02 | **3.51%** | **0.976** | **0.777** |
| XGBoost | 2.70 | **3.99** | 3.54% | **0.976** | 0.769 |
| ARIMA(2,1,2) | 5.03 | 8.49 | 7.26% | 0.894 | 0.718 |
| Naive (last value) | 5.54 | 9.87 | 7.35% | 0.858 | 0.703 |
| Naive (24h-ago) | 7.60 | 13.67 | 10.21% | 0.648 | 0.490 |

**Headline:** Random Forest and XGBoost cut the error roughly in half vs. ARIMA. This is the opposite of what the current report says (which claims ARIMA wins).

## How to re-run

```bash
source .venv/bin/activate
cd ml/scripts
python train_naive.py        # 2 sec
python train_arima.py        # ~30 sec (fits 3 SARIMAX models)
python train_xgboost.py      # ~30 sec (early-stopped at iter 1151)
python train_rf.py           # ~1 min (300 trees, 12k rows)
python aggregate_results.py  # produces leaderboard.csv + plots
```

Total time end-to-end: ~3 minutes on M-series Mac.

## Files produced

```
ml/models/checkpoints/
  arima_Hog_office_Lizzie.pkl    # one per building
  arima_Hog_office_Myles.pkl
  arima_Hog_office_Sydney.pkl
  random_forest.pkl
  xgboost.pkl

ml/models/results/
  leaderboard.csv
  leaderboard_mae.png
  metrics/<model>.json               # per-model macro + per-building numbers
  predictions/<model>.parquet        # all test-set (y_true, y_pred) pairs
  feature_importance/xgboost.png     # top 20 features
  feature_importance/random_forest.png
  actual_vs_predicted/<model>__<building>.png   # 1 week zoom-in plots
```

## What each metric tells you (for viva)

- **MAE** (Mean Absolute Error, kWh) — average size of the prediction error. The most interpretable metric.
- **RMSE** (Root Mean Squared Error) — penalizes large errors more. If RMSE >> MAE, the model occasionally makes big mistakes.
- **MAPE** (Mean Absolute Percentage Error) — error as % of actual. Easier to compare across buildings with different scales.
- **R²** — fraction of variance the model explains. 1.0 = perfect, 0 = no better than predicting the mean.
- **Peak F1** — does the model correctly flag the top-5% busiest hours? 1.0 = perfect peak detection.

## Why ARIMA underperforms here

The report claims ARIMA wins because of "small dataset + clear seasonality." On our real BDG2 data, ARIMA only beats naive by a small margin (5.03 vs 5.54 MAE) because:

1. **ARIMA can't use weather.** Tree models see temperature, humidity, wind — strong consumption drivers. ARIMA only sees past consumption.
2. **ARIMA can't use calendar features.** No `is_weekend`, no `is_holiday`, no `is_peak_hour`.
3. **ARIMA can't share information across buildings.** Each building gets its own model with no transfer; tree models train on all 3 jointly.

A SARIMAX with exogenous weather variables would close some of the gap, but tree-based models on engineered features are still the more practical winner.

## What the feature importance plots show

Open `ml/models/results/feature_importance/*.png`:
- `lag_1` (consumption 1 hour ago) is by far the most important feature
- `lag_24` and `lag_168` matter too — daily and weekly patterns
- Building dummies (`bld_*`) appear because the 3 buildings have very different scales
- Weather features (`airTemperature`, `dewTemperature`) make the top 20

This validates the feature engineering choices from Week 1.

## What the report needs to change

Current report (Chapter 5 — RESULTS) shows:

| Model | MAE | RMSE | MAPE | R² |
|---|---|---|---|---|
| ARIMA(2,1,2) | **6.33** | **7.79** | 8.22 | -0.00 |
| BiLSTM | 6.37 | 7.85 | 8.25 | -0.02 |
| LSTM | 6.42 | 8.07 | 8.22 | -0.08 |
| Naive | 9.00 | 10.73 | 11.71 | -0.90 |

These are the placeholder numbers and they're wrong in several ways:
- All R² are negative (impossible if models beat naive — our real R² is 0.85–0.98)
- ARIMA shown as winner — actually tree models win by ~50%
- No XGBoost/RandomForest in the table at all

We'll rewrite this whole section in Week 6 with the real numbers, plus add the deep-learning rows after Week 3–4.

## Subtleties worth understanding

**Why "macro" averaging?** We have 3 buildings with very different scales (Sydney ~126 kWh, Myles ~53). Pooling raw errors would let the largest building dominate. Macro averaging gives each building equal weight in the headline number.

**Why one-hot the building_id?** Without it, the tree models would have to learn each building's baseline level from scratch using only lag features. The dummy variable lets them learn "Sydney's typical level is X" once and reuse it.

**Why early-stopping for XGBoost?** Without it the model would overfit — keep adding trees that improve train accuracy but hurt val accuracy. Stopping at the best val iteration (1151) gives the true generalization optimum.

## Next (Week 3)

Deep learning models in PyTorch — LSTM, BiLSTM, CNN-LSTM. Same evaluation pipeline, so the leaderboard just gains 3 more rows. The bar to beat is XGBoost/RF at 2.64–2.70 MAE.
