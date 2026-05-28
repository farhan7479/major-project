# Week 3 — Deep learning models

Three PyTorch architectures trained on the same data and graded by the same metrics as Week 2.

## Final leaderboard (test set, macro across 3 buildings)

| Rank | Model | MAE | RMSE | MAPE | R² | Peak F1 |
|---|---|---|---|---|---|---|
| 1 | Random Forest | **2.64** | 4.02 | **3.51%** | **0.976** | **0.774** |
| 2 | XGBoost | 2.72 | **4.00** | 3.56% | **0.976** | 0.764 |
| 3 | **CNN-LSTM** | **4.54** | 6.46 | 6.21% | 0.936 | 0.685 |
| 4 | ARIMA(2,1,2) | 5.03 | 8.49 | 7.26% | 0.894 | 0.718 |
| 5 | **LSTM** | 5.14 | 7.27 | 6.80% | 0.920 | 0.600 |
| 6 | Naive (last-value) | 5.54 | 9.87 | 7.35% | 0.858 | 0.703 |
| 7 | **BiLSTM** | 5.97 | 8.29 | 7.94% | 0.895 | 0.626 |
| 8 | Naive (24h-ago) | 7.60 | 13.67 | 10.21% | 0.648 | 0.490 |

**Bold rows are the new Week-3 additions.**

## How to re-run

```bash
source .venv/bin/activate
cd ml/scripts
python train_lstm.py        # ~7 min on M-series MPS
python train_bilstm.py      # ~5 min
python train_cnn_lstm.py    # ~5 min
python aggregate_results.py # refresh leaderboard.csv + plots
```

## What we built

### Shared PyTorch infrastructure (`torch_data.py` + `torch_train.py`)
- **Windowing.** For each `(building, hour)` we build a 168-step sequence of per-hour features ending the hour before the prediction target.
- **Normalization.** Per-building z-score on the target (so a single model can fit Sydney's 126 kWh/hr and Myles's 53 kWh/hr together), global z-score on the weather columns. Stats are fit on the train split only.
- **Sequence input.** 17 features per timestep: previous consumption, six weather columns, four cyclical hour/day-of-year encodings, three calendar flags, three one-hot building dummies.
- **Training loop.** Adam (lr 1e-3, weight_decay 1e-5), MSE loss in z-score space, ReduceLROnPlateau on val MAE, early stopping after 6 epochs without improvement, gradient clipping at norm 1.0. Runs on Apple MPS automatically.

### Three model architectures

| Model | Architecture | Params |
|---|---|---|
| LSTM | 2 LSTM layers (hidden 64) → linear head | ~36k |
| BiLSTM | 2 bidirectional LSTM layers (hidden 64 each direction) → linear head | ~72k |
| CNN-LSTM | 2 Conv1d layers (64 channels, kernel 3) → MaxPool → 2 LSTM layers → linear head | ~58k |

## What the numbers tell us

### CNN-LSTM is the best deep learning model (4.54 MAE)
The Conv1d front-end picks up short-range patterns (sharp ramps, hour-to-hour deltas) before feeding the LSTM, which then focuses on longer-range memory. The combination beats vanilla LSTM by ~12% and beats ARIMA by ~10%.

### Vanilla LSTM (5.14 MAE) ≈ ARIMA (5.03 MAE)
The LSTM is in the same ballpark as ARIMA. This is honest and defensible: 12k training hours per building is small for deep learning, and ARIMA's explicit AR/MA structure is well-suited to smooth seasonal-AR-like consumption.

### BiLSTM (5.97 MAE) actually does *worse* than LSTM
Counterintuitive but real:
1. **Bidirectional doesn't help at inference.** At prediction time, the future isn't available. The "backward" pass only sees the past sequence from a different direction — no new information.
2. **Double the parameters → more overfit risk.** Early-stopped at epoch 9 vs LSTM's 29 — the model was already over-fitting validation despite ~72k params.

This is a useful finding to discuss at viva: "we tested BiLSTM and it didn't help — bidirectionality is for tasks where future context exists (sentence classification, biomedical sequences), not forecasting."

### Tree models still win
RF and XGBoost stay on top with MAE 2.64/2.72. Why deep learning didn't catch up:
- **Data size.** 12k hours per building is small. Tree models need less data to find sharp piecewise rules.
- **Engineered features.** Tree models eat `lag_24`, `lag_168`, `roll_mean_24` directly. The LSTM has to discover these summaries from raw sequences — which it can do, but needs more data.
- **Categorical splits.** Tree models split sharply on `is_weekend`, `is_holiday`, `hour`. LSTMs have to learn smooth approximations.

This is the standard finding on tabular-ish forecasting problems with <100k samples. Deep learning starts to win clearly when you have hundreds of thousands of samples or hundreds of related series.

## Files produced this week

```
ml/models/checkpoints/
  lstm.pt
  bilstm.pt
  cnn_lstm.pt

ml/models/results/
  metrics/{lstm,bilstm,cnn_lstm}.json
  predictions/{lstm,bilstm,cnn_lstm}.parquet
  actual_vs_predicted/{lstm,bilstm,cnn_lstm}__Hog_office_*.png   (9 new plots)
  leaderboard.csv                                                 (refreshed)
  leaderboard_mae.png                                             (refreshed)
```

## Things to look at

- `ml/models/results/leaderboard_mae.png` — visual comparison of all 8 models
- `ml/models/results/actual_vs_predicted/cnn_lstm__Hog_office_Sydney.png` — see how the best DL model tracks reality on one week of test data
- `ml/models/results/actual_vs_predicted/bilstm__Hog_office_Sydney.png` — compare with the worst DL model for the same building

## Things to defend at viva

**"Why didn't your LSTM beat your XGBoost?"**
> The dataset is small (~12k training hours per building × 3 buildings = 36k sequences). XGBoost on engineered lag/rolling features is the strong baseline at this scale. Literature shows deep models pull ahead at hundreds of thousands of samples or with many related series, neither of which applies here. We report both honestly.

**"Why test BiLSTM if it doesn't make sense for forecasting?"**
> Because the report's architecture survey listed it, and it's worth showing empirically that bidirectionality doesn't help here — that's a useful negative result.

**"What about the CNN-LSTM beating plain LSTM?"**
> The Conv1d front-end captures local patterns (e.g. sharp ramp at 8 AM) more cheaply than a recurrent layer would, freeing the LSTM to model longer dependencies. ~12% MAE reduction over plain LSTM, consistent with results reported in the hybrid-architecture literature cited in the report.

## What the report needs to change (for Week 6)

Current report claims (Chapter 5):
- BiLSTM 6.37 MAE, 8.25% MAPE → real: 5.97 MAE, 7.94% MAPE
- LSTM 6.42 MAE, 8.22% MAPE → real: 5.14 MAE, 6.80% MAPE
- ARIMA wins with 6.33 MAE → real: ARIMA is rank 4, CNN-LSTM beats it

Plus the report's conclusion says "BiLSTM achieving 1.98% MAPE" — that's contradicted by every model we trained. We'll rewrite the conclusion in Week 6.

## Next (Week 4)

Temporal Fusion Transformer (TFT) via `pytorch-forecasting`. It's the most modern architecture in the report and the one we're most likely to drop if it gives trouble — we'll budget one session for it and time-box. Plus final eval: peak-detection plots, per-building breakdown, generate the figures the report's RESULTS chapter references.
