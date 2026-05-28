# Week 6 — Frontend + report rewrite

The React app is rebuilt around the new FastAPI service, and the report's Results / Conclusion / Tech-stack / Appendix sections now match what the code actually does.

## Frontend

Two-page React app, replacing the old single-page demo.

```
energy-forecasting-app/frontend/src/
  App.jsx                  router + nav, redirects / → /forecast
  index.css                Tailwind imports + light global styles
  services/api.js          axios client for the 6 backend endpoints
  pages/
    Forecast.jsx           building/datetime/model selector, single + compare modes
    Dashboard.jsx          leaderboard table + MAE bar chart + model cards
```

### Forecast page

- Loads `/buildings` and `/models` on mount.
- Drop-downs for building and (when in single-model mode) model.
- Native HTML5 `datetime-local` picker constrained to the test window (2017-09-14 → 2017-12-31).
- Two modes:
  - **Single model** → calls `POST /forecast`, renders a card with predicted kWh, actual kWh, and signed error coloured green/amber/red by magnitude.
  - **Compare all models** → calls `POST /batch-forecast` with every model name, renders a sortable table of all predictions and their errors.

### Dashboard page

- Loads `/metrics` and `/models` on mount.
- Three sections:
  - **Comparison chart** — Chart.js horizontal bar of test-set MAE per model.
  - **Leaderboard** — sortable table of MAE / RMSE / MAPE / R² / Peak F1 with the winner of each column marked `★` and bold green.
  - **Available models** — coloured cards showing display name, kind (`tabular` / `sequence` / `arima`), and registry name.

### API client (`services/api.js`)

Single object exporting:
- `api.health()`
- `api.getModels()` → `[{name, display_name, kind}, …]`
- `api.getBuildings()` → `{buildings, testPeriod}`
- `api.getMetrics()` → leaderboard rows
- `api.forecast({model, buildingId, targetDatetime, weatherOverride?})` → `{prediction_kwh, actual_kwh, …}`
- `api.batchForecast({models, buildingId, targetDatetime, weatherOverride?})` → `{predictions: {model: kwh}, actual_kwh, …}`

## Backend addition

Added one endpoint:

```
GET /metrics → {"leaderboard": [{model, mae, rmse, mape, r2, peak_f1, …}, …]}
```

reading `ml/models/results/leaderboard.csv`. No other backend changes — the existing `/health`, `/models`, `/buildings`, `/forecast`, `/batch-forecast` from Week 5 are unchanged.

## Running the full stack

Note: Vite 7 needs Node ≥ 20. The repo has `nvm` installed with v20.20.2 and v23.11.0; on a fresh shell select one explicitly:

```bash
# Terminal 1 — backend
source ~/dev/major-project/.venv/bin/activate
cd ~/dev/major-project/energy-forecasting-app/backend
uvicorn main:app --port 8000

# Terminal 2 — frontend
export PATH="$HOME/.nvm/versions/node/v20.20.2/bin:$PATH"   # or v23.11.0
cd ~/dev/major-project/energy-forecasting-app/frontend
npm run dev   # http://localhost:5173
```

Open `http://localhost:5173/`. You'll be redirected to `/forecast`. Both pages talk to `http://localhost:8000` via CORS-enabled axios calls.

## Report rewrite

Edits to `project.tex`:

| Section | Before | After |
|---|---|---|
| Sim Parameters (Ch 4) | 8,760 hours, 5 features, single year | BDG2 3 buildings, 17,544 rows/building, 30 features, 2 years, chronological 70/15/15 |
| Tech Stack (Ch 4) | "TensorFlow 2.x", MLflow, Docker | PyTorch 2.5, XGBoost 2.1, sklearn 1.6, statsmodels 0.14, FastAPI + Uvicorn, React 19 + Vite 7 + Tailwind + Chart.js. Notes the MPS-CPU inference choice. |
| Results table (Ch 5) | 6 models with tiny placeholder MAE values (0.25–0.34) | 9-row leaderboard with real kWh values (RF 2.64 best, BiLSTM 12.80 worst) and corrected discussion |
| Analysis (Ch 5) | "BiLSTM achieves best overall MAPE (1.98%)" | Tree models dominate, only Transformer beats naive among DL, LSTM/BiLSTM worse than naive, references M4/M5 |
| Conclusion (Ch 6) | "Deep learning consistently outperforms classical … BiLSTM 1.98% MAPE" + "validates literature claims on deep learning superiority" | Tree models win, only Transformer beats naive, M4/M5-aligned framing |
| Appendix C | BiLSTM feature ablation (all numbers ~0.5 kWh range, wrong) | Per-building MAE breakdown table for all 9 models |
| Appendix D | (missing in repo version) | Full repo layout + run instructions for training pipeline and live demo |

## Status check — all 6 weeks complete

- ✅ W1 — Data + feature pipeline
- ✅ W2 — Classical baselines (ARIMA / RF / XGBoost / naive)
- ✅ W3 — Deep learning (LSTM / BiLSTM / CNN-LSTM) [numbers later corrected when the MPS bug was caught in W5]
- ✅ W4 — Transformer encoder + per-building heatmap + headline figure
- ✅ W5 — Real FastAPI serving 7 models, CPU inference for cross-process determinism
- ✅ W6 — Two-page React app + report aligned with code

## What's still worth doing before viva

These are optional polish items, not blockers:

1. **Screenshot the live UI** for the report's Implementation chapter (figures 1–5 currently reference `1.png` … `5.png` which are generic placeholders). Replace with:
   - Forecast page on single-model mode with a prediction card
   - Forecast page on compare-all mode with the model table
   - Dashboard page with the leaderboard chart
   - One actual-vs-predicted plot (`ml/models/results/actual_vs_predicted/random_forest__Hog_office_Sydney.png`)
   - The 14-day showcase (`ml/models/results/showcase_top3_models.png`)

2. **Add EDA figures to the report.** `ml/models/results/eda/*.png` has 6 plots from Week 1 (daily pattern, weekly pattern, weather correlation, etc.) that would slot naturally into the Data Analysis chapter.

3. **Commit and push.** The repo is currently dirty (the entire `ml/` tree, the new `backend/main.py`, the new frontend pages, the report edits). One commit per week or one squashed PR works either way for viva.

4. **Practice the viva story.** Three rehearsed lines:
   - *Why does tree win?* "M4/M5 competition outcomes — gradient boosting on engineered features wins at 10⁴-sample scale; we confirm empirically on BDG2."
   - *Why did the W3 numbers change in W5?* "Caught a PyTorch MPS cross-process non-determinism bug for nn.LSTM via a verify-consistency harness. Switched inference to CPU; numbers are now reproducible and honest."
   - *What's the system's contribution?* "End-to-end honest benchmark plus a deployable FastAPI + React demo that lets you click any (model × building × hour) combination and see real predictions vs ground truth — built and tested with an automated consistency check between training metrics and live API output."

Project is in a defensible, shippable state.
