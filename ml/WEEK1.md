# Week 1 — Data and features

What got built, how to re-run, and what each piece means. Keep this open during viva — it's the cheat sheet for "where does your data come from" and "what features did you use."

## What we have

Three real office buildings from the BDG2 (Building Data Genome 2) public dataset, two full years of hourly electricity meter readings (2016-01-01 to 2017-12-31), plus weather for the same site.

| Building | Mean kWh/hr | sqm |
|---|---|---|
| Hog_office_Sydney | 126 | 8,094 |
| Hog_office_Lizzie | 89 | 10,351 |
| Hog_office_Myles | 53 | 7,412 |

All three live at site `Hog`, which means one weather frame covers all of them — simpler downstream.

## How to re-run from scratch

```bash
source .venv/bin/activate
python ml/scripts/download_bdg2.py      # ~195 MB download, one-off
python ml/scripts/select_buildings.py   # picks 3 offices, writes parquet files
python ml/scripts/eda.py                # writes 6 plots to models/results/eda/
python ml/scripts/build_features.py     # writes features.parquet + split_index.parquet
```

Everything is reproducible — re-running gives identical output.

## Files produced

```
ml/data/raw/                                # raw downloads (gitignored, ~195 MB)
  metadata.csv
  weather.csv
  electricity_cleaned.csv

ml/data/processed/                          # cleaned + engineered (gitignored)
  selected_buildings.csv                    # the 3 chosen buildings + metadata
  electricity_selected.parquet              # just those 3 buildings, indexed by timestamp
  weather_selected.parquet                  # site Hog's weather
  features.parquet                          # final model-ready table
  split_index.parquet                       # timestamp -> train/val/test label

ml/models/results/eda/                      # committed plots
  consumption_distribution.png
  daily_pattern.png
  monthly_pattern.png
  sample_week.png
  weather_correlation.png
  weekly_pattern.png
```

## Feature columns (30 total in features.parquet)

The dataframe is indexed by `(building_id, timestamp)` and has one column per feature.

### Target
- `consumption` — kWh in this hour. This is what models predict.

### Lag features (consumption from the past)
- `lag_1` — consumption 1 hour ago
- `lag_24` — same hour yesterday
- `lag_168` — same hour last week

These three are usually the strongest predictors. "Today often looks like yesterday at the same hour" is captured by `lag_24`; weekly patterns by `lag_168`.

### Rolling statistics (smoothed recent history)
For each window w ∈ {3, 6, 12, 24}:
- `roll_mean_w` — average consumption over the last w hours
- `roll_std_w` — how much consumption varied over the last w hours

Important: these are built with `.shift(1).rolling(w)` so the window only sees PAST hours, never the current one. If we included the current hour we'd be cheating (lookahead).

### Time encodings
- `hour`, `dayofweek`, `month` — raw integer encodings
- `hour_sin`, `hour_cos`, `doy_sin`, `doy_cos` — cyclical sin/cos pairs

The cyclical pair matters because a model would otherwise think hour 0 (midnight) and hour 23 (11 PM) are very far apart. With sin/cos, they're neighbours on a circle — which is closer to physical reality.

### Calendar flags
- `is_weekend` — 1 on Sat/Sun, else 0
- `is_holiday` — 1 on US federal holidays (proxy — BDG2 site locations are anonymized), else 0
- `is_peak_hour` — 1 if hour ∈ [9–12] or [18–21], else 0 (matches the report's peak-hour definition)

### Weather (9 columns, joined from the same site)
- `airTemperature` (°C)
- `dewTemperature` (°C) — proxy for humidity
- `cloudCoverage` (0–8 octas)
- `windSpeed` (m/s)
- `windDirection` (degrees)
- `seaLvlPressure` (hPa)
- `precipDepth1HR`, `precipDepth6HR` (mm)

Weather had a couple of missing hours — filled with `ffill().bfill()`.

## Train / val / test split

Chronological (no shuffling — that would leak future into past):

| Split | Hours | Date range |
|---|---|---|
| Train | 12,163 | 2016-01-08 → 2017-05-28 |
| Val | 2,606 | 2017-05-28 → 2017-09-14 |
| Test | 2,607 | 2017-09-14 → 2017-12-31 |

(The first week of 2016 is missing because rows without enough lag history get dropped — the `lag_168` feature needs at least a week of warm-up.)

Roughly 70/15/15. Models train on the first 17 months, tune hyperparameters on the next 3.5 months, and report final numbers on the last 3.5 months.

## For the report (Week 6)

When you update the report, the data section needs to change from "8,760 hourly observations, mean 77.06 kWh, std 8.14 kWh" to something like:

> Three office buildings from the BDG2 public dataset, site Hog, two years of hourly meter readings (2016–2017). Mean consumption ranges from 53 to 126 kWh/hr across buildings. 30 engineered features include three lag horizons (1, 24, 168 hours), rolling mean/std at four windows, cyclical hour and day-of-year encodings, weekend / holiday / peak-hour flags, and nine weather variables joined from the same site.

## Next (Week 2)

Build the baseline models — statsmodels ARIMA / SARIMA, XGBoost, RandomForest, and a naive last-value predictor. First real metrics on the test set.
