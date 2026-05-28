"""Build the model-ready feature table from the selected buildings + weather.

Output:
  data/processed/features.parquet      (one row per (building, hour) with all features)
  data/processed/split_index.parquet   (timestamp -> split label: train/val/test)

Lag and rolling features are constructed with .shift(1) before .rolling() so the
window never sees the target hour itself — no lookahead bias.
"""

from __future__ import annotations

import sys
from pathlib import Path

import holidays
import numpy as np
import pandas as pd

PROCESSED = Path(__file__).resolve().parents[1] / "data" / "processed"

LAGS = (1, 24, 168)
ROLL_WINDOWS = (3, 6, 12, 24)
PEAK_HOURS = set(range(9, 13)) | set(range(18, 22))
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15


def add_lags(g: pd.DataFrame) -> pd.DataFrame:
    for k in LAGS:
        g[f"lag_{k}"] = g["consumption"].shift(k)
    return g


def add_rolling(g: pd.DataFrame) -> pd.DataFrame:
    past = g["consumption"].shift(1)
    for w in ROLL_WINDOWS:
        g[f"roll_mean_{w}"] = past.rolling(w).mean()
        g[f"roll_std_{w}"] = past.rolling(w).std()
    return g


def add_calendar(g: pd.DataFrame, us_holidays: holidays.HolidayBase) -> pd.DataFrame:
    ts = g.index.get_level_values("timestamp")
    g["hour"] = ts.hour
    g["dayofweek"] = ts.dayofweek
    g["month"] = ts.month
    g["hour_sin"] = np.sin(2 * np.pi * ts.hour / 24)
    g["hour_cos"] = np.cos(2 * np.pi * ts.hour / 24)
    g["doy_sin"] = np.sin(2 * np.pi * ts.dayofyear / 365.25)
    g["doy_cos"] = np.cos(2 * np.pi * ts.dayofyear / 365.25)
    g["is_weekend"] = (ts.dayofweek >= 5).astype("int8")
    g["is_peak_hour"] = ts.hour.isin(PEAK_HOURS).astype("int8")
    g["is_holiday"] = pd.Index(ts.date).isin(us_holidays).astype("int8")
    return g


def build() -> pd.DataFrame:
    elec_wide = pd.read_parquet(PROCESSED / "electricity_selected.parquet")
    weather = pd.read_parquet(PROCESSED / "weather_selected.parquet").drop(columns=["site_id"])

    long = (
        elec_wide.reset_index()
        .melt(id_vars="timestamp", var_name="building_id", value_name="consumption")
        .sort_values(["building_id", "timestamp"])
        .set_index(["building_id", "timestamp"])
    )

    us_holidays = holidays.country_holidays("US", years=range(2016, 2018))

    parts = []
    for _, g in long.groupby(level="building_id", group_keys=False):
        g = add_lags(g.copy())
        g = add_rolling(g)
        g = add_calendar(g, us_holidays)
        parts.append(g)
    features = pd.concat(parts)

    # Join weather (same for all buildings, indexed by timestamp). BDG2's
    # weather file is missing a couple of hourly rows; reindex to a complete
    # hourly grid before ffill so windowed sequences never hit NaNs.
    full_range = pd.date_range(weather.index.min(), weather.index.max(), freq="h")
    weather_ffilled = weather.sort_index().reindex(full_range).ffill().bfill()
    weather_ffilled.index.name = "timestamp"
    features = features.join(weather_ffilled, on="timestamp", how="left")

    # Drop the warm-up rows that don't have full lag/rolling history
    features = features.dropna(subset=[f"lag_{max(LAGS)}", f"roll_std_{max(ROLL_WINDOWS)}"])
    return features


def assign_splits(features: pd.DataFrame) -> pd.Series:
    timestamps = features.index.get_level_values("timestamp").unique().sort_values()
    n = len(timestamps)
    train_end = int(n * TRAIN_FRAC)
    val_end = int(n * (TRAIN_FRAC + VAL_FRAC))
    split = pd.Series(index=timestamps, dtype="object")
    split.iloc[:train_end] = "train"
    split.iloc[train_end:val_end] = "val"
    split.iloc[val_end:] = "test"
    split.index.name = "timestamp"
    return split


def main() -> int:
    features = build()
    split = assign_splits(features)

    features.to_parquet(PROCESSED / "features.parquet")
    split.to_frame("split").to_parquet(PROCESSED / "split_index.parquet")

    print(f"features shape: {features.shape}")
    print(f"columns ({len(features.columns)}):")
    for c in features.columns:
        print(f"  {c}")
    print()
    counts = split.value_counts()
    for label in ("train", "val", "test"):
        ts = split[split == label].index
        print(f"{label:5s}: {counts[label]:>5d} hours  ({ts.min()} → {ts.max()})")
    print(f"\nwrote {PROCESSED}/features.parquet and split_index.parquet")
    return 0


if __name__ == "__main__":
    sys.exit(main())
