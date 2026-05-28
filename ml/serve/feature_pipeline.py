"""Inference-time feature engineering.

Mirrors ml/scripts/build_features.py exactly so the API row a model sees at
inference matches what it saw during training. Lives separately so the
training script can keep its batch-processing shape while inference handles
single-target rows fast.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import holidays
import numpy as np
import pandas as pd

LAGS = (1, 24, 168)
ROLL_WINDOWS = (3, 6, 12, 24)
PEAK_HOURS = set(range(9, 13)) | set(range(18, 22))
WEATHER_COLS = ("airTemperature", "dewTemperature", "cloudCoverage", "windSpeed",
                "seaLvlPressure", "precipDepth1HR", "windDirection")


@dataclass
class Artifacts:
    context: pd.DataFrame              # (building_id, timestamp, consumption + weather)
    scalers: dict
    manifest: dict
    holidays: holidays.HolidayBase

    @classmethod
    def load(cls, artifacts_dir: Path) -> "Artifacts":
        context = pd.read_parquet(artifacts_dir / "context.parquet")
        context["timestamp"] = pd.to_datetime(context["timestamp"])
        context = context.set_index(["building_id", "timestamp"]).sort_index()
        scalers = json.loads((artifacts_dir / "scalers.json").read_text())
        manifest = json.loads((artifacts_dir / "manifest.json").read_text())
        us = holidays.country_holidays("US", years=range(2016, 2018))
        return cls(context=context, scalers=scalers, manifest=manifest, holidays=us)


def _window(art: Artifacts, building_id: str, target_ts: pd.Timestamp, hours: int) -> pd.DataFrame:
    """Return the `hours` rows ending at target_ts - 1h for the given building."""
    series = art.context.loc[building_id]
    end = target_ts - pd.Timedelta(hours=1)
    start = end - pd.Timedelta(hours=hours - 1)
    window = series.loc[start:end]
    if len(window) < hours:
        raise ValueError(f"need {hours} hours of context ending {end}, got {len(window)}")
    return window


def build_tabular_row(
    art: Artifacts,
    building_id: str,
    target_ts: pd.Timestamp,
    weather_override: Optional[dict] = None,
) -> pd.Series:
    """Produce the single feature row a tree model expects.

    Columns and order match manifest['tabular_feature_columns'].
    """
    cols = art.manifest["tabular_feature_columns"]
    window = _window(art, building_id, target_ts, hours=max(LAGS) + 1)  # need lag_168
    consumption = window["consumption"]

    row = pd.Series(0.0, index=cols)

    # Lags
    for k in LAGS:
        row[f"lag_{k}"] = consumption.iloc[-k]

    # Rolling stats over the last w hours (excluding the target hour itself).
    # Use pandas default ddof=1 to match build_features.py's .rolling(w).std().
    for w in ROLL_WINDOWS:
        recent = consumption.iloc[-w:]
        row[f"roll_mean_{w}"] = recent.mean()
        row[f"roll_std_{w}"] = recent.std()

    # Calendar features computed at the target hour
    row["hour"] = target_ts.hour
    row["dayofweek"] = target_ts.dayofweek
    row["month"] = target_ts.month
    row["hour_sin"] = np.sin(2 * np.pi * target_ts.hour / 24)
    row["hour_cos"] = np.cos(2 * np.pi * target_ts.hour / 24)
    row["doy_sin"] = np.sin(2 * np.pi * target_ts.dayofyear / 365.25)
    row["doy_cos"] = np.cos(2 * np.pi * target_ts.dayofyear / 365.25)
    row["is_weekend"] = int(target_ts.dayofweek >= 5)
    row["is_peak_hour"] = int(target_ts.hour in PEAK_HOURS)
    row["is_holiday"] = int(target_ts.date() in art.holidays)

    # Weather — default to actual value at target_ts from context, allow overrides
    weather_row = art.context.loc[(building_id, target_ts)] if (building_id, target_ts) in art.context.index else None
    for col in WEATHER_COLS:
        if weather_override and col in weather_override:
            row[col] = float(weather_override[col])
        elif weather_row is not None:
            row[col] = float(weather_row[col])
        else:
            row[col] = float(window[col].iloc[-1])

    # Building one-hot
    row[f"bld_{building_id}"] = 1
    return row


def build_sequence(
    art: Artifacts,
    building_id: str,
    target_ts: pd.Timestamp,
) -> np.ndarray:
    """Produce the (1, seq_len, n_features) tensor a DL model expects.

    Per-timestep features = TIMESTEP_FEATURES + 3 building one-hot columns.
    Normalized with the same scalers used at training time. Note that DL
    models see only the past sequence ending at target_ts - 1h, so weather
    overrides at target_ts don't apply here (they apply only to tabular).
    """
    seq_len = art.manifest["seq_len"]
    window = _window(art, building_id, target_ts, hours=seq_len + 1)  # need lag_1 chain

    # Window covers timestamps [target_ts - (seq_len + 1)h, target_ts - 1h]. The
    # input sequence covers [target_ts - seq_len, target_ts - 1] (the last seq_len
    # rows). At each step i its lag_1 = consumption one hour earlier, i.e. window
    # row at the position before step i.
    cons = window["consumption"].to_numpy(dtype=np.float32)
    lag_1_per_step = cons[:-1]                # 168 values, lag_1 at the seq positions
    step_rows = window.iloc[1:]               # 168 rows, weather + timestamps at the seq positions
    step_timestamps = step_rows.index

    rows = []
    for i, ts in enumerate(step_timestamps):
        weather_at_step = step_rows.iloc[i]
        # Override only applies to the target hour, not to past timesteps
        row = [
            lag_1_per_step[i],
            float(weather_at_step["airTemperature"]),
            float(weather_at_step["dewTemperature"]),
            float(weather_at_step["cloudCoverage"]),
            float(weather_at_step["windSpeed"]),
            float(weather_at_step["seaLvlPressure"]),
            float(weather_at_step["precipDepth1HR"]),
            np.sin(2 * np.pi * ts.hour / 24),
            np.cos(2 * np.pi * ts.hour / 24),
            np.sin(2 * np.pi * ts.dayofyear / 365.25),
            np.cos(2 * np.pi * ts.dayofyear / 365.25),
            int(ts.dayofweek >= 5),
            int(ts.hour in PEAK_HOURS),
            int(ts.date() in art.holidays),
        ]
        rows.append(row)
    feats = np.asarray(rows, dtype=np.float32)

    # Apply training-time normalization
    feat_mean = np.asarray(art.scalers["feat_mean"], dtype=np.float32)
    feat_std = np.asarray(art.scalers["feat_std"], dtype=np.float32)
    feats = (feats - feat_mean) / feat_std

    # Building one-hot tiled across all timesteps
    buildings = art.manifest["buildings"]
    one_hot = np.zeros(len(buildings), dtype=np.float32)
    one_hot[buildings.index(building_id)] = 1.0
    one_hot_seq = np.tile(one_hot, (seq_len, 1))

    seq = np.concatenate([feats, one_hot_seq], axis=1)
    return seq[np.newaxis, :, :]  # (1, seq_len, n_features)


def denormalize_target(y_norm: float, building_id: str, scalers: dict) -> float:
    return y_norm * scalers["target_std"][building_id] + scalers["target_mean"][building_id]


def consumption_series(art: Artifacts, building_id: str) -> pd.Series:
    """Full hourly consumption series for a building — used by ARIMA inference."""
    series = art.context.loc[building_id, "consumption"].sort_index()
    series.index.freq = "h"
    return series
