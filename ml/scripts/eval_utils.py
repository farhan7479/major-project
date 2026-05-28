"""Shared helpers used by every train_*.py script.

Keeps the feature loading, split slicing, and result persistence consistent
so the leaderboard compares like-for-like.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from metrics import compute_all

PROCESSED = Path(__file__).resolve().parents[1] / "data" / "processed"
RESULTS = Path(__file__).resolve().parents[1] / "models" / "results"
PREDICTIONS_DIR = RESULTS / "predictions"
METRICS_DIR = RESULTS / "metrics"

TARGET = "consumption"


def load_features_and_splits() -> tuple[pd.DataFrame, pd.Series]:
    features = pd.read_parquet(PROCESSED / "features.parquet")
    split = pd.read_parquet(PROCESSED / "split_index.parquet")["split"]
    return features, split


def split_mask(features: pd.DataFrame, split: pd.Series, label: str) -> pd.Series:
    """Boolean index aligned to `features` selecting rows whose timestamp is in `label`."""
    ts = features.index.get_level_values("timestamp")
    timestamps_in = split[split == label].index
    return pd.Series(ts.isin(timestamps_in), index=features.index)


def feature_columns(features: pd.DataFrame, exclude: tuple[str, ...] = (TARGET,)) -> list[str]:
    return [c for c in features.columns if c not in exclude]


def save_predictions(model_name: str, df: pd.DataFrame) -> Path:
    """df must have columns: building_id, timestamp, y_true, y_pred."""
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    path = PREDICTIONS_DIR / f"{model_name}.parquet"
    df.to_parquet(path, index=False)
    return path


def save_metrics(model_name: str, metrics: dict) -> Path:
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    path = METRICS_DIR / f"{model_name}.json"
    path.write_text(json.dumps(metrics, indent=2, default=float))
    return path


def evaluate_predictions(predictions: pd.DataFrame, y_train_by_building: dict[str, np.ndarray]) -> dict:
    """Per-building metrics + macro average across buildings."""
    per_building = {}
    for bid, g in predictions.groupby("building_id"):
        per_building[bid] = compute_all(
            g["y_true"].to_numpy(),
            g["y_pred"].to_numpy(),
            y_train=y_train_by_building.get(bid),
            season=24,
        )
    metric_names = next(iter(per_building.values())).keys()
    macro = {m: float(np.mean([per_building[b][m] for b in per_building])) for m in metric_names}
    return {"per_building": per_building, "macro": macro}


def get_train_targets_by_building(features: pd.DataFrame, split: pd.Series) -> dict[str, np.ndarray]:
    train_mask = split_mask(features, split, "train")
    out = {}
    for bid, g in features[train_mask].groupby(level="building_id"):
        out[bid] = g[TARGET].to_numpy()
    return out
