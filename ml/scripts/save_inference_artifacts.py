"""Materialize everything the FastAPI service needs to load at startup.

Writes to ml/serve/artifacts/:
  context.parquet     — full (building, timestamp) consumption + weather, used as
                        historical context when the API is asked to predict for a
                        timestamp inside the BDG2 range
  scalers.json        — per-building target mean/std + global feature mean/std,
                        identical to what the DL training scripts used
  manifest.json       — ordered building list, test-period start/end, and the
                        feature-column order the models expect
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from eval_utils import load_features_and_splits
from torch_data import TIMESTEP_FEATURES

OUT = Path(__file__).resolve().parents[1] / "serve" / "artifacts"

CONTEXT_COLUMNS = [
    "consumption",
    "airTemperature",
    "dewTemperature",
    "cloudCoverage",
    "windSpeed",
    "windDirection",
    "seaLvlPressure",
    "precipDepth1HR",
]


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    features, split = load_features_and_splits()

    # Context = all (building, timestamp) rows with raw consumption + weather.
    # We deliberately keep the full series so the API can predict for any
    # timestamp in the BDG2 window without re-running feature engineering.
    context = features[CONTEXT_COLUMNS].reset_index()
    context.to_parquet(OUT / "context.parquet", index=False)
    print(f"context: {context.shape} rows, {context['building_id'].nunique()} buildings")

    # Scalers — recompute the same way torch_data does so the API matches training
    train_ts = split[split == "train"].index
    train_mask = features.index.get_level_values("timestamp").isin(train_ts)
    train = features[train_mask]

    feat_mean = train[TIMESTEP_FEATURES].mean().tolist()
    feat_std = train[TIMESTEP_FEATURES].std(ddof=0).replace(0, 1).tolist()
    target_mean = {bid: float(g["consumption"].mean()) for bid, g in train.groupby(level="building_id")}
    target_std = {bid: float(g["consumption"].std(ddof=0) or 1.0) for bid, g in train.groupby(level="building_id")}

    scalers = {
        "timestep_features": TIMESTEP_FEATURES,
        "feat_mean": feat_mean,
        "feat_std": feat_std,
        "target_mean": target_mean,
        "target_std": target_std,
    }
    (OUT / "scalers.json").write_text(json.dumps(scalers, indent=2))

    # Manifest
    buildings = sorted(features.index.get_level_values("building_id").unique())
    test_ts = pd.DatetimeIndex(split[split == "test"].index)
    val_ts = pd.DatetimeIndex(split[split == "val"].index)

    # The full set of feature columns the tree models expect, in order.
    # Equivalent to feature_columns(features) + building one-hot columns.
    tabular_features = [c for c in features.columns if c != "consumption"]
    building_dummies = [f"bld_{b}" for b in buildings]
    tabular_features_full = tabular_features + building_dummies

    manifest = {
        "buildings": buildings,
        "test_period": {
            "start": test_ts.min().isoformat(),
            "end": test_ts.max().isoformat(),
        },
        "val_period": {
            "start": val_ts.min().isoformat(),
            "end": val_ts.max().isoformat(),
        },
        "tabular_feature_columns": tabular_features_full,
        "seq_len": 168,
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"wrote {OUT}/context.parquet, scalers.json, manifest.json")
    print(f"buildings: {buildings}")
    print(f"test period: {manifest['test_period']['start']} → {manifest['test_period']['end']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
