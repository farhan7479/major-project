"""Sequence dataset shared by every deep-learning model.

For each (building, timestamp) in the chosen split we emit:
    X: (seq_len, n_features) of past per-hour features (consumption + weather + calendar + building one-hot)
    y: scalar consumption at the target hour, in per-building z-score space

Sequences never cross building boundaries.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from eval_utils import load_features_and_splits

SEQ_LEN = 168

# Features fed at each timestep. We deliberately omit the engineered lag_24 /
# lag_168 / roll_* columns — the LSTM should learn those summaries from the
# sequence itself. Building identity comes in via one-hot columns appended later.
TIMESTEP_FEATURES = [
    "lag_1",
    "airTemperature",
    "dewTemperature",
    "cloudCoverage",
    "windSpeed",
    "seaLvlPressure",
    "precipDepth1HR",
    "hour_sin",
    "hour_cos",
    "doy_sin",
    "doy_cos",
    "is_weekend",
    "is_peak_hour",
    "is_holiday",
]


@dataclass
class Scalers:
    """Normalization stats computed on the train split only."""
    feat_mean: np.ndarray
    feat_std: np.ndarray
    target_mean: dict[str, float]
    target_std: dict[str, float]


def _fit_scalers(features: pd.DataFrame, train_mask: pd.Series, cols: list[str]) -> Scalers:
    train = features[train_mask]
    feat_mean = train[cols].mean().to_numpy()
    feat_std = train[cols].std(ddof=0).replace(0, 1).to_numpy()
    target_mean = {bid: float(g["consumption"].mean()) for bid, g in train.groupby(level="building_id")}
    target_std = {bid: float(g["consumption"].std(ddof=0) or 1.0) for bid, g in train.groupby(level="building_id")}
    return Scalers(feat_mean, feat_std, target_mean, target_std)


@dataclass
class SplitBundle:
    X: torch.Tensor          # (n_samples, seq_len, n_features)
    y: torch.Tensor          # (n_samples,)  z-score normalized
    building_id: list[str]   # per-sample, for de-normalization
    timestamp: pd.DatetimeIndex


def _windows_for_building(
    g: pd.DataFrame,
    cols: list[str],
    bid: str,
    all_buildings: list[str],
    target_in_split: pd.DatetimeIndex,
    scalers: Scalers,
    seq_len: int,
) -> tuple[np.ndarray, np.ndarray, list[str], list[pd.Timestamp]]:
    g = g.sort_index()
    timestamps = g.index.get_level_values("timestamp")
    feats = ((g[cols].to_numpy() - scalers.feat_mean) / scalers.feat_std).astype(np.float32)
    consumption = g["consumption"].to_numpy(dtype=np.float32)

    one_hot = np.zeros(len(all_buildings), dtype=np.float32)
    one_hot[all_buildings.index(bid)] = 1.0

    target_set = set(target_in_split)
    xs, ys, bids, tss = [], [], [], []
    for t_idx in range(seq_len, len(g)):
        ts = timestamps[t_idx]
        if ts not in target_set:
            continue
        window = feats[t_idx - seq_len:t_idx]
        # Append building one-hot to every timestep
        window = np.concatenate([window, np.tile(one_hot, (seq_len, 1))], axis=1)
        y_norm = (consumption[t_idx] - scalers.target_mean[bid]) / scalers.target_std[bid]
        xs.append(window)
        ys.append(y_norm)
        bids.append(bid)
        tss.append(ts)
    return np.stack(xs), np.array(ys, dtype=np.float32), bids, tss


def build_splits(seq_len: int = SEQ_LEN) -> tuple[SplitBundle, SplitBundle, SplitBundle, Scalers, list[str]]:
    features, split = load_features_and_splits()
    cols = TIMESTEP_FEATURES
    train_ts = pd.DatetimeIndex(split[split == "train"].index)
    train_mask = pd.Series(
        features.index.get_level_values("timestamp").isin(train_ts),
        index=features.index,
    )
    scalers = _fit_scalers(features, train_mask, cols)

    all_buildings = sorted(features.index.get_level_values("building_id").unique())

    bundles = {}
    for label in ("train", "val", "test"):
        target_ts = pd.DatetimeIndex(split[split == label].index)
        Xs, Ys, BIDs, TSs = [], [], [], []
        for bid, g in features.groupby(level="building_id"):
            x, y, bids_g, tss_g = _windows_for_building(g, cols, bid, all_buildings, target_ts, scalers, seq_len)
            Xs.append(x)
            Ys.append(y)
            BIDs.extend(bids_g)
            TSs.extend(tss_g)
        X = torch.from_numpy(np.concatenate(Xs))
        Y = torch.from_numpy(np.concatenate(Ys))
        bundles[label] = SplitBundle(X=X, y=Y, building_id=BIDs, timestamp=pd.DatetimeIndex(TSs))

    return bundles["train"], bundles["val"], bundles["test"], scalers, all_buildings


class WindowDataset(Dataset):
    def __init__(self, bundle: SplitBundle):
        self.X = bundle.X
        self.y = bundle.y

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, i):
        return self.X[i], self.y[i]


def denormalize(y_norm: np.ndarray, building_ids: list[str], scalers: Scalers) -> np.ndarray:
    out = np.empty_like(y_norm, dtype=np.float64)
    for i, b in enumerate(building_ids):
        out[i] = y_norm[i] * scalers.target_std[b] + scalers.target_mean[b]
    return out
