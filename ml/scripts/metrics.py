"""Forecasting metrics used across all model evaluations.

Every metric takes 1-D arrays of the same length. NaNs are masked out so a
single missing prediction doesn't poison the whole score.
"""

from __future__ import annotations

import numpy as np


def _align(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    return y_true[mask], y_pred[mask]


def mae(y_true, y_pred) -> float:
    y, p = _align(y_true, y_pred)
    return float(np.mean(np.abs(y - p)))


def rmse(y_true, y_pred) -> float:
    y, p = _align(y_true, y_pred)
    return float(np.sqrt(np.mean((y - p) ** 2)))


def mape(y_true, y_pred) -> float:
    y, p = _align(y_true, y_pred)
    nz = np.abs(y) > 1e-6
    if not nz.any():
        return float("nan")
    return float(np.mean(np.abs((y[nz] - p[nz]) / y[nz])) * 100)


def smape(y_true, y_pred) -> float:
    y, p = _align(y_true, y_pred)
    denom = (np.abs(y) + np.abs(p)) / 2
    nz = denom > 1e-6
    if not nz.any():
        return float("nan")
    return float(np.mean(np.abs(y[nz] - p[nz]) / denom[nz]) * 100)


def mase(y_true, y_pred, y_train, season: int = 24) -> float:
    """Mean Absolute Scaled Error vs. in-sample seasonal-naive."""
    y, p = _align(y_true, y_pred)
    y_train = np.asarray(y_train, dtype=float)
    y_train = y_train[np.isfinite(y_train)]
    if len(y_train) <= season:
        return float("nan")
    scale = np.mean(np.abs(y_train[season:] - y_train[:-season]))
    if scale < 1e-9:
        return float("nan")
    return float(np.mean(np.abs(y - p)) / scale)


def r2(y_true, y_pred) -> float:
    y, p = _align(y_true, y_pred)
    ss_res = np.sum((y - p) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    if ss_tot < 1e-9:
        return float("nan")
    return float(1 - ss_res / ss_tot)


def peak_metrics(y_true, y_pred, top_pct: float = 0.05) -> dict[str, float]:
    """Precision / recall / F1 for whether each hour falls in the top `top_pct` of the test set."""
    y, p = _align(y_true, y_pred)
    if len(y) == 0:
        return {"peak_precision": float("nan"), "peak_recall": float("nan"), "peak_f1": float("nan")}
    k = max(1, int(len(y) * top_pct))
    true_thresh = np.partition(y, -k)[-k]
    pred_thresh = np.partition(p, -k)[-k]
    true_peak = y >= true_thresh
    pred_peak = p >= pred_thresh
    tp = int(np.sum(true_peak & pred_peak))
    fp = int(np.sum(~true_peak & pred_peak))
    fn = int(np.sum(true_peak & ~pred_peak))
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"peak_precision": precision, "peak_recall": recall, "peak_f1": f1}


def compute_all(y_true, y_pred, y_train=None, season: int = 24) -> dict[str, float]:
    out = {
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "mape": mape(y_true, y_pred),
        "smape": smape(y_true, y_pred),
        "r2": r2(y_true, y_pred),
    }
    if y_train is not None:
        out["mase"] = mase(y_true, y_pred, y_train, season=season)
    out.update(peak_metrics(y_true, y_pred))
    return out
