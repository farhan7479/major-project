"""Visualize how well the best model identifies the test-set peak hours.

For each building we pick a 2-week window with high variability and overlay
true peaks (top 5% of test consumption) in red and predicted peaks in
orange — true positives sit under both markers, false positives stand
alone in orange, and missed peaks (false negatives) stand alone in red.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RESULTS = Path(__file__).resolve().parents[1] / "models" / "results"
PREDICTIONS_DIR = RESULTS / "predictions"
METRICS_DIR = RESULTS / "metrics"

WINDOWS = {
    "Hog_office_Lizzie": ("2017-10-15", 14),
    "Hog_office_Sydney": ("2017-10-15", 14),
    "Hog_office_Myles": ("2017-10-15", 14),
}


def best_model_overall() -> str:
    """Pick the model with the lowest macro MAE."""
    best, best_mae = "", float("inf")
    for path in METRICS_DIR.glob("*.json"):
        data = json.loads(path.read_text())
        mae = data["macro"]["mae"]
        if mae < best_mae:
            best_mae = mae
            best = path.stem
    if not best:
        raise RuntimeError(f"no metric files found in {METRICS_DIR}")
    return best


def plot(model: str, building: str, start: str, days: int) -> None:
    df = pd.read_parquet(PREDICTIONS_DIR / f"{model}.parquet")
    df = df[df["building_id"] == building].copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.set_index("timestamp").sort_index()

    k = max(1, int(len(df) * 0.05))
    true_thresh = np.partition(df["y_true"].to_numpy(), -k)[-k]
    pred_thresh = np.partition(df["y_pred"].to_numpy(), -k)[-k]
    df["is_true_peak"] = df["y_true"] >= true_thresh
    df["is_pred_peak"] = df["y_pred"] >= pred_thresh

    start_ts = pd.Timestamp(start)
    window = df.loc[start_ts:start_ts + pd.Timedelta(days=days)]

    fig, ax = plt.subplots(figsize=(13, 4.5))
    ax.plot(window.index, window["y_true"], label="actual", color="black", linewidth=1.2)
    ax.plot(window.index, window["y_pred"], label=f"{model} prediction", color="#1f77b4", linewidth=1.0, alpha=0.85)

    true_peaks = window[window["is_true_peak"]]
    pred_peaks = window[window["is_pred_peak"]]
    ax.scatter(true_peaks.index, true_peaks["y_true"], color="#d62728", s=42, zorder=3, label="true peak (top 5%)")
    ax.scatter(pred_peaks.index, pred_peaks["y_pred"], color="#ff7f0e", s=42, marker="x", zorder=3, label="predicted peak")

    ax.set_ylabel("kWh / hr")
    ax.set_title(f"Peak detection — {model} on {building.replace('Hog_office_', '')} "
                 f"({start_ts.date()} → {(start_ts + pd.Timedelta(days=days)).date()})")
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    out = RESULTS / "peak_detection" / f"{model}__{building.replace('Hog_office_', '')}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    model = best_model_overall()
    print(f"best model overall (by macro MAE): {model}")
    for building, (start, days) in WINDOWS.items():
        plot(model, building, start, days)
        print(f"  wrote peak_detection/{model}__{building.replace('Hog_office_', '')}.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
