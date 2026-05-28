"""Combine all trained-model metrics into a single leaderboard + comparison plots."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

RESULTS = Path(__file__).resolve().parents[1] / "models" / "results"
METRICS_DIR = RESULTS / "metrics"
PREDICTIONS_DIR = RESULTS / "predictions"

MODEL_DISPLAY_ORDER = [
    "naive_last",
    "naive_seasonal_24h",
    "arima",
    "lstm",
    "bilstm",
    "cnn_lstm",
    "transformer",
    "random_forest",
    "xgboost",
]


def build_leaderboard() -> pd.DataFrame:
    rows = []
    for path in sorted(METRICS_DIR.glob("*.json")):
        data = json.loads(path.read_text())
        row = {"model": path.stem, **data["macro"]}
        rows.append(row)
    df = pd.DataFrame(rows)
    df["_order"] = df["model"].map({m: i for i, m in enumerate(MODEL_DISPLAY_ORDER)}).fillna(99)
    df = df.sort_values("_order").drop(columns="_order").reset_index(drop=True)
    return df


def plot_mae_comparison(lb: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    palette = {
        "naive_last": "#aaaaaa",
        "naive_seasonal_24h": "#cccccc",
        "arima": "#3a6ea5",
        "lstm": "#9b59b6",
        "bilstm": "#8e44ad",
        "cnn_lstm": "#6c3483",
        "transformer": "#117a65",
        "random_forest": "#2e8b57",
        "xgboost": "#d2691e",
    }
    colors = [palette.get(m, "#888") for m in lb["model"]]
    ax.bar(lb["model"], lb["mae"], color=colors)
    for i, v in enumerate(lb["mae"]):
        ax.text(i, v + 0.05, f"{v:.2f}", ha="center", fontsize=10)
    ax.set_ylabel("test MAE (kWh)")
    ax.set_title("Model comparison — test set MAE (lower is better)")
    ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    fig.savefig(RESULTS / "leaderboard_mae.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_actual_vs_predicted(model_name: str, building: str, start: str = "2017-10-09", days: int = 7) -> None:
    df = pd.read_parquet(PREDICTIONS_DIR / f"{model_name}.parquet")
    df = df[df["building_id"] == building].copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.set_index("timestamp").sort_index()
    start_ts = pd.Timestamp(start)
    window = df.loc[start_ts:start_ts + pd.Timedelta(days=days)]
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(window.index, window["y_true"], label="actual", color="black", linewidth=1.5)
    ax.plot(window.index, window["y_pred"], label="predicted", color="#d2691e", linewidth=1.5, alpha=0.85)
    ax.set_title(f"{model_name} — {building} ({start_ts.date()} → {(start_ts + pd.Timedelta(days=days)).date()})")
    ax.set_ylabel("kWh / hr")
    ax.legend()
    fig.tight_layout()
    out = RESULTS / "actual_vs_predicted" / f"{model_name}__{building}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    lb = build_leaderboard()
    lb.to_csv(RESULTS / "leaderboard.csv", index=False)

    cols = ["model", "mae", "rmse", "mape", "r2", "peak_f1"]
    print("\nLEADERBOARD (macro across 3 buildings, test split)")
    print("=" * 72)
    print(lb[cols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    plot_mae_comparison(lb)
    for model_name in lb["model"]:
        for building in ["Hog_office_Lizzie", "Hog_office_Sydney", "Hog_office_Myles"]:
            plot_actual_vs_predicted(model_name, building)

    print(f"\nwrote {RESULTS}/leaderboard.csv, leaderboard_mae.png, actual_vs_predicted/*.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
