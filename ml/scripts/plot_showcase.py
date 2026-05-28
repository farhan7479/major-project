"""One headline figure for the report's RESULTS chapter.

Three subplots (one per building), each showing actual consumption and the
top-3 models overlaid for a representative 2-week test window. This is the
figure the report's text will refer to as "Figure 5.1 — Model forecasts on
the test set."
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

RESULTS = Path(__file__).resolve().parents[1] / "models" / "results"
PREDICTIONS_DIR = RESULTS / "predictions"
METRICS_DIR = RESULTS / "metrics"

WINDOW_START = "2017-11-13"
WINDOW_DAYS = 14
BUILDINGS = ["Hog_office_Sydney", "Hog_office_Lizzie", "Hog_office_Myles"]

MODEL_LABEL = {
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
    "cnn_lstm": "CNN-LSTM",
    "lstm": "LSTM",
    "bilstm": "BiLSTM",
    "arima": "ARIMA",
    "transformer": "Transformer",
}
MODEL_COLOR = {
    "random_forest": "#2e8b57",
    "xgboost": "#d2691e",
    "cnn_lstm": "#6c3483",
    "lstm": "#9b59b6",
    "bilstm": "#8e44ad",
    "arima": "#3a6ea5",
    "transformer": "#117a65",
}


def top_n_models(n: int = 3) -> list[str]:
    scores = []
    for path in METRICS_DIR.glob("*.json"):
        data = json.loads(path.read_text())
        scores.append((path.stem, data["macro"]["mae"]))
    scores.sort(key=lambda t: t[1])
    return [m for m, _ in scores[:n]]


def load_predictions(model: str, building: str, start_ts: pd.Timestamp, days: int) -> pd.Series:
    df = pd.read_parquet(PREDICTIONS_DIR / f"{model}.parquet")
    df = df[df["building_id"] == building].copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.set_index("timestamp").sort_index()
    return df.loc[start_ts:start_ts + pd.Timedelta(days=days), "y_pred"]


def main() -> int:
    models = top_n_models(3)
    print(f"top 3 models: {models}")
    start_ts = pd.Timestamp(WINDOW_START)
    end_ts = start_ts + pd.Timedelta(days=WINDOW_DAYS)

    fig, axes = plt.subplots(len(BUILDINGS), 1, figsize=(13, 3.2 * len(BUILDINGS)), sharex=True)
    for ax, building in zip(axes, BUILDINGS):
        # Pull actuals from any model (same y_true everywhere)
        actuals = pd.read_parquet(PREDICTIONS_DIR / f"{models[0]}.parquet")
        actuals = actuals[actuals["building_id"] == building].copy()
        actuals["timestamp"] = pd.to_datetime(actuals["timestamp"])
        actuals = actuals.set_index("timestamp").sort_index().loc[start_ts:end_ts, "y_true"]

        ax.plot(actuals.index, actuals.values, color="black", linewidth=1.6, label="actual", zorder=5)
        for m in models:
            preds = load_predictions(m, building, start_ts, WINDOW_DAYS)
            ax.plot(preds.index, preds.values, color=MODEL_COLOR.get(m, "#888"),
                    linewidth=1.0, alpha=0.85, label=MODEL_LABEL.get(m, m))
        ax.set_ylabel(f"{building.replace('Hog_office_', '')}\nkWh / hr")
        ax.grid(True, alpha=0.3)
        if ax is axes[0]:
            ax.legend(loc="upper right", ncol=len(models) + 1, fontsize=9)

    axes[-1].set_xlabel(f"hour ({start_ts.date()} → {end_ts.date()})")
    fig.suptitle(f"Top-3 model forecasts vs actual consumption — {WINDOW_DAYS}-day test window", y=1.00)
    fig.tight_layout()
    out = RESULTS / "showcase_top3_models.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
