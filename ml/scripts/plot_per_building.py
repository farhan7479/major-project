"""Per-building breakdown of every model's test-set performance.

Outputs:
  results/per_building_leaderboard.csv  — flat table (model, building, metrics)
  results/per_building_heatmap.png      — MAE heatmap (rows=model, cols=building)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

RESULTS = Path(__file__).resolve().parents[1] / "models" / "results"
METRICS_DIR = RESULTS / "metrics"

MODEL_ORDER = [
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


def collect() -> pd.DataFrame:
    rows = []
    for path in sorted(METRICS_DIR.glob("*.json")):
        data = json.loads(path.read_text())
        model = path.stem
        for bid, metrics in data["per_building"].items():
            rows.append({"model": model, "building": bid.replace("Hog_office_", ""), **metrics})
    df = pd.DataFrame(rows)
    df["_order"] = df["model"].map({m: i for i, m in enumerate(MODEL_ORDER)}).fillna(99)
    df = df.sort_values(["_order", "building"]).drop(columns="_order").reset_index(drop=True)
    return df


def plot_heatmap(df: pd.DataFrame, metric: str, out: Path, title: str) -> None:
    pivot = df.pivot(index="model", columns="building", values=metric).reindex(
        [m for m in MODEL_ORDER if m in df["model"].unique()]
    )
    fig, ax = plt.subplots(figsize=(6, 0.55 * len(pivot) + 1.5))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".2f",
        cmap="RdYlGn_r",
        cbar_kws={"label": title},
        ax=ax,
    )
    ax.set_title(f"{title} per model per building (lower = better)")
    ax.set_ylabel("")
    ax.set_xlabel("")
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    df = collect()
    df.to_csv(RESULTS / "per_building_leaderboard.csv", index=False)

    cols = ["model", "building", "mae", "rmse", "mape", "r2", "peak_f1"]
    print("\nPER-BUILDING LEADERBOARD")
    print("=" * 80)
    print(df[cols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    plot_heatmap(df, "mae", RESULTS / "per_building_mae_heatmap.png", "MAE (kWh)")
    plot_heatmap(df, "mape", RESULTS / "per_building_mape_heatmap.png", "MAPE (%)")
    print(f"\nwrote {RESULTS}/per_building_leaderboard.csv + 2 heatmaps")
    return 0


if __name__ == "__main__":
    sys.exit(main())
