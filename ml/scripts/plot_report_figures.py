"""Generate the additional figures referenced by the rewritten report.

Writes to ml/models/results/report/:
  system_architecture.png    block diagram of the data + model + serving pipeline
  training_curves.png        train loss + val MAE per epoch for all 4 DL models
  residual_histograms.png    error distribution per model (test set)
  hourly_error_heatmap.png   winner model's MAE by (hour, weekday)
  params_vs_mae.png          parameter count vs test MAE for every model
"""

from __future__ import annotations

import pickle
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
PREDICTIONS = ROOT / "models" / "results" / "predictions"
CHECKPOINTS = ROOT / "models" / "checkpoints"
OUT = ROOT / "models" / "results" / "report"

DL_LOGS = {
    "LSTM": Path("/tmp/lstm_final.log"),
    "BiLSTM": Path("/tmp/bilstm_final.log"),
    "CNN-LSTM": Path("/tmp/cnn_lstm_final.log"),
    "Transformer": Path("/tmp/transformer_final.log"),
}

LEADERBOARD_ORDER = [
    "random_forest", "xgboost", "transformer", "arima",
    "naive_last", "cnn_lstm", "naive_seasonal_24h", "lstm", "bilstm",
]
DISPLAY = {
    "naive_last": "Naive (last)",
    "naive_seasonal_24h": "Naive (24h)",
    "arima": "ARIMA",
    "lstm": "LSTM",
    "bilstm": "BiLSTM",
    "cnn_lstm": "CNN-LSTM",
    "transformer": "Transformer",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
}
COLORS = {
    "random_forest": "#2e8b57",
    "xgboost": "#d2691e",
    "transformer": "#117a65",
    "cnn_lstm": "#6c3483",
    "lstm": "#9b59b6",
    "bilstm": "#8e44ad",
    "arima": "#3a6ea5",
    "naive_last": "#888888",
    "naive_seasonal_24h": "#bbbbbb",
}


def plot_system_architecture() -> None:
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 60)
    ax.axis("off")

    def box(x, y, w, h, label, fc, ec="#333333"):
        rect = mpatches.FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.02,rounding_size=1.2",
            linewidth=1.0, edgecolor=ec, facecolor=fc,
        )
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=9, color="#111111")

    def arrow(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color="#555555", lw=1.0))

    # Data layer
    box(2, 48, 22, 8, "BDG2 raw\nelectricity + weather + metadata", "#fff2cc")
    box(2, 36, 22, 8, "select_buildings.py\n3 office buildings @ site Hog", "#fff2cc")
    box(2, 24, 22, 8, "build_features.py\n30-col engineered features", "#fff2cc")
    box(2, 12, 22, 8, "features.parquet\n+ chronological train/val/test", "#fff2cc")

    arrow(13, 48, 13, 44)
    arrow(13, 36, 13, 32)
    arrow(13, 24, 13, 20)

    # Training layer (three columns)
    box(30, 38, 20, 8, "Tree models\nRF + XGBoost", "#d5e8d4")
    box(30, 26, 20, 8, "DL models\nLSTM / BiLSTM / CNN-LSTM /\nTransformer encoder", "#d5e8d4")
    box(30, 14, 20, 8, "Classical\nARIMA(2,1,2) per building", "#d5e8d4")

    arrow(24, 16, 30, 18)
    arrow(24, 16, 30, 30)
    arrow(24, 16, 30, 42)

    # Checkpoint + registry
    box(56, 26, 16, 20, "ml/serve/\nmodel_registry.py\n\n7 trained\ncheckpoints\nloaded once\nat startup", "#dae8fc")
    arrow(50, 42, 56, 38)
    arrow(50, 30, 56, 33)
    arrow(50, 18, 56, 28)

    # Service layer
    box(78, 32, 20, 12, "FastAPI\n/health  /models\n/buildings  /metrics\n/forecast\n/batch-forecast", "#f8cecc")
    arrow(72, 36, 78, 38)

    # Frontend
    box(78, 12, 20, 12, "React + Vite + Tailwind\n\n/forecast page\n/dashboard page", "#e1d5e7")
    arrow(88, 32, 88, 24)

    fig.suptitle("End-to-end system architecture", fontsize=12, y=0.97)
    fig.tight_layout()
    fig.savefig(OUT / "system_architecture.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def parse_log(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    pattern = re.compile(r"^epoch\s+(\d+)\s+train_loss=([\d.]+)\s+val_MAE=([\d.]+)")
    rows = []
    for line in path.read_text().splitlines():
        m = pattern.match(line)
        if m:
            rows.append({"epoch": int(m.group(1)), "train_loss": float(m.group(2)), "val_mae": float(m.group(3))})
    return pd.DataFrame(rows)


def plot_training_curves() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    palette = {"LSTM": "#9b59b6", "BiLSTM": "#8e44ad", "CNN-LSTM": "#6c3483", "Transformer": "#117a65"}
    for label, log_path in DL_LOGS.items():
        df = parse_log(log_path)
        if df.empty:
            continue
        axes[0].plot(df["epoch"], df["train_loss"], label=label, color=palette[label], linewidth=1.4)
        axes[1].plot(df["epoch"], df["val_mae"], label=label, color=palette[label], linewidth=1.4)
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("training loss (MSE, z-score space)")
    axes[0].set_title("Training loss")
    axes[0].set_yscale("log")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=8)
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("validation MAE (kWh)")
    axes[1].set_title("Validation MAE")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "training_curves.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_residual_histograms() -> None:
    models = ["random_forest", "xgboost", "transformer", "arima", "cnn_lstm", "lstm", "bilstm"]
    fig, axes = plt.subplots(2, 4, figsize=(12, 5.5), sharex=True)
    axes = axes.flatten()
    for ax, m in zip(axes, models):
        df = pd.read_parquet(PREDICTIONS / f"{m}.parquet")
        res = df["y_pred"] - df["y_true"]
        ax.hist(res, bins=60, color=COLORS[m], alpha=0.85, edgecolor="white", linewidth=0.3)
        ax.axvline(0, color="#333", linewidth=0.8, linestyle="--")
        ax.set_title(f"{DISPLAY[m]}\nmean={res.mean():.2f}  std={res.std():.2f}", fontsize=9)
        ax.set_xlim(-50, 50)
        ax.grid(True, alpha=0.25)
    for ax in axes[len(models):]:
        ax.axis("off")
    for ax in axes:
        ax.set_xlabel("residual (predicted − actual, kWh)", fontsize=8)
    fig.suptitle("Residual distributions on the test set (all 3 buildings pooled)", y=1.00)
    fig.tight_layout()
    fig.savefig(OUT / "residual_histograms.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_hourly_error_heatmap() -> None:
    df = pd.read_parquet(PREDICTIONS / "random_forest.parquet")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["hour"] = df["timestamp"].dt.hour
    df["weekday"] = df["timestamp"].dt.day_name()
    df["abs_err"] = (df["y_pred"] - df["y_true"]).abs()
    pivot = df.groupby(["weekday", "hour"])["abs_err"].mean().unstack(fill_value=0)
    week_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    pivot = pivot.reindex(week_order)
    fig, ax = plt.subplots(figsize=(12, 3.8))
    sns.heatmap(pivot, cmap="YlOrRd", ax=ax, cbar_kws={"label": "mean |error| (kWh)"}, linewidth=0.2, linecolor="#fff")
    ax.set_title("Random Forest mean absolute error by hour of day × weekday (test set)")
    ax.set_xlabel("hour of day")
    ax.set_ylabel("")
    fig.tight_layout()
    fig.savefig(OUT / "hourly_error_heatmap.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def count_torch_params(pt_path: Path) -> int:
    import torch
    state = torch.load(pt_path, map_location="cpu", weights_only=True)
    return int(sum(v.numel() for v in state.values()))


def count_sklearn_params(pkl_path: Path) -> int:
    with pkl_path.open("rb") as f:
        bundle = pickle.load(f)
    model = bundle["model"]
    if hasattr(model, "estimators_"):
        return int(sum(getattr(t, "tree_", t).node_count if hasattr(t, "tree_") else 1 for t in model.estimators_))
    if hasattr(model, "get_booster"):
        # XGBoost: number of leaf nodes across trees as a proxy
        b = model.get_booster()
        df = b.trees_to_dataframe()
        return int(len(df))
    return -1


def plot_params_vs_mae() -> None:
    leaderboard = pd.read_csv(ROOT / "models" / "results" / "leaderboard.csv")
    points = []
    for _, row in leaderboard.iterrows():
        name = row["model"]
        mae = row["mae"]
        n = None
        if (CHECKPOINTS / f"{name}.pt").exists():
            n = count_torch_params(CHECKPOINTS / f"{name}.pt")
        elif (CHECKPOINTS / f"{name}.pkl").exists():
            n = count_sklearn_params(CHECKPOINTS / f"{name}.pkl")
        elif name == "arima":
            # 3 per-building ARIMA(2,1,2): order params + scale + intercept ≈ 7 per
            n = 21
        if n is None or n < 0:
            continue
        points.append({"model": name, "params": n, "mae": mae})
    if not points:
        return
    df = pd.DataFrame(points)
    fig, ax = plt.subplots(figsize=(8, 5))
    for _, p in df.iterrows():
        ax.scatter(p["params"], p["mae"], s=180, color=COLORS.get(p["model"], "#666"), edgecolor="white", zorder=3)
        ax.annotate(DISPLAY[p["model"]], (p["params"], p["mae"]),
                    xytext=(8, 4), textcoords="offset points", fontsize=9)
    ax.set_xscale("log")
    ax.set_xlabel("model size (parameters / tree nodes, log scale)")
    ax.set_ylabel("test MAE (kWh)")
    ax.set_title("Model complexity vs. test-set accuracy")
    ax.grid(True, which="both", alpha=0.3)
    ax.invert_yaxis()  # higher = better
    fig.tight_layout()
    fig.savefig(OUT / "params_vs_mae.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="paper")
    plot_system_architecture()
    plot_training_curves()
    plot_residual_histograms()
    plot_hourly_error_heatmap()
    plot_params_vs_mae()
    print(f"wrote 5 report figures to {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
