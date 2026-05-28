"""Exploratory analysis on the three selected Hog-site office buildings.

Writes summary statistics to stdout and saves figures to
models/results/eda/ for use in the report.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

PROCESSED = Path(__file__).resolve().parents[1] / "data" / "processed"
FIG_DIR = Path(__file__).resolve().parents[1] / "models" / "results" / "eda"

sns.set_theme(style="whitegrid", context="paper")


def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    elec = pd.read_parquet(PROCESSED / "electricity_selected.parquet")
    weather = pd.read_parquet(PROCESSED / "weather_selected.parquet")
    return elec, weather


def print_summary(elec: pd.DataFrame) -> None:
    print("=" * 60)
    print("ELECTRICITY SUMMARY (kWh per hour, per building)")
    print("=" * 60)
    print(elec.describe().round(2).to_string())
    print(f"\ntime span: {elec.index.min()} to {elec.index.max()}")
    print(f"observations per building: {len(elec)}")


def plot_distribution(elec: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, len(elec.columns), figsize=(4 * len(elec.columns), 3.5), sharey=True)
    for ax, col in zip(axes, elec.columns):
        ax.hist(elec[col].dropna(), bins=60, color="#3a6ea5", alpha=0.85)
        ax.set_title(col.split("_", 1)[1])
        ax.set_xlabel("kWh / hr")
    axes[0].set_ylabel("hours")
    fig.suptitle("Hourly consumption distribution", y=1.02)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "consumption_distribution.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_daily_pattern(elec: pd.DataFrame) -> None:
    by_hour = elec.groupby(elec.index.hour).mean()
    fig, ax = plt.subplots(figsize=(8, 4))
    by_hour.plot(ax=ax, marker="o")
    ax.set_xlabel("hour of day")
    ax.set_ylabel("mean consumption (kWh)")
    ax.set_title("Average consumption by hour of day")
    ax.set_xticks(range(0, 24, 2))
    ax.legend([c.split("_", 1)[1] for c in by_hour.columns], loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "daily_pattern.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_weekly_pattern(elec: pd.DataFrame) -> None:
    days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    by_dow = elec.groupby(elec.index.dayofweek).mean()
    fig, ax = plt.subplots(figsize=(8, 4))
    by_dow.plot(kind="bar", ax=ax)
    ax.set_xticklabels(days, rotation=0)
    ax.set_ylabel("mean consumption (kWh)")
    ax.set_title("Average consumption by day of week")
    ax.legend([c.split("_", 1)[1] for c in by_dow.columns], loc="lower left", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "weekly_pattern.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_monthly_pattern(elec: pd.DataFrame) -> None:
    by_month = elec.groupby(elec.index.month).mean()
    fig, ax = plt.subplots(figsize=(8, 4))
    by_month.plot(marker="o", ax=ax)
    ax.set_xlabel("month")
    ax.set_ylabel("mean consumption (kWh)")
    ax.set_title("Average consumption by month")
    ax.set_xticks(range(1, 13))
    ax.legend([c.split("_", 1)[1] for c in by_month.columns], loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "monthly_pattern.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_one_week(elec: pd.DataFrame) -> None:
    start = pd.Timestamp("2016-06-06")
    week = elec.loc[start:start + pd.Timedelta(days=7)]
    fig, ax = plt.subplots(figsize=(11, 4))
    week.plot(ax=ax)
    ax.set_ylabel("kWh / hr")
    ax.set_title(f"One week of consumption ({start.date()} to {(start + pd.Timedelta(days=7)).date()})")
    ax.legend([c.split("_", 1)[1] for c in week.columns], loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "sample_week.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_weather_correlation(elec: pd.DataFrame, weather: pd.DataFrame) -> None:
    primary = elec.columns[0]
    joined = elec[[primary]].join(weather.drop(columns=["site_id"]), how="inner")
    corr = joined.corr().iloc[1:, [0]].rename(columns={primary: "consumption"})
    fig, ax = plt.subplots(figsize=(4.5, 4))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0, vmin=-1, vmax=1, ax=ax)
    ax.set_title(f"Weather correlation\n({primary.split('_', 1)[1]})")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "weather_correlation.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    elec, weather = load()
    print_summary(elec)
    plot_distribution(elec)
    plot_daily_pattern(elec)
    plot_weekly_pattern(elec)
    plot_monthly_pattern(elec)
    plot_one_week(elec)
    plot_weather_correlation(elec, weather)
    print(f"\nfigures saved to {FIG_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
