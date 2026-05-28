"""Pick 3 office buildings from BDG2 with low missingness and shared weather site.

Strategy: rank all Office-primary buildings by % non-null hours in the
electricity file, restrict to a single site so a single weather frame
covers all three, and write the chosen subset to data/processed/.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
OUT = Path(__file__).resolve().parents[1] / "data" / "processed"

N_BUILDINGS = 3
MIN_COVERAGE = 0.98  # fraction of non-null hours required


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)

    meta = pd.read_csv(RAW / "metadata.csv")
    offices = meta[meta["primaryspaceusage"] == "Office"]["building_id"].tolist()
    print(f"office buildings in metadata: {len(offices)}")

    elec = pd.read_csv(RAW / "electricity_cleaned.csv", parse_dates=["timestamp"])
    elec = elec.set_index("timestamp")

    office_cols = [c for c in elec.columns if c in offices]
    print(f"office buildings with electricity series: {len(office_cols)}")

    coverage = elec[office_cols].notna().mean().sort_values(ascending=False)
    eligible = coverage[coverage >= MIN_COVERAGE].index.tolist()
    print(f"buildings with >= {MIN_COVERAGE:.0%} coverage: {len(eligible)}")

    # Each column starts with "<site>_..."; pick the site with the most eligible
    # buildings so the three we keep all share a single weather frame.
    eligible_meta = meta[meta["building_id"].isin(eligible)].copy()
    site_counts = eligible_meta["site_id"].value_counts()
    print("\neligible by site (top 5):")
    print(site_counts.head())

    chosen_site = site_counts.index[0]
    site_buildings = (
        eligible_meta[eligible_meta["site_id"] == chosen_site]
        .merge(coverage.rename("coverage"), left_on="building_id", right_index=True)
        .sort_values("coverage", ascending=False)
        .head(N_BUILDINGS)
    )

    print(f"\nchosen site: {chosen_site}")
    print(site_buildings[["building_id", "sqm", "yearbuilt", "coverage"]].to_string(index=False))

    chosen_ids = site_buildings["building_id"].tolist()
    site_buildings.to_csv(OUT / "selected_buildings.csv", index=False)

    # Save the chosen buildings' electricity series + site weather for downstream steps
    elec[chosen_ids].to_parquet(OUT / "electricity_selected.parquet")

    weather = pd.read_csv(RAW / "weather.csv", parse_dates=["timestamp"])
    site_weather = weather[weather["site_id"] == chosen_site].set_index("timestamp")
    site_weather.to_parquet(OUT / "weather_selected.parquet")
    print(f"\nweather rows for site {chosen_site}: {len(site_weather)}")
    print(f"weather columns: {list(site_weather.columns)}")

    print(f"\nwrote: {OUT}/selected_buildings.csv, electricity_selected.parquet, weather_selected.parquet")
    return 0


if __name__ == "__main__":
    sys.exit(main())
