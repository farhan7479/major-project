"""Smoke test for the FastAPI service.

Spins through every endpoint and verifies the API's predictions match what
verify_consistency.py reports — same model, same building, same target,
same kWh.

Usage:
  # in one shell: cd energy-forecasting-app/backend && uvicorn main:app
  # in another:   python ml/scripts/smoke_test_api.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import requests

BASE = "http://localhost:8000"
PREDICTIONS = Path(__file__).resolve().parents[1] / "models" / "results" / "predictions"


def main() -> int:
    # /health
    r = requests.get(f"{BASE}/health", timeout=5)
    r.raise_for_status()
    print(f"/health → {r.json()}")

    # /models
    models = requests.get(f"{BASE}/models", timeout=5).json()["models"]
    model_names = [m["name"] for m in models]
    print(f"/models → {model_names}")

    # /buildings
    bld = requests.get(f"{BASE}/buildings", timeout=5).json()
    print(f"/buildings → {bld['buildings']}")
    print(f"   test window: {bld['test_period']['start']} → {bld['test_period']['end']}")

    # /forecast: one call per (model, building) for a mid-test hour
    target = "2017-11-15T14:00:00"
    print(f"\n/forecast on {target}:")
    for building in bld["buildings"]:
        print(f"  {building.replace('Hog_office_', '')}:")
        for model_name in model_names:
            payload = {"model": model_name, "building_id": building, "target_datetime": target}
            r = requests.post(f"{BASE}/forecast", json=payload, timeout=30)
            r.raise_for_status()
            d = r.json()
            print(f"    {model_name:14s}  predicted {d['prediction_kwh']:>7.2f}  (actual {d['actual_kwh']:>7.2f})")

    # /batch-forecast: ask several models in one call
    print(f"\n/batch-forecast on {target} for Sydney:")
    payload = {"models": model_names, "building_id": "Hog_office_Sydney", "target_datetime": target}
    r = requests.post(f"{BASE}/batch-forecast", json=payload, timeout=30)
    r.raise_for_status()
    d = r.json()
    print(f"  actual: {d['actual_kwh']}")
    for k, v in d["predictions"].items():
        print(f"  {k:14s} → {v}")

    # Consistency: pick 5 random stored predictions per model, hit the API, compare
    print("\nconsistency check (API vs stored test predictions):")
    for model_name in model_names:
        stored = pd.read_parquet(PREDICTIONS / f"{model_name}.parquet")
        stored["timestamp"] = pd.to_datetime(stored["timestamp"])
        sample = stored.sample(5, random_state=7)
        max_diff = 0.0
        for _, row in sample.iterrows():
            payload = {
                "model": model_name,
                "building_id": row["building_id"],
                "target_datetime": row["timestamp"].isoformat(),
            }
            r = requests.post(f"{BASE}/forecast", json=payload, timeout=30)
            api_pred = r.json()["prediction_kwh"]
            max_diff = max(max_diff, abs(api_pred - row["y_pred"]))
        flag = "OK " if max_diff < 0.5 else "BAD"
        print(f"  [{flag}] {model_name:14s}  max diff {max_diff:.4f} kWh")

    return 0


if __name__ == "__main__":
    sys.exit(main())
