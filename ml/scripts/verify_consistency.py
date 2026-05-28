"""Sanity check: reload a trained model from disk and confirm its predictions
match the stored predictions parquet within FP tolerance.

If they don't, the saved model doesn't reflect the metric the leaderboard
reports — usually a Save/load bug somewhere upstream. Run this after every
training session.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve.model_registry import ModelRegistry  # noqa: E402

ARTIFACTS = Path(__file__).resolve().parents[1] / "serve" / "artifacts"
CHECKPOINTS = Path(__file__).resolve().parents[1] / "models" / "checkpoints"
PREDICTIONS = Path(__file__).resolve().parents[1] / "models" / "results" / "predictions"

TOLERANCE_KWH = 0.5  # tree models have small FP noise; >0.5 indicates a real divergence


def verify(reg: ModelRegistry, model_name: str, sample_size: int = 30) -> tuple[bool, float, float]:
    pred_path = PREDICTIONS / f"{model_name}.parquet"
    if not pred_path.exists():
        print(f"  {model_name}: no stored predictions, skipping")
        return True, 0.0, 0.0
    stored = pd.read_parquet(pred_path)
    stored["timestamp"] = pd.to_datetime(stored["timestamp"])
    sample = stored.sample(sample_size, random_state=42).reset_index(drop=True)
    diffs = []
    for _, row in sample.iterrows():
        live = reg.predict(model_name, row["building_id"], row["timestamp"])
        diffs.append(abs(live - row["y_pred"]))
    diffs = np.array(diffs)
    max_diff = float(diffs.max())
    mean_diff = float(diffs.mean())
    ok = max_diff < TOLERANCE_KWH
    flag = "OK " if ok else "BAD"
    print(f"  [{flag}] {model_name:14s}  max diff {max_diff:7.4f} kWh  mean diff {mean_diff:7.4f} kWh")
    return ok, max_diff, mean_diff


def main() -> int:
    reg = ModelRegistry(ARTIFACTS, CHECKPOINTS)
    sample_size = 30
    print(f"verifying {sample_size} random test samples per model\n")
    all_ok = True
    for m in reg.list_models():
        ok, _, _ = verify(reg, m["name"], sample_size=sample_size)
        all_ok = all_ok and ok
    print(f"\noverall: {'all consistent' if all_ok else 'INCONSISTENT — saved model does not match stored predictions'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
