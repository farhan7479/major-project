"""Single source of truth for loading + serving every trained model.

`ModelRegistry()` is instantiated once at FastAPI startup. It loads every
checkpoint, every saved ARIMA result, and the inference artifacts. It exposes
one `.predict(model_name, building_id, target_ts, weather_override=None)` call
that dispatches to the right backend and returns kWh.
"""

from __future__ import annotations

import pickle
import warnings
from pathlib import Path
from typing import Optional

import pandas as pd
import torch

from .dl_models import (
    BiLSTMForecaster,
    CNNLSTMForecaster,
    LSTMForecaster,
    TransformerForecaster,
)
from .feature_pipeline import (
    Artifacts,
    build_sequence,
    build_tabular_row,
    consumption_series,
    denormalize_target,
)

DL_CLASSES = {
    "lstm": LSTMForecaster,
    "bilstm": BiLSTMForecaster,
    "cnn_lstm": CNNLSTMForecaster,
    "transformer": TransformerForecaster,
}

MODEL_KIND = {
    "random_forest": "tabular",
    "xgboost": "tabular",
    "arima": "arima",
    "lstm": "sequence",
    "bilstm": "sequence",
    "cnn_lstm": "sequence",
    "transformer": "sequence",
}

MODEL_DISPLAY = {
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
    "arima": "ARIMA(2,1,2) — per building",
    "lstm": "LSTM",
    "bilstm": "BiLSTM",
    "cnn_lstm": "CNN-LSTM",
    "transformer": "Transformer encoder",
}


def _pick_device() -> torch.device:
    # Inference always runs on CPU. MPS gives non-deterministic output across
    # Python processes for nn.LSTM, which breaks consistency between what the
    # training scripts wrote to predictions/*.parquet and what the API serves.
    # CPU inference on these models is fast enough (<10 ms per request).
    return torch.device("cpu")


class ModelRegistry:
    def __init__(self, artifacts_dir: Path, checkpoints_dir: Path):
        self.art = Artifacts.load(artifacts_dir)
        self.checkpoints_dir = checkpoints_dir
        self.device = _pick_device()

        self.tabular_models: dict[str, dict] = {}
        self.sequence_models: dict[str, torch.nn.Module] = {}
        self.arima_models: dict[str, dict] = {}

        self._load_tabular("random_forest")
        self._load_tabular("xgboost")
        self._load_sequence_models()
        self._load_arima()

    def _load_tabular(self, name: str) -> None:
        path = self.checkpoints_dir / f"{name}.pkl"
        with path.open("rb") as f:
            self.tabular_models[name] = pickle.load(f)

    def _load_sequence_models(self) -> None:
        n_features = len(self.art.manifest["buildings"]) + len(self.art.scalers["timestep_features"])
        for name, cls in DL_CLASSES.items():
            path = self.checkpoints_dir / f"{name}.pt"
            if not path.exists():
                continue
            model = cls(input_dim=n_features).to(self.device)
            state = torch.load(path, map_location=self.device, weights_only=True)
            model.load_state_dict(state)
            model.eval()
            self.sequence_models[name] = model

    def _load_arima(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for bid in self.art.manifest["buildings"]:
                path = self.checkpoints_dir / f"arima_{bid}.pkl"
                with path.open("rb") as f:
                    fitted = pickle.load(f)
                full_series = consumption_series(self.art, bid)
                extended = fitted.apply(full_series, refit=False)
                self.arima_models[bid] = {"fitted": fitted, "extended": extended}

    # ----- public API -----

    def list_models(self) -> list[dict]:
        out = []
        for name, kind in MODEL_KIND.items():
            available = (
                name in self.tabular_models
                or name in self.sequence_models
                or (kind == "arima" and len(self.arima_models) == len(self.art.manifest["buildings"]))
            )
            if not available:
                continue
            out.append({"name": name, "display_name": MODEL_DISPLAY[name], "kind": kind})
        return out

    def list_buildings(self) -> list[str]:
        return list(self.art.manifest["buildings"])

    def test_period(self) -> dict:
        return self.art.manifest["test_period"]

    def predict(
        self,
        model_name: str,
        building_id: str,
        target_ts: pd.Timestamp,
        weather_override: Optional[dict] = None,
    ) -> float:
        if building_id not in self.art.manifest["buildings"]:
            raise ValueError(f"unknown building '{building_id}'")
        kind = MODEL_KIND.get(model_name)
        if kind is None:
            raise ValueError(f"unknown model '{model_name}'")

        if kind == "tabular":
            return self._predict_tabular(model_name, building_id, target_ts, weather_override)
        if kind == "sequence":
            return self._predict_sequence(model_name, building_id, target_ts, weather_override)
        return self._predict_arima(building_id, target_ts)

    def _predict_tabular(self, name: str, bid: str, ts: pd.Timestamp, override) -> float:
        bundle = self.tabular_models[name]
        row = build_tabular_row(self.art, bid, ts, weather_override=override)
        X = row[bundle["feature_cols"]].to_numpy().reshape(1, -1)
        return float(bundle["model"].predict(X)[0])

    def _predict_sequence(self, name: str, bid: str, ts: pd.Timestamp, override) -> float:
        del override  # DL models only see past timesteps, target-hour overrides don't apply
        model = self.sequence_models[name]
        seq = build_sequence(self.art, bid, ts)
        tensor = torch.from_numpy(seq).to(self.device)
        with torch.no_grad():
            y_norm = model(tensor).squeeze().item()
        return denormalize_target(y_norm, bid, self.art.scalers)

    def _predict_arima(self, bid: str, ts: pd.Timestamp) -> float:
        extended = self.arima_models[bid]["extended"]
        return float(extended.predict(start=ts, end=ts, dynamic=False).iloc[0])

    def actual_consumption(self, building_id: str, target_ts: pd.Timestamp) -> Optional[float]:
        """Look up the ground-truth value if we have it (test-set hours)."""
        key = (building_id, target_ts)
        if key in self.art.context.index:
            return float(self.art.context.loc[key, "consumption"])
        return None
