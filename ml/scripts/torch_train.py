"""Shared training loop for PyTorch forecasters.

Adam + MSE on z-scored targets, early stopping on val MAE in de-normalized
kWh space (so the stopping criterion matches what the leaderboard reports),
ReduceLROnPlateau on val MAE.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from eval_utils import (
    evaluate_predictions,
    get_train_targets_by_building,
    load_features_and_splits,
    save_metrics,
    save_predictions,
)
from torch_data import Scalers, SplitBundle, WindowDataset, denormalize


def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@dataclass
class TrainConfig:
    model_name: str
    batch_size: int = 256
    epochs: int = 40
    lr: float = 1e-3
    weight_decay: float = 1e-5
    patience: int = 6
    grad_clip: float = 1.0


def _eval_mae_kwh(model: nn.Module, bundle: SplitBundle, scalers: Scalers, device: torch.device, batch_size: int) -> tuple[float, np.ndarray]:
    model.eval()
    preds_norm = []
    loader = DataLoader(WindowDataset(bundle), batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for xb, _ in loader:
            xb = xb.to(device)
            out = model(xb).squeeze(-1).cpu().numpy()
            preds_norm.append(out)
    preds_norm = np.concatenate(preds_norm)
    preds_kwh = denormalize(preds_norm, bundle.building_id, scalers)
    truth_kwh = denormalize(bundle.y.numpy(), bundle.building_id, scalers)
    return float(np.mean(np.abs(preds_kwh - truth_kwh))), preds_kwh


def train_model(
    model_factory,
    train_bundle: SplitBundle,
    val_bundle: SplitBundle,
    test_bundle: SplitBundle,
    scalers: Scalers,
    config: TrainConfig,
    checkpoint_dir: Path,
) -> tuple[float, np.ndarray]:
    device = pick_device()
    print(f"device: {device}")
    model = model_factory().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=2, factor=0.5)
    criterion = nn.MSELoss()

    train_loader = DataLoader(WindowDataset(train_bundle), batch_size=config.batch_size, shuffle=True, drop_last=True)

    best_val_mae = float("inf")
    best_state = None
    bad_epochs = 0
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, config.epochs + 1):
        t0 = time.time()
        model.train()
        running_loss = 0.0
        n = 0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            pred = model(xb).squeeze(-1)
            loss = criterion(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
            running_loss += loss.item() * len(yb)
            n += len(yb)
        train_loss = running_loss / n
        val_mae, _ = _eval_mae_kwh(model, val_bundle, scalers, device, config.batch_size)
        scheduler.step(val_mae)

        improved = val_mae < best_val_mae - 1e-4
        if improved:
            best_val_mae = val_mae
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1

        dt = time.time() - t0
        flag = " *best*" if improved else ""
        print(f"epoch {epoch:>2d}  train_loss={train_loss:.4f}  val_MAE={val_mae:.3f}  ({dt:.1f}s){flag}")
        if bad_epochs >= config.patience:
            print(f"early stopping at epoch {epoch} (no improvement in {config.patience} epochs)")
            break

    assert best_state is not None
    checkpoint_path = checkpoint_dir / f"{config.model_name}.pt"
    torch.save(best_state, checkpoint_path)
    # Final test eval on CPU. MPS gives non-deterministic output across Python
    # processes for nn.LSTM, which means the API loading the same checkpoint
    # would report different predictions than what the training run wrote to
    # predictions.parquet. CPU is slower but reproducible, so the leaderboard
    # numbers match what the API serves.
    cpu = torch.device("cpu")
    fresh = model_factory().to(cpu)
    fresh.load_state_dict(torch.load(checkpoint_path, map_location=cpu, weights_only=True))
    fresh.eval()
    test_mae, test_preds_kwh = _eval_mae_kwh(fresh, test_bundle, scalers, cpu, config.batch_size)
    print(f"\nfinal val_MAE={best_val_mae:.3f}  test_MAE={test_mae:.3f}")
    return test_mae, test_preds_kwh


def persist_predictions_and_metrics(
    model_name: str,
    test_bundle: SplitBundle,
    test_preds_kwh: np.ndarray,
    scalers: Scalers,
) -> dict:
    truth_kwh = denormalize(test_bundle.y.numpy(), test_bundle.building_id, scalers)
    preds_df = pd.DataFrame({
        "building_id": test_bundle.building_id,
        "timestamp": test_bundle.timestamp,
        "y_true": truth_kwh,
        "y_pred": test_preds_kwh,
    })
    features, split = load_features_and_splits()
    y_train_by_building = get_train_targets_by_building(features, split)
    metrics = evaluate_predictions(preds_df, y_train_by_building)
    save_predictions(model_name, preds_df)
    save_metrics(model_name, metrics)

    print(f"\n=== {model_name} ===")
    print(f"macro MAE:  {metrics['macro']['mae']:.2f}")
    print(f"macro RMSE: {metrics['macro']['rmse']:.2f}")
    print(f"macro MAPE: {metrics['macro']['mape']:.2f}%")
    print(f"macro R2:   {metrics['macro']['r2']:.3f}")
    print(f"peak F1:    {metrics['macro']['peak_f1']:.3f}")
    return metrics
