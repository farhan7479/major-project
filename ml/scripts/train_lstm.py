"""2-layer LSTM forecaster.

Input  : (batch, 168, 17) — last week of per-hour features
Output : scalar consumption at the next hour (z-scored, de-normed for eval)
"""

from __future__ import annotations

import sys
from pathlib import Path

from torch import nn

from torch_data import build_splits
from torch_train import TrainConfig, persist_predictions_and_metrics, train_model

CHECKPOINT_DIR = Path(__file__).resolve().parents[1] / "models" / "checkpoints"


class LSTMForecaster(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x):  # (B, T, F)
        out, _ = self.lstm(x)
        return self.head(out[:, -1])  # last timestep -> prediction


def main() -> int:
    train_b, val_b, test_b, scalers, _ = build_splits()
    input_dim = train_b.X.shape[-1]
    config = TrainConfig(model_name="lstm", batch_size=256, epochs=40, lr=1e-3, patience=6)
    _, test_preds = train_model(lambda: LSTMForecaster(input_dim=input_dim),
                                train_b, val_b, test_b, scalers, config, CHECKPOINT_DIR)
    persist_predictions_and_metrics("lstm", test_b, test_preds, scalers)
    return 0


if __name__ == "__main__":
    sys.exit(main())
