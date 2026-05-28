"""CNN-LSTM hybrid.

A 1-D convolutional front-end picks up local patterns (e.g. sharp ramps in
consumption), the LSTM stack then models the longer-range temporal
dependencies on the convolved sequence.
"""

from __future__ import annotations

import sys
from pathlib import Path

from torch import nn

from torch_data import build_splits
from torch_train import TrainConfig, persist_predictions_and_metrics, train_model

CHECKPOINT_DIR = Path(__file__).resolve().parents[1] / "models" / "checkpoints"


class CNNLSTMForecaster(nn.Module):
    def __init__(self, input_dim: int, conv_channels: int = 64, hidden_dim: int = 64, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(input_dim, conv_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(conv_channels, conv_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
        )
        self.lstm = nn.LSTM(
            input_size=conv_channels,
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
        # Conv1d wants (B, F, T)
        conv_out = self.conv(x.transpose(1, 2)).transpose(1, 2)  # back to (B, T', C)
        out, _ = self.lstm(conv_out)
        return self.head(out[:, -1])


def main() -> int:
    train_b, val_b, test_b, scalers, _ = build_splits()
    input_dim = train_b.X.shape[-1]
    config = TrainConfig(model_name="cnn_lstm", batch_size=256, epochs=40, lr=1e-3, patience=6)
    _, test_preds = train_model(lambda: CNNLSTMForecaster(input_dim=input_dim),
                                train_b, val_b, test_b, scalers, config, CHECKPOINT_DIR)
    persist_predictions_and_metrics("cnn_lstm", test_b, test_preds, scalers)
    return 0


if __name__ == "__main__":
    sys.exit(main())
