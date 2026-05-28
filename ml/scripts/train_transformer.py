"""Transformer encoder forecaster (attention-based, in the spirit of TFT).

The full Temporal Fusion Transformer adds variable-selection networks,
static covariate encoders, and quantile output heads. We omit those for
this dataset size (36k sequences) where they'd be over-parameterized,
keeping just the multi-head self-attention encoder over the past sequence.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch
from torch import nn

from torch_data import build_splits
from torch_train import TrainConfig, persist_predictions_and_metrics, train_model

CHECKPOINT_DIR = Path(__file__).resolve().parents[1] / "models" / "checkpoints"


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 1024):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class TransformerForecaster(nn.Module):
    def __init__(
        self,
        input_dim: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, x):  # (B, T, F)
        x = self.input_proj(x)
        x = self.pos(x)
        x = self.encoder(x)
        return self.head(x[:, -1])  # last token -> forecast


def main() -> int:
    train_b, val_b, test_b, scalers, _ = build_splits()
    input_dim = train_b.X.shape[-1]
    config = TrainConfig(model_name="transformer", batch_size=256, epochs=40, lr=5e-4, patience=6)
    _, test_preds = train_model(lambda: TransformerForecaster(input_dim=input_dim),
                                train_b, val_b, test_b, scalers, config, CHECKPOINT_DIR)
    persist_predictions_and_metrics("transformer", test_b, test_preds, scalers)
    return 0


if __name__ == "__main__":
    sys.exit(main())
