# Derived from https://github.com/wmcnally/golfdb (model.py) — CPU-safe LSTM init, no mobilenet sidecar.

from __future__ import annotations

import torch
import torch.nn as nn

from lib.golfdb_swingnet.mobilenet_v2 import MobileNetV2


class EventDetector(nn.Module):
    """SwingNet hybrid CNN + BiLSTM for 9-way per-frame golf event logits (8 events + background)."""

    def __init__(
        self,
        *,
        width_mult: float = 1.0,
        lstm_layers: int = 1,
        lstm_hidden: int = 256,
        bidirectional: bool = True,
        dropout: bool = False,
    ):
        super().__init__()
        self.width_mult = width_mult
        self.lstm_layers = lstm_layers
        self.lstm_hidden = lstm_hidden
        self.bidirectional = bidirectional
        self.use_dropout = dropout

        net = MobileNetV2(width_mult=width_mult)
        feats = list(net.features.children())
        self.cnn = nn.Sequential(*feats[:19])

        cnn_out = int(1280 * width_mult) if width_mult > 1.0 else 1280
        self.rnn = nn.LSTM(
            cnn_out,
            lstm_hidden,
            lstm_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )
        if bidirectional:
            self.lin = nn.Linear(2 * lstm_hidden, 9)
        else:
            self.lin = nn.Linear(lstm_hidden, 9)
        self.drop = nn.Dropout(0.5) if dropout else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, timesteps, c, h, w = x.size()
        c_in = x.reshape(batch_size * timesteps, c, h, w)
        c_out = self.cnn(c_in)
        c_out = c_out.mean(3).mean(2)
        if self.drop is not None:
            c_out = self.drop(c_out)
        r_in = c_out.view(batch_size, timesteps, -1)
        r_out, _ = self.rnn(r_in)
        out = self.lin(r_out)
        return out.reshape(batch_size * timesteps, 9)
