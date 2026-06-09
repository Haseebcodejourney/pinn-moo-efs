"""Physics-informed neural network for spatio-temporal inputs (t, x)."""

from __future__ import annotations

import torch
import torch.nn as nn


class PINN(nn.Module):
    def __init__(self, input_dim: int = 2, hidden_dim: int = 50, n_layers: int = 4):
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(input_dim, hidden_dim), nn.Tanh()]
        for _ in range(n_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.Tanh()])
        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, tx: torch.Tensor) -> torch.Tensor:
        return self.net(tx)
