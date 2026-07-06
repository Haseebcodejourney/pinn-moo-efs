"""Physics-informed neural network for spatio-temporal inputs.

The PINN is a small MLP.  For 1D problems (Burgers, Poisson) it produces a
single scalar per input; for multi-output problems (cylinder wake: u, v, p)
``output_dim`` is set to the number of physical fields and the final layer
is widened accordingly.

The default ``output_dim=1`` keeps the original single-output behaviour,
so existing PDEs (Burgers, Poisson, 2D Poisson) work without any change.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class PINN(nn.Module):
    def __init__(
        self,
        input_dim: int = 2,
        hidden_dim: int = 50,
        n_layers: int = 4,
        output_dim: int = 1,
    ):
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(input_dim, hidden_dim), nn.Tanh()]
        for _ in range(n_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.Tanh()])
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, tx: torch.Tensor) -> torch.Tensor:
        return self.net(tx)
