"""Optional scalar adaptive loss balancers for PINN training."""

from __future__ import annotations

import torch


class GradNormBalancer:
    """Simplified GradNorm-style adaptive weights on {PDE, data} losses."""

    def __init__(self, alpha: float = 1.5, lr: float = 0.025):
        self.alpha = alpha
        self.lr = lr
        self.weights = torch.tensor([1.0, 1.0], dtype=torch.float32)
        self.initial_losses: list[float] | None = None

    def update(self, losses: list[torch.Tensor], shared_params: list[torch.nn.Parameter]) -> torch.Tensor:
        if self.initial_losses is None:
            self.initial_losses = [float(l.detach().item()) for l in losses]

        grads = []
        for loss in losses:
            g = torch.autograd.grad(loss, shared_params, retain_graph=True, allow_unused=True)
            flat = torch.cat([gi.reshape(-1) for gi in g if gi is not None])
            grads.append(flat.norm())

        mean_grad = torch.stack(grads).mean() + 1e-8
        targets = []
        for i, loss in enumerate(losses):
            ratio = (loss.detach() / (self.initial_losses[i] + 1e-8)) ** self.alpha
            targets.append(mean_grad * ratio)

        with torch.no_grad():
            for i in range(len(self.weights)):
                self.weights[i] *= float((targets[i] / (grads[i] + 1e-8)).clamp(0.5, 2.0))
            self.weights = self.weights / self.weights.sum() * len(self.weights)

        return (self.weights[0] * losses[0] + self.weights[1] * losses[1])


class ReLoBRaLoBalancer:
    """Random lookback rebalancing of loss weights."""

    def __init__(self, temperature: float = 1.0, lookback: int = 10):
        self.temperature = temperature
        self.lookback = lookback
        self.history: list[tuple[float, float]] = []

    def update(self, loss_pde: torch.Tensor, loss_data: torch.Tensor) -> torch.Tensor:
        self.history.append((float(loss_pde.detach()), float(loss_data.detach())))
        if len(self.history) < 2:
            return loss_pde + loss_data

        window = self.history[-self.lookback :]
        pde_vals = torch.tensor([h[0] for h in window])
        data_vals = torch.tensor([h[1] for h in window])
        rho = (pde_vals.std() + 1e-8) / (data_vals.std() + 1e-8)
        lam = torch.sigmoid(rho / self.temperature)
        return lam * loss_pde + (1.0 - lam) * loss_data
