"""Train a single PINN candidate for given loss weights and data budget."""

from __future__ import annotations

import torch
import torch.nn as nn

from adaptive_balancers import GradNormBalancer, ReLoBRaLoBalancer
from objectives import evaluate_objectives
from pde_registry import get_pde
import pde_burgers  # noqa: F401 — register PDEs
import pde_cylinder_wake  # noqa: F401 — register PDEs (real experimental cylinder wake)
import pde_poisson  # noqa: F401
from pinn import PINN


def train_pinn(
    pde_name: str = "burgers",
    lam_data: float = 1.0,
    lam_pde: float = 1.0,
    n_collocation: int = 200,
    n_obs: int = 0,
    obs_noise: float = 0.0,
    epochs: int = 3000,
    lr: float = 1e-3,
    n_ic: int = 100,
    n_bc: int = 100,
    hidden_dim: int = 50,
    n_layers: int = 4,
    balancer: str | None = None,
    seed: int = 42,
    device: str | torch.device | None = None,
    verbose: bool = False,
) -> tuple[PINN, dict[str, float]]:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    torch.manual_seed(seed)
    pde = get_pde(pde_name)
    n_collocation = int(max(20, n_collocation))
    n_obs = int(max(0, n_obs))

    model = PINN(input_dim=pde.input_dim, hidden_dim=hidden_dim, n_layers=n_layers, output_dim=pde.output_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mse = nn.MSELoss()

    tx_coll = pde.sample_collocation(n_collocation, seed=seed).to(device)
    tx_bc = pde.sample_boundary(n_bc, seed=seed + 2).to(device)

    tx_ic = None
    u_ic = None
    if pde.sample_initial is not None and pde.ic_target is not None:
        tx_ic = pde.sample_initial(n_ic, seed=seed + 1).to(device)
        u_ic = pde.ic_target(tx_ic).to(device)

    tx_obs, u_obs = None, None
    if n_obs > 0:
        tx_obs, u_obs = pde.sample_observations(n_obs, obs_noise, seed=seed + 3)
        tx_obs, u_obs = tx_obs.to(device), u_obs.to(device)

    grad_balancer = GradNormBalancer() if balancer == "gradnorm" else None
    relo_balancer = ReLoBRaLoBalancer() if balancer == "relobralo" else None

    shared_params = list(model.parameters())

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        loss_pde = pde.residual_fn(model, tx_coll).pow(2).mean()

        data_terms = []
        if tx_ic is not None and u_ic is not None:
            data_terms.append(mse(model(tx_ic), u_ic))
        data_terms.append(mse(model(tx_bc), torch.zeros(tx_bc.shape[0], pde.output_dim, device=device)))
        if tx_obs is not None and u_obs is not None:
            data_terms.append(mse(model(tx_obs), u_obs))
        loss_data = sum(data_terms) / len(data_terms)

        if grad_balancer is not None:
            loss = grad_balancer.update([loss_pde, loss_data], shared_params)
        elif relo_balancer is not None:
            loss = relo_balancer.update(loss_pde, loss_data)
        else:
            loss = lam_pde * loss_pde + lam_data * loss_data

        loss.backward()
        optimizer.step()

        if verbose and (epoch + 1) % 500 == 0:
            print(f"epoch {epoch + 1}/{epochs} loss={loss.item():.4e}")

    metrics = evaluate_objectives(model, pde_name, n_collocation, tx_coll, device, n_obs=n_obs)
    metrics.update(
        {
            "pde": pde_name,
            "lam_data": float(lam_data),
            "lam_pde": float(lam_pde),
            "n_collocation": float(n_collocation),
            "n_obs": float(n_obs),
            "obs_noise": float(obs_noise),
            "balancer": balancer or "fixed",
            "hidden_dim": float(hidden_dim),
            "n_layers": float(n_layers),
        }
    )
    return model, metrics
