"""1D Poisson equation: -u_xx = pi^2 sin(pi x), u(0)=u(1)=0."""

from __future__ import annotations

import numpy as np
import torch

from pde_registry import PDEProblem, register_pde

X_MIN, X_MAX = 0.0, 1.0


def _source(x: np.ndarray) -> np.ndarray:
    return (np.pi**2) * np.sin(np.pi * x)


def exact_solution(x: np.ndarray) -> np.ndarray:
    return np.sin(np.pi * x)


def sample_collocation(n: int, seed: int | None = None) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    x = rng.uniform(X_MIN, X_MAX, size=n)
    return torch.tensor(x, dtype=torch.float32).unsqueeze(1)


def sample_boundary(n: int, seed: int | None = None) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    x = rng.choice([X_MIN, X_MAX], size=n)
    return torch.tensor(x, dtype=torch.float32).unsqueeze(1)


def poisson_residual(model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    x = x.requires_grad_(True)
    u = model(x)
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
    src_np = _source(x.detach().cpu().numpy().ravel())
    source = torch.tensor(src_np, dtype=torch.float32, device=x.device).unsqueeze(1)
    return -u_xx - source


def reference_solution(t: np.ndarray, x: np.ndarray | None = None) -> np.ndarray:
    if x is None:
        x = t
    return exact_solution(np.asarray(x))


def evaluation_grid(nx: int = 256) -> torch.Tensor:
    x = np.linspace(X_MIN, X_MAX, nx)
    return torch.tensor(x, dtype=torch.float32).unsqueeze(1)


def sample_observations(n_obs: int, noise_sigma: float, seed: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    rng = np.random.default_rng(seed)
    x = rng.uniform(X_MIN, X_MAX, size=n_obs)
    u = exact_solution(x)
    if noise_sigma > 0:
        u = u + rng.normal(0.0, noise_sigma, size=n_obs)
    tx = torch.tensor(x, dtype=torch.float32).unsqueeze(1)
    u_t = torch.tensor(u, dtype=torch.float32).unsqueeze(1)
    return tx, u_t


def _register() -> None:
    register_pde(
        PDEProblem(
            name="poisson",
            input_dim=1,
            residual_fn=poisson_residual,
            reference_solution=reference_solution,
            evaluation_grid=evaluation_grid,
            sample_collocation=sample_collocation,
            sample_initial=None,
            sample_boundary=sample_boundary,
            ic_target=None,
            sample_observations=sample_observations,
            coord_slices=(slice(None),),
        )
    )


_register()
