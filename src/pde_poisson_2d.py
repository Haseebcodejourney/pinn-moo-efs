"""2D Poisson equation on the unit square.

PDE:  -u_xx - u_yy = f(x, y),   (x, y) in (0, 1) x (0, 1)
BC:   u = 0  on the four edges

Reference solution: u(x, y) = sin(pi x) sin(pi y)
                     => -Laplacian u = 2 pi^2 sin(pi x) sin(pi y)

This is the second benchmark the professor asked for ("move to at least
one, preferably two, 2D problems").  Burgers is registered with
input_dim=2 but is 1D-spatial + 1D-temporal; this problem is genuinely
2D in space.
"""

from __future__ import annotations

import numpy as np
import torch

from pde_registry import PDEProblem, register_pde

X_MIN, X_MAX = 0.0, 1.0
Y_MIN, Y_MAX = 0.0, 1.0


# --------------------------------------------------------------------------
# Reference solution and source
# --------------------------------------------------------------------------

def _source(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """f(x, y) = 2 pi^2 sin(pi x) sin(pi y)."""
    return (2.0 * np.pi**2) * np.sin(np.pi * x) * np.sin(np.pi * y)


def exact_solution(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """u(x, y) = sin(pi x) sin(pi y)."""
    return np.sin(np.pi * x) * np.sin(np.pi * y)


# --------------------------------------------------------------------------
# PDE residual at collocation points (autograd through the model)
# --------------------------------------------------------------------------

def poisson2d_residual(model: torch.nn.Module, xy: torch.Tensor) -> torch.Tensor:
    """residual = u_xx + u_yy + 2 pi^2 sin(pi x) sin(pi y).

    Convention: zero residual means the model satisfies -u_xx - u_yy = f.
    """
    xy = xy.requires_grad_(True)
    u = model(xy)

    grads = torch.autograd.grad(
        u, xy, grad_outputs=torch.ones_like(u), create_graph=True
    )[0]
    u_x = grads[:, 0:1]
    u_y = grads[:, 1:2]

    u_xx = torch.autograd.grad(u_x, xy, grad_outputs=torch.ones_like(u_x), create_graph=True)[0][:, 0:1]
    u_yy = torch.autograd.grad(u_y, xy, grad_outputs=torch.ones_like(u_y), create_graph=True)[0][:, 1:2]

    src_np = _source(xy[:, 0].detach().cpu().numpy(), xy[:, 1].detach().cpu().numpy())
    source = torch.tensor(src_np, dtype=torch.float32, device=xy.device).unsqueeze(1)

    return u_xx + u_yy + source


# --------------------------------------------------------------------------
# Sampling helpers
# --------------------------------------------------------------------------

def sample_collocation(n: int, seed: int | None = None) -> torch.Tensor:
    """Latin-hypercube-ish: uniform random in (0, 1)^2."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(X_MIN, X_MAX, size=n)
    y = rng.uniform(Y_MIN, Y_MAX, size=n)
    return torch.tensor(np.stack([x, y], axis=1), dtype=torch.float32)


def sample_boundary(n: int, seed: int | None = None) -> torch.Tensor:
    """Sample the four edges of the unit square.

    Distributes roughly n/4 points on each edge, with one corner shared
    between the two halves for compatibility with how train.py aggregates
    boundary MSE.
    """
    rng = np.random.default_rng(seed)
    per_edge = max(2, n // 4)

    pts = []
    # bottom y = 0, x in [0, 1]
    x = rng.uniform(X_MIN, X_MAX, size=per_edge)
    y = np.zeros_like(x)
    pts.append(np.stack([x, y], axis=1))
    # top y = 1
    x = rng.uniform(X_MIN, X_MAX, size=per_edge)
    y = np.ones_like(x)
    pts.append(np.stack([x, y], axis=1))
    # left x = 0, y in [0, 1]
    x = np.zeros(per_edge)
    y = rng.uniform(Y_MIN, Y_MAX, size=per_edge)
    pts.append(np.stack([x, y], axis=1))
    # right x = 1
    x = np.ones(per_edge)
    y = rng.uniform(Y_MIN, Y_MAX, size=per_edge)
    pts.append(np.stack([x, y], axis=1))

    xy = np.concatenate(pts, axis=0)
    return torch.tensor(xy, dtype=torch.float32)


def reference_solution(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Wrapped for the registry: maps (x, y) -> u(x, y)."""
    return exact_solution(np.asarray(x), np.asarray(y))


def evaluation_grid(nx: int = 64, ny: int = 64) -> torch.Tensor:
    x = np.linspace(X_MIN, X_MAX, nx)
    y = np.linspace(Y_MIN, Y_MAX, ny)
    xx, yy = np.meshgrid(x, y, indexing="xy")
    return torch.tensor(np.stack([xx.ravel(), yy.ravel()], axis=1), dtype=torch.float32)


def sample_observations(
    n_obs: int, noise_sigma: float, seed: int | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    rng = np.random.default_rng(seed)
    x = rng.uniform(X_MIN, X_MAX, size=n_obs)
    y = rng.uniform(Y_MIN, Y_MAX, size=n_obs)
    u = exact_solution(x, y)
    if noise_sigma > 0:
        u = u + rng.normal(0.0, noise_sigma, size=n_obs)
    xy = torch.tensor(np.stack([x, y], axis=1), dtype=torch.float32)
    u_t = torch.tensor(u, dtype=torch.float32).unsqueeze(1)
    return xy, u_t


# --------------------------------------------------------------------------
# Registration with the PDEProblem registry
# --------------------------------------------------------------------------

def _register() -> None:
    register_pde(
        PDEProblem(
            name="poisson2d",
            input_dim=2,
            residual_fn=poisson2d_residual,
            reference_solution=reference_solution,
            evaluation_grid=evaluation_grid,
            sample_collocation=sample_collocation,
            sample_initial=None,    # no time component
            sample_boundary=sample_boundary,
            ic_target=None,         # no initial condition
            sample_observations=sample_observations,
            coord_slices=(slice(None), slice(None)),
        )
    )


_register()
