"""1D viscous Burgers equation: u_t + u*u_x = nu*u_xx."""

from __future__ import annotations

import os

import numpy as np
import torch
NU = 0.01 / np.pi
X_MIN, X_MAX = -1.0, 1.0
T_MIN, T_MAX = 0.0, 1.0

# When PINN_USE_REAL_DATA=1, sample_observations delegates to
# real_data_burgers.sample_real_observations, which uses a high-fidelity
# Radau reference (1024 x 512) as ground truth with a noise sigma derived
# from subsampling interpolation error (non-arbitrary). See
# src/real_data_burgers.py for the derivation.
USE_REAL_DATA = os.environ.get("PINN_USE_REAL_DATA", "0") == "1"


def initial_condition(x: np.ndarray) -> np.ndarray:
    return -np.sin(np.pi * x)


def sample_collocation(n: int, seed: int | None = None) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    t = rng.uniform(T_MIN, T_MAX, size=n)
    x = rng.uniform(X_MIN, X_MAX, size=n)
    return torch.tensor(np.stack([t, x], axis=1), dtype=torch.float32)


def sample_initial(n: int, seed: int | None = None) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    x = rng.uniform(X_MIN, X_MAX, size=n)
    t = np.zeros_like(x)
    return torch.tensor(np.stack([t, x], axis=1), dtype=torch.float32)


def sample_boundary(n: int, seed: int | None = None) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    t = rng.uniform(T_MIN, T_MAX, size=n)
    x = np.empty(n)
    x[: n // 2] = X_MIN
    x[n // 2 :] = X_MAX
    rng.shuffle(x)
    return torch.tensor(np.stack([t, x], axis=1), dtype=torch.float32)


def burgers_residual(model: torch.nn.Module, tx: torch.Tensor) -> torch.Tensor:
    """PDE residual at collocation points."""
    tx = tx.requires_grad_(True)
    u = model(tx)
    grads = torch.autograd.grad(u, tx, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_t = grads[:, 0:1]
    u_x = grads[:, 1:2]
    u_xx = torch.autograd.grad(u_x, tx, grad_outputs=torch.ones_like(u_x), create_graph=True)[0][:, 1:2]
    return u_t + u * u_x - NU * u_xx


def _reference_grid(nx: int = 201, nt: int = 101) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stable FD reference via implicit Radau time integration (method of lines)."""
    from scipy.integrate import solve_ivp

    x = np.linspace(X_MIN, X_MAX, nx)
    dx = x[1] - x[0]
    t_eval = np.linspace(T_MIN, T_MAX, nt)

    def rhs(_t: float, u: np.ndarray) -> np.ndarray:
        du = np.zeros_like(u)
        ux = (u[2:] - u[:-2]) / (2 * dx)
        uxx = (u[2:] - 2 * u[1:-1] + u[:-2]) / dx**2
        du[1:-1] = -u[1:-1] * ux + NU * uxx
        return du

    u0 = initial_condition(x).astype(np.float64)
    u0[0] = 0.0
    u0[-1] = 0.0

    sol = solve_ivp(
        rhs,
        (T_MIN, T_MAX),
        u0,
        t_eval=t_eval,
        method="Radau",
        rtol=1e-5,
        atol=1e-8,
        max_step=0.01,
    )
    u = sol.y.T
    u[:, 0] = 0.0
    u[:, -1] = 0.0
    return t_eval, x, u


_REF_CACHE: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None


def clear_reference_cache() -> None:
    global _REF_CACHE
    _REF_CACHE = None


def reference_solution(t: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Bilinear interpolation on cached reference grid."""
    global _REF_CACHE
    if _REF_CACHE is None:
        _REF_CACHE = _reference_grid()

    t_grid, x_grid, u_grid = _REF_CACHE
    t = np.asarray(t)
    x = np.asarray(x)
    t_idx = np.searchsorted(t_grid, t, side="right") - 1
    x_idx = np.searchsorted(x_grid, x, side="right") - 1
    t_idx = np.clip(t_idx, 0, u_grid.shape[0] - 2)
    x_idx = np.clip(x_idx, 0, u_grid.shape[1] - 2)

    t0 = t_grid[t_idx]
    t1 = t_grid[t_idx + 1]
    x0 = x_grid[x_idx]
    x1 = x_grid[x_idx + 1]
    wt = np.where(t1 > t0, (t - t0) / (t1 - t0), 0.0)
    wx = np.where(x1 > x0, (x - x0) / (x1 - x0), 0.0)

    u00 = u_grid[t_idx, x_idx]
    u01 = u_grid[t_idx, x_idx + 1]
    u10 = u_grid[t_idx + 1, x_idx]
    u11 = u_grid[t_idx + 1, x_idx + 1]
    u0 = u00 * (1 - wx) + u01 * wx
    u1 = u10 * (1 - wx) + u11 * wx
    return u0 * (1 - wt) + u1 * wt


def evaluation_grid(nt: int = 100, nx: int = 256) -> torch.Tensor:
    t = np.linspace(T_MIN, T_MAX, nt)
    x = np.linspace(X_MIN, X_MAX, nx)
    tt, xx = np.meshgrid(t, x, indexing="ij")
    return torch.tensor(np.stack([tt.ravel(), xx.ravel()], axis=1), dtype=torch.float32)


def sample_observations(n_obs: int, noise_sigma: float, seed: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample (t, x, u) observation triples.

    If the module-level USE_REAL_DATA flag is set, observations are drawn
    from a high-fidelity Radau reference (see src/real_data_burgers.py).
    Otherwise the standard synthetic path is used.

    In the real-data path, ``noise_sigma`` is interpreted as an override:
    if > 0, that value is added on top of the derived subsampling noise;
    if <= 0, only the derived subsampling noise is applied.
    """
    if USE_REAL_DATA:
        from real_data_burgers import sample_real_observations as _sample_real
        sigma_override = noise_sigma if noise_sigma > 0 else None
        return _sample_real(n_obs=n_obs, seed=seed, sigma=sigma_override)

    rng = np.random.default_rng(seed)
    t = rng.uniform(T_MIN, T_MAX, size=n_obs)
    x = rng.uniform(X_MIN, X_MAX, size=n_obs)
    u = reference_solution(t, x)
    if noise_sigma > 0:
        u = u + rng.normal(0.0, noise_sigma, size=n_obs)
    tx = torch.tensor(np.stack([t, x], axis=1), dtype=torch.float32)
    u_t = torch.tensor(u, dtype=torch.float32).unsqueeze(1)
    return tx, u_t


def _register() -> None:
    from pde_registry import PDEProblem, register_pde

    register_pde(
        PDEProblem(
            name="burgers",
            input_dim=2,
            residual_fn=burgers_residual,
            reference_solution=reference_solution,
            evaluation_grid=evaluation_grid,
            sample_collocation=sample_collocation,
            sample_initial=sample_initial,
            sample_boundary=sample_boundary,
            ic_target=lambda tx: torch.tensor(initial_condition(tx[:, 1].cpu().numpy()), dtype=torch.float32).unsqueeze(1),
            sample_observations=sample_observations,
            coord_slices=(slice(None), slice(None)),
        )
    )


_register()
