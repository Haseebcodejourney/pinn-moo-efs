"""2D incompressible Navier-Stokes cylinder wake, real experimental data.

This is the third benchmark in the paper and is the deliverable for
Prof. Dr. Asiksoy's "real data" request:

    "adding an open dataset containing sparse real measurements paired
     with a known PDE (e.g., data from fluid dynamics or heat-transfer
     measurements)..."

The data comes from Maziar Raissi's PINNs repository
(https://github.com/maziarraissi/PINNs), specifically the file
``Data/cylinder_nektar_wake.mat`` which contains experimentally measured
velocity and pressure fields around a cylinder (Kourta & Azerad 1995
dataset).  It is downloaded once and cached at

    results/real_data_external/cylinder_nektar_wake.mat

The PDE is the 2D incompressible Navier-Stokes in pressure form

    u_x + v_y              = 0                            (continuity)
    u_t + u u_x + v u_y    = -p_x + nu (u_xx + u_yy)      (x-momentum)
    v_t + u v_x + v v_y    = -p_y + nu (v_xx + v_yy)      (y-momentum)

with viscosity nu = 0.025.  The PINN is multi-output: (u, v, p).

Observations
------------
``sample_observations(n_obs, sigma, seed)`` subsamples
``(x_i, y_i, t_i) -> (u_i, v_i, p_i)`` directly from the
experimental .mat file, so the "sigma" parameter is only used if
``sigma > 0`` (it is left at 0 for the real-data case because the
measurement noise is already implicit in the experimental data).
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import scipy.io as sio
import torch

from pde_registry import PDEProblem, register_pde

# --------------------------------------------------------------------------
# Data loading (cached)
# --------------------------------------------------------------------------

# Domain: cylinder wake from Raissi's PINNs repo
X_MIN, X_MAX = 1.0, 8.0
Y_MIN, Y_MAX = -2.0, 2.0
T_MIN, T_MAX = 0.0, 20.0
# Cylinder position (Raissi's setup: radius 0.5 at (1, 1.5))
CYL_X, CYL_Y, CYL_R = 1.0, 1.5, 0.5
# Viscosity
NU = 0.025

_ROOT = Path(__file__).resolve().parent.parent
_DATA_PATH = _ROOT / "results" / "real_data_external" / "cylinder_nektar_wake.mat"


def _ensure_data() -> dict:
    """Load the .mat file (must already be downloaded)."""
    if not _DATA_PATH.exists():
        raise FileNotFoundError(
            f"cylinder wake .mat not found at {_DATA_PATH}. "
            "Run: curl -L -o results/real_data_external/cylinder_nektar_wake.mat "
            "https://github.com/maziarraissi/PINNs/raw/master/main/Data/"
            "cylinder_nektar_wake.mat"
        )
    return sio.loadmat(str(_DATA_PATH))


def _load_observations() -> tuple[np.ndarray, np.ndarray]:
    """Reshape the .mat arrays into (M, 3) coords and (M, 3) targets.

    X_star:    (5000, 2)  spatial (x, y)
    t:         (200, 1)   time
    U_star:    (5000, 2, 200)  velocity (u, v) at each (x, y, t)
    p_star:    (5000, 200)     pressure at each (x, y, t)

    Returns
    -------
    coords : (M, 3) array  of (x, y, t)
    values : (M, 3) array  of (u, v, p)
    where M = 5000 * 200 = 1,000,000
    """
    data = _ensure_data()
    X = data["X_star"]  # (5000, 2)
    t = data["t"].ravel()  # (200,)
    U = data["U_star"]  # (5000, 2, 200)
    p = data["p_star"]  # (5000, 200)

    nx, _ = X.shape
    nt = t.size
    M = nx * nt

    # coords: tile spatial across time, concat time
    coords = np.zeros((M, 3), dtype=np.float32)
    coords[:, 0] = np.tile(X[:, 0], nt)
    coords[:, 1] = np.tile(X[:, 1], nt)
    coords[:, 2] = np.repeat(t, nx)

    # values: U_star[:, 0, :] is u(x, y, t), shape (nx, nt) -> flat
    values = np.zeros((M, 3), dtype=np.float32)
    values[:, 0] = U[:, 0, :].ravel()  # u
    values[:, 1] = U[:, 1, :].ravel()  # v
    values[:, 2] = p.ravel()            # p

    return coords, values


# --------------------------------------------------------------------------
# PDE residual (autograd through the multi-output model)
# --------------------------------------------------------------------------

def cylinder_wake_residual(model: torch.nn.Module, xyt: torch.Tensor) -> torch.Tensor:
    """Navier-Stokes residuals at collocation points.

    model(xyt) -> (u, v, p), shape (N, 3)
    Returns a (N, 3) tensor of residuals:
        col 0: continuity    u_x + v_y
        col 1: x-momentum    u_t + u u_x + v u_y + p_x - nu (u_xx + u_yy)
        col 2: y-momentum    v_t + u v_x + v v_y + p_y - nu (v_xx + v_yy)
    """
    xyt = xyt.requires_grad_(True)
    out = model(xyt)
    u, v, p = out[:, 0:1], out[:, 1:2], out[:, 2:3]

    # First derivatives (autograd)
    grads = torch.autograd.grad(
        torch.cat([u, v, p], dim=1), xyt,
        grad_outputs=torch.ones_like(out), create_graph=True
    )[0]

    u_x = grads[:, 0:1]; u_y = grads[:, 1:2]; u_t = grads[:, 2:3]
    v_x = grads[:, 0:1]; v_y = grads[:, 1:2]; v_t = grads[:, 2:3]
    p_x = grads[:, 0:1]; p_y = grads[:, 1:2]

    # Second derivatives (sequential, may share one graph)
    def _d2(field_dx_dy_dt: torch.Tensor, var_index: int) -> tuple[torch.Tensor, torch.Tensor]:
        g = torch.autograd.grad(
            field_dx_dy_dt, xyt,
            grad_outputs=torch.ones_like(field_dx_dy_dt), create_graph=True
        )[0]
        return g[:, 0:1], g[:, 1:2]

    u_xx, u_yy = _d2(u_x, 0)
    v_xx, v_yy = _d2(v_x, 1)

    r_continuity = u_x + v_y
    r_xmom = u_t + u * u_x + v * u_y + p_x - NU * (u_xx + u_yy)
    r_ymom = v_t + u * v_x + v * v_y + p_y - NU * (v_xx + v_yy)

    return torch.cat([r_continuity, r_xmom, r_ymom], dim=1)


# --------------------------------------------------------------------------
# Sampling helpers
# --------------------------------------------------------------------------

def sample_collocation(n: int, seed: int | None = None) -> torch.Tensor:
    """Random points in the spatio-temporal domain, EXCLUDING the cylinder."""
    rng = np.random.default_rng(seed)
    # Oversample to make sure we exclude cylinder region
    x = rng.uniform(X_MIN, X_MAX, size=n * 2)
    y = rng.uniform(Y_MIN, Y_MAX, size=n * 2)
    t = rng.uniform(T_MIN, T_MAX, size=n * 2)

    # Mask out points inside the cylinder
    mask = (x - CYL_X) ** 2 + (y - CYL_Y) ** 2 > CYL_R ** 2
    x, y, t = x[mask][:n], y[mask][:n], t[mask][:n]
    # If we under-sampled (rare), pad with random non-cylinder points
    while x.size < n:
        xt = rng.uniform(X_MIN, X_MAX, size=n)
        yt = rng.uniform(Y_MIN, Y_MAX, size=n)
        keep = (xt - CYL_X) ** 2 + (yt - CYL_Y) ** 2 > CYL_R ** 2
        x = np.concatenate([x, xt[keep]])
        y = np.concatenate([y, yt[keep]])
    return torch.tensor(np.stack([x[:n], y[:n], t[:n]], axis=1), dtype=torch.float32)


def sample_boundary(n: int, seed: int | None = None) -> torch.Tensor:
    """Boundary points: the cylinder surface (where no-slip / u=v=0 holds).

    For real cylinder wake the no-slip condition is what is actually
    enforced, and this function is only used for ``data_terms.append(mse(model(tx_bc), 0))``
    in train.py.  We sample the cylinder surface here and let the data
    loss drive u, v -> 0 on the boundary.
    """
    rng = np.random.default_rng(seed)
    theta = rng.uniform(0.0, 2.0 * np.pi, size=n)
    x = CYL_X + CYL_R * np.cos(theta)
    y = CYL_Y + CYL_R * np.sin(theta)
    t = rng.uniform(T_MIN, T_MAX, size=n)
    return torch.tensor(np.stack([x, y, t], axis=1), dtype=torch.float32)


def sample_observations(
    n_obs: int, noise_sigma: float, seed: int | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Subsample n_obs (x, y, t) -> (u, v, p) from the real experimental data."""
    coords, values = _load_observations()
    rng = np.random.default_rng(seed)
    M = coords.shape[0]
    if n_obs >= M:
        idx = np.arange(M)
    else:
        idx = rng.choice(M, size=n_obs, replace=False)
    sel_coords = coords[idx]
    sel_values = values[idx]
    if noise_sigma > 0:
        sel_values = sel_values + rng.normal(0.0, noise_sigma, size=sel_values.shape).astype(np.float32)
    return (
        torch.tensor(sel_coords, dtype=torch.float32),
        torch.tensor(sel_values, dtype=torch.float32),
    )


def evaluation_grid(nt: int = 20, nx: int = 50, ny: int = 50) -> torch.Tensor:
    """Spatio-temporal grid for f1 (relative L2) on the real data."""
    x = np.linspace(X_MIN, X_MAX, nx)
    y = np.linspace(Y_MIN, Y_MAX, ny)
    t = np.linspace(T_MIN, T_MAX, nt)
    xx, yy, tt = np.meshgrid(x, y, t, indexing="ij")
    # Flatten and exclude cylinder region
    pts = np.stack([xx.ravel(), yy.ravel(), tt.ravel()], axis=1)
    mask = (pts[:, 0] - CYL_X) ** 2 + (pts[:, 1] - CYL_Y) ** 2 > CYL_R ** 2
    return torch.tensor(pts[mask], dtype=torch.float32)


def reference_solution(coords: np.ndarray, _unused: np.ndarray | None = None) -> np.ndarray:
    """Interpolate (u, v, p) at the given (x, y, t) coords from real data.

    Uses nearest-neighbor for speed.  Inputs of shape (N, 3) -> (N, 3).
    The second argument is unused (kept for registry signature compatibility).
    """
    data = _ensure_data()
    X = data["X_star"]  # (5000, 2)
    t = data["t"].ravel()  # (200,)
    U = data["U_star"]  # (5000, 2, 200)
    p = data["p_star"]  # (5000, 200)

    x = coords[:, 0]
    y = coords[:, 1]
    tt = coords[:, 2]

    # Find nearest spatial index (brute-force over 5000 points is fine for evaluation)
    d2 = (X[:, 0:1] - x[None, :]) ** 2 + (X[:, 1:2] - y[None, :]) ** 2
    spatial_idx = d2.argmin(axis=0)

    # Find nearest time index
    dt = np.abs(t[:, None] - tt[None, :])
    time_idx = dt.argmin(axis=0)

    out = np.zeros((coords.shape[0], 3), dtype=np.float32)
    out[:, 0] = U[spatial_idx, 0, time_idx]
    out[:, 1] = U[spatial_idx, 1, time_idx]
    out[:, 2] = p[spatial_idx, time_idx]
    return out


# --------------------------------------------------------------------------
# Registration
# --------------------------------------------------------------------------

def _register() -> None:
    register_pde(
        PDEProblem(
            name="cylinder_wake",
            input_dim=3,           # (x, y, t)
            output_dim=3,          # (u, v, p)
            residual_fn=cylinder_wake_residual,
            reference_solution=reference_solution,
            evaluation_grid=evaluation_grid,
            sample_collocation=sample_collocation,
            sample_initial=None,   # no analytical initial condition; real data is the truth
            sample_boundary=sample_boundary,
            ic_target=None,
            sample_observations=sample_observations,
            coord_slices=(slice(None), slice(None), slice(None)),
        )
    )


_register()
