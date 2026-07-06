"""Three objectives: accuracy, physics consistency, data efficiency."""

from __future__ import annotations

import numpy as np
import torch

from pde_registry import PDEProblem, get_pde


@torch.no_grad()
def relative_l2_error(model: torch.nn.Module, pde: PDEProblem, device: torch.device) -> float:
    """f1: relative L2 error between prediction and reference, averaged over outputs."""
    coords = pde.evaluation_grid().to(device)
    pred = model(coords).cpu().numpy()

    if pde.input_dim == 1:
        x = coords[:, 0].cpu().numpy()
        truth = pde.reference_solution(x, None)
    elif pde.input_dim == 2:
        t = coords[:, 0].cpu().numpy()
        x = coords[:, 1].cpu().numpy()
        truth = pde.reference_solution(t, x)
    else:  # 3D, e.g. cylinder wake (x, y, t)
        truth = pde.reference_solution(coords.cpu().numpy(), None)

    pred = np.asarray(pred).reshape(-1)
    truth = np.asarray(truth).reshape(-1)
    num = np.linalg.norm(pred - truth)
    den = np.linalg.norm(truth) + 1e-12
    return float(num / den)


def mean_pde_residual(model: torch.nn.Module, pde: PDEProblem, tx_coll: torch.Tensor) -> float:
    """f2: mean absolute PDE residual at collocation points."""
    model.eval()
    residual = pde.residual_fn(model, tx_coll)
    return float(residual.abs().mean().detach().cpu().item())


def data_efficiency_cost(n_collocation: int, n_obs: int = 0) -> float:
    """f3: total data budget (collocation + sparse observations)."""
    return float(n_collocation + n_obs)


def evaluate_objectives(
    model: torch.nn.Module,
    pde_name: str,
    n_collocation: int,
    tx_coll: torch.Tensor,
    device: torch.device,
    n_obs: int = 0,
) -> dict[str, float]:
    pde = get_pde(pde_name)
    f1 = relative_l2_error(model, pde, device)
    f2 = mean_pde_residual(model, pde, tx_coll)
    f3 = data_efficiency_cost(n_collocation, n_obs)
    return {"f1_l2_error": f1, "f2_pde_residual": f2, "f3_data_budget": f3}