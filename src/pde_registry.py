"""PDE problem registry for multi-benchmark PINN training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch


@dataclass
class PDEProblem:
    name: str
    input_dim: int
    residual_fn: Callable[[torch.nn.Module, torch.Tensor], torch.Tensor]
    reference_solution: Callable[[np.ndarray, np.ndarray | None], np.ndarray]
    evaluation_grid: Callable[[], torch.Tensor]
    sample_collocation: Callable[[int, int | None], torch.Tensor]
    sample_initial: Callable[[int, int | None], torch.Tensor] | None
    sample_boundary: Callable[[int, int | None], torch.Tensor]
    ic_target: Callable[[torch.Tensor], torch.Tensor] | None
    sample_observations: Callable[[int, float, int | None], tuple[torch.Tensor, torch.Tensor]]
    coord_slices: tuple[slice, ...]  # slices into input tensor for reference lookup
    output_dim: int = 1  # 1 for scalar PDEs, >1 for multi-field PDEs (e.g. NS)


_REGISTRY: dict[str, PDEProblem] = {}


def register_pde(problem: PDEProblem) -> None:
    _REGISTRY[problem.name] = problem


def get_pde(name: str) -> PDEProblem:
    if name not in _REGISTRY:
        raise KeyError(f"Unknown PDE '{name}'. Available: {list(_REGISTRY)}")
    return _REGISTRY[name]


def list_pdes() -> list[str]:
    return list(_REGISTRY.keys())
