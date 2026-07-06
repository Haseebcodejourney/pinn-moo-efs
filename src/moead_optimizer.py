"""MOEA/D optimizer over PINN hyperparameters.

MOEA/D (Multi-Objective Evolutionary Algorithm based on Decomposition)
decomposes a multi-objective problem into N scalar subproblems, each
optimized by collaborating with neighbours. The professor's review
explicitly asked for a non-NSGA-II baseline; MOEA/D is the standard
counterpart.

Reference: Zhang & Li, "MOEA/D: A Multiobjective Evolutionary Algorithm
Based on Decomposition", IEEE Trans. Evol. Comput. 2007.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pymoo.algorithms.moo.moead import MOEAD
from pymoo.core.problem import ElementwiseProblem
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from pymoo.util.ref_dirs import get_reference_directions

from pareto_metrics import OBJECTIVE_COLS
from train import train_pinn


class PINNHyperparamProblemMOEAD(ElementwiseProblem):
    def __init__(
        self,
        pde_name: str = "burgers",
        n_obs: int = 0,
        obs_noise: float = 0.0,
        epochs: int = 2500,
        seed: int = 42,
        device: str | None = None,
    ):
        super().__init__(
            n_var=3,
            n_obj=3,
            n_ieq_constr=0,
            xl=np.array([1e-3, 1e-3, 50.0]),
            xu=np.array([10.0, 10.0, 500.0]),
        )
        self.pde_name = pde_name
        self.n_obs = n_obs
        self.obs_noise = obs_noise
        self.epochs = epochs
        self.seed = seed
        self.device = device
        self.eval_count = 0

    def _evaluate(self, x, out, *args, **kwargs):
        lam_data, lam_pde, n_coll = float(x[0]), float(x[1]), int(round(x[2]))
        run_seed = self.seed + self.eval_count
        self.eval_count += 1

        _, metrics = train_pinn(
            pde_name=self.pde_name,
            lam_data=lam_data,
            lam_pde=lam_pde,
            n_collocation=n_coll,
            n_obs=self.n_obs,
            obs_noise=self.obs_noise,
            epochs=self.epochs,
            seed=run_seed,
            device=self.device,
            verbose=False,
        )
        out["F"] = [metrics[c] for c in OBJECTIVE_COLS]


def _results_dataframe(res, pde_name: str, algorithm: str, n_obs: int, obs_noise: float) -> pd.DataFrame:
    rows = []
    for i in range(len(res.X)):
        rows.append(
            {
                "pde": pde_name,
                "algorithm": algorithm,
                "n_obs": n_obs,
                "obs_noise": obs_noise,
                "lam_data": float(res.X[i, 0]),
                "lam_pde": float(res.X[i, 1]),
                "n_collocation": int(round(res.X[i, 2])),
                "f1_l2_error": float(res.F[i, 0]),
                "f2_pde_residual": float(res.F[i, 1]),
                "f3_data_budget": float(res.F[i, 2]),
            }
        )
    return pd.DataFrame(rows)


def run_moead(
    pde_name: str = "burgers",
    n_obs: int = 0,
    obs_noise: float = 0.0,
    n_partitions: int = 12,
    pop_size: int | None = None,
    n_gen: int = 30,
    epochs: int = 2500,
    seed: int = 42,
    device: str | None = None,
) -> pd.DataFrame:
    """Run MOEA/D with Tchebycheff decomposition.

    pop_size is set to the number of reference directions if not provided.
    """
    ref_dirs = get_reference_directions("das-dennis", 3, n_partitions=n_partitions)
    if pop_size is None:
        pop_size = len(ref_dirs)
    elif pop_size < len(ref_dirs):
        pop_size = len(ref_dirs)
        n_partitions = max(1, _partitions_for_pop(pop_size, 3))

    problem = PINNHyperparamProblemMOEAD(
        pde_name=pde_name,
        n_obs=n_obs,
        obs_noise=obs_noise,
        epochs=epochs,
        seed=seed,
        device=device,
    )
    algorithm = MOEAD(
        ref_dirs=ref_dirs,
        n_neighbors=15,
        prob_neighbor_mating=0.7,
    )
    termination = get_termination("n_gen", n_gen)

    res = minimize(problem, algorithm, termination, seed=seed, verbose=True, save_history=False)
    return _results_dataframe(res, pde_name, "moead", n_obs, obs_noise)


def _partitions_for_pop(pop: int, k: int) -> int:
    """Approximate n_partitions such that C(n_partitions + k - 1, k - 1) ~ pop."""
    p = 1
    while True:
        combos = 1
        for i in range(1, k):
            combos = combos * (p + k - 1 - i) // i
        if combos >= pop or p > 30:
            return p
        p += 1
