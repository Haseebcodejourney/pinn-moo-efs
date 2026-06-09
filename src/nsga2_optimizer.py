"""NSGA-II / NSGA-III over PINN hyperparameters."""

from __future__ import annotations

import numpy as np
import pandas as pd
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.core.problem import ElementwiseProblem
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from pymoo.util.ref_dirs import get_reference_directions

from pareto_metrics import OBJECTIVE_COLS, sanitize_objectives_df
from train import train_pinn


class PINNHyperparamProblem(ElementwiseProblem):
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


def run_nsga2(
    pde_name: str = "burgers",
    n_obs: int = 0,
    obs_noise: float = 0.0,
    pop_size: int = 12,
    n_gen: int = 8,
    epochs: int = 2500,
    seed: int = 42,
    device: str | None = None,
) -> pd.DataFrame:
    problem = PINNHyperparamProblem(
        pde_name=pde_name,
        n_obs=n_obs,
        obs_noise=obs_noise,
        epochs=epochs,
        seed=seed,
        device=device,
    )
    algorithm = NSGA2(pop_size=pop_size)
    termination = get_termination("n_gen", n_gen)

    res = minimize(problem, algorithm, termination, seed=seed, verbose=True, save_history=False)
    return _results_dataframe(res, pde_name, "nsga2", n_obs, obs_noise)


def run_nsga3(
    pde_name: str = "burgers",
    n_obs: int = 0,
    obs_noise: float = 0.0,
    pop_size: int = 12,
    n_gen: int = 8,
    n_partitions: int = 8,
    epochs: int = 2500,
    seed: int = 42,
    device: str | None = None,
) -> pd.DataFrame:
    ref_dirs = get_reference_directions("das-dennis", 3, n_partitions=n_partitions)
    if pop_size < len(ref_dirs):
        pop_size = len(ref_dirs)

    problem = PINNHyperparamProblem(
        pde_name=pde_name,
        n_obs=n_obs,
        obs_noise=obs_noise,
        epochs=epochs,
        seed=seed,
        device=device,
    )
    algorithm = NSGA3(pop_size=pop_size, ref_dirs=ref_dirs)
    termination = get_termination("n_gen", n_gen)

    res = minimize(problem, algorithm, termination, seed=seed, verbose=True, save_history=False)
    return _results_dataframe(res, pde_name, "nsga3", n_obs, obs_noise)


def pareto_nondominated(df: pd.DataFrame) -> pd.DataFrame:
    """Return nondominated rows; supports legacy f3 column name."""
    df = sanitize_objectives_df(df)
    if df.empty:
        return df
    cols = ["f1_l2_error", "f2_pde_residual", "f3_data_budget"]
    objs = df[cols].to_numpy()
    keep = []
    for i, a in enumerate(objs):
        dominated = False
        for j, b in enumerate(objs):
            if i == j:
                continue
            if np.all(b <= a) and np.any(b < a):
                dominated = True
                break
        if not dominated:
            keep.append(i)
    return df.iloc[keep].reset_index(drop=True)
