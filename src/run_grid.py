"""Grid search over loss weights and collocation budget."""

from __future__ import annotations

import itertools
from pathlib import Path

import pandas as pd

from nsga2_optimizer import pareto_nondominated
from train import train_pinn


def run_grid(
    pde_name: str = "burgers",
    n_obs: int = 0,
    obs_noise: float = 0.0,
    lam_values: list[float] | None = None,
    n_values: list[int] | None = None,
    epochs: int = 2500,
    seed: int = 42,
    device: str | None = None,
    out_name: str | None = "grid_results.csv",
    balancer: str | None = None,
) -> pd.DataFrame:
    """Grid search over loss weights and collocation budget.

    Parameters
    ----------
    balancer : None | "gradnorm" | "relobralo"
        When set, loss-balancing is delegated to the adaptive strategy and
        the lambda grid collapses to a single (1, 1) entry per n_coll --
        GradNorm/ReLoBRaLo override the lambdas at every step.
    """
    if lam_values is None:
        lam_values = [0.1, 1.0, 10.0]
    if n_values is None:
        n_values = [100, 200, 300, 400]

    if balancer is not None:
        # adaptive balancers override lam_d/lam_p, so collapse that axis
        lam_values = [1.0]
        algorithm_name = f"grid_{balancer}"
    else:
        algorithm_name = "grid"

    rows = []
    for i, (lam_d, lam_p, n_coll) in enumerate(itertools.product(lam_values, lam_values, n_values)):
        try:
            _, metrics = train_pinn(
                pde_name=pde_name,
                lam_data=lam_d,
                lam_pde=lam_p,
                n_collocation=n_coll,
                n_obs=n_obs,
                obs_noise=obs_noise,
                epochs=epochs,
                seed=seed + i,
                device=device,
                verbose=False,
                balancer=balancer,
            )
        except Exception as exc:
            print(f"[{i + 1}] SKIPPED lam=({lam_d},{lam_p}) N={n_coll}: {exc}")
            continue

        required = ("f1_l2_error", "f2_pde_residual", "f3_data_budget")
        if any(k not in metrics or metrics[k] is None for k in required):
            print(f"[{i + 1}] SKIPPED incomplete metrics for lam=({lam_d},{lam_p}) N={n_coll}")
            continue

        rows.append(
            {
                "pde": pde_name,
                "algorithm": algorithm_name,
                "n_obs": n_obs,
                "obs_noise": obs_noise,
                "lam_data": lam_d,
                "lam_pde": lam_p,
                "n_collocation": n_coll,
                "balancer": balancer or "fixed",
                **metrics,
            }
        )
        print(
            f"[{i + 1}] pde={pde_name} balancer={balancer or 'fixed'} "
            f"N={n_coll} obs={n_obs} "
            f"L2={metrics['f1_l2_error']:.4f} res={metrics['f2_pde_residual']:.4f}"
        )

    if not rows:
        raise RuntimeError("Grid search produced no valid runs.")

    df = pd.DataFrame(rows)

    # Only save if out_name is provided
    if out_name is not None:
        out = Path(__file__).resolve().parents[1] / "results" / out_name
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)

        pareto = pareto_nondominated(df)
        pareto_out = out.parent / out_name.replace(".csv", "_pareto.csv")
        pareto.to_csv(pareto_out, index=False)
        print(f"Saved {len(df)} runs -> {out}")
        print(f"Pareto set ({len(pareto)} points) -> {pareto_out}")

    return df