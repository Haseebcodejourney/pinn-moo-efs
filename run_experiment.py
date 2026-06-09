"""
PINN multi-objective experiment CLI.

Usage:
  python run_experiment.py --mode baseline --pde burgers --seed 42
  python run_experiment.py --mode grid --pde burgers --n-obs 20 --noise 0.05
  python run_experiment.py --mode nsga2 --pde poisson --epochs 2500
  python run_experiment.py --mode nsga3 --pde burgers
  python run_experiment.py --mode efs --pde burgers
  python run_experiment.py --mode plot
  python run_experiment.py --mode matrix --quick
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from efs_optimizer import run_efs
from experiment_log import append_run_log, result_csv
from nsga2_optimizer import pareto_nondominated, run_nsga2, run_nsga3
from plot_results import aggregate_all_summaries, build_summary_table, main as plot_main
from run_grid import run_grid
from train import train_pinn

OBS_COUNTS = [5, 10, 20, 50, 100]
NOISE_LEVELS = [0.0, 0.01, 0.05, 0.1]
PDES = ["burgers", "poisson"]


def _save(df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    pareto = pareto_nondominated(df)
    pareto.to_csv(path.with_name(path.stem + "_pareto.csv"), index=False)
    print(f"Saved {len(df)} runs -> {path}")
    return path


def run_baseline(
    pde: str,
    epochs: int,
    seed: int,
    n_obs: int,
    obs_noise: float,
    balancer: str | None,
) -> pd.DataFrame:
    print(f"Baseline PINN pde={pde} n_obs={n_obs} noise={obs_noise} balancer={balancer or 'fixed'}")
    _, metrics = train_pinn(
        pde_name=pde,
        n_obs=n_obs,
        obs_noise=obs_noise,
        balancer=balancer,
        epochs=epochs,
        seed=seed,
        verbose=True,
    )
    row = {
        "pde": pde,
        "algorithm": f"baseline_{balancer or 'fixed'}",
        "n_obs": n_obs,
        "obs_noise": obs_noise,
        **metrics,
    }
    df = pd.DataFrame([row])
    path = result_csv(pde, f"baseline_{balancer or 'fixed'}", n_obs, obs_noise)
    _save(df, path)
    append_run_log(row)
    return df


def run_efs_mode(pde: str, n_obs: int, obs_noise: float, seed: int) -> None:
    candidates = [
        result_csv(pde, "nsga2", n_obs, obs_noise),
        result_csv(pde, "nsga3", n_obs, obs_noise),
        result_csv(pde, "grid", n_obs, obs_noise),
        ROOT / "results" / "grid_results.csv",
        ROOT / "results" / "moo_results.csv",
    ]
    pareto_path = None
    for path in candidates:
        p = path if path.suffix == ".csv" else path
        if p.exists():
            pareto_path = p.with_name(p.stem + "_pareto.csv")
            if not pareto_path.exists():
                pareto_path = p
            break
    if pareto_path is None:
        raise FileNotFoundError("Run grid or nsga2/nsga3 before efs.")

    out = run_efs(pareto_path, ROOT / "results", seed=seed)
    print(f"EFS improves over static fuzzy: {out['result']['efs_improves']}")
    print(out["comparison"])


def run_matrix(args: argparse.Namespace) -> None:
    pdes = PDES if args.pde == "all" else [args.pde]
    obs_counts = [20, 50] if args.quick else OBS_COUNTS
    noise_levels = [0.0, 0.05] if args.quick else NOISE_LEVELS
    epochs = 800 if args.quick else args.epochs
    pop, gen = (6, 3) if args.quick else (args.pop_size, args.generations)
    lam_values = [0.1, 1.0, 10.0] if not args.quick else [0.1, 1.0]
    n_values = [100, 200] if args.quick else [100, 200, 300, 400]

    summaries = []
    for pde in pdes:
        if pde == "poisson":
            configs = [(0, 0.0)]
        else:
            configs = [(n, nz) for n in obs_counts for nz in noise_levels]

        for n_obs, noise in configs:
            print(f"\n=== matrix: {pde} n_obs={n_obs} noise={noise} ===")
            run_baseline(pde, epochs, args.seed, n_obs, noise, balancer=None)

            run_grid(
                pde_name=pde,
                n_obs=n_obs,
                obs_noise=noise,
                lam_values=lam_values,
                n_values=n_values,
                epochs=epochs,
                seed=args.seed,
                out_name=result_csv(pde, "grid", n_obs, noise).name,
            )
            moo_df = run_nsga2(
                pde_name=pde,
                n_obs=n_obs,
                obs_noise=noise,
                pop_size=pop,
                n_gen=gen,
                epochs=epochs,
                seed=args.seed,
            )
            _save(moo_df, result_csv(pde, "nsga2", n_obs, noise))

            if not args.quick:
                moo3_df = run_nsga3(
                    pde_name=pde,
                    n_obs=n_obs,
                    obs_noise=noise,
                    pop_size=pop,
                    n_gen=gen,
                    epochs=epochs,
                    seed=args.seed,
                )
                _save(moo3_df, result_csv(pde, "nsga3", n_obs, noise))

            pareto_path = result_csv(pde, "nsga2", n_obs, noise)
            p_pareto = pareto_path.with_name(pareto_path.stem + "_pareto.csv")
            if p_pareto.exists():
                efs_out = run_efs(p_pareto, ROOT / "results", seed=args.seed)
                efs_params = efs_out["params"]
            else:
                efs_params = None

            summary = build_summary_table(moo_df, efs_params=efs_params)
            summaries.append(summary)

    if summaries:
        master = pd.concat(summaries, ignore_index=True)
        master.to_csv(ROOT / "results" / "summary_table.csv", index=False)
        print(f"Master summary -> {ROOT / 'results' / 'summary_table.csv'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="PINN multi-objective + EFS experiment")
    parser.add_argument(
        "--mode",
        choices=["baseline", "grid", "nsga2", "nsga3", "efs", "plot", "matrix", "all"],
        default="baseline",
    )
    parser.add_argument("--pde", choices=["burgers", "poisson", "all"], default="burgers")
    parser.add_argument("--epochs", type=int, default=2500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pop-size", type=int, default=10)
    parser.add_argument("--generations", type=int, default=6)
    parser.add_argument("--n-obs", type=int, default=0)
    parser.add_argument("--noise", type=float, default=0.0)
    parser.add_argument("--balancer", choices=["fixed", "gradnorm", "relobralo"], default="fixed")
    parser.add_argument("--quick", action="store_true", help="Reduced matrix for smoke testing")
    args = parser.parse_args()

    balancer = None if args.balancer == "fixed" else args.balancer
    pde = args.pde if args.pde != "all" else "burgers"

    if args.mode in ("baseline", "all"):
        run_baseline(pde, args.epochs, args.seed, args.n_obs, args.noise, balancer)

    if args.mode in ("grid", "all"):
        df = run_grid(
            pde_name=pde,
            n_obs=args.n_obs,
            obs_noise=args.noise,
            epochs=args.epochs,
            seed=args.seed,
            out_name=result_csv(pde, "grid", args.n_obs, args.noise).name,
        )
        append_run_log({"mode": "grid", "n_runs": len(df)})

    if args.mode in ("nsga2", "all"):
        df = run_nsga2(
            pde_name=pde,
            n_obs=args.n_obs,
            obs_noise=args.noise,
            pop_size=args.pop_size,
            n_gen=args.generations,
            epochs=args.epochs,
            seed=args.seed,
        )
        _save(df, result_csv(pde, "nsga2", args.n_obs, args.noise))

    if args.mode in ("nsga3", "all"):
        df = run_nsga3(
            pde_name=pde,
            n_obs=args.n_obs,
            obs_noise=args.noise,
            pop_size=args.pop_size,
            n_gen=args.generations,
            epochs=args.epochs,
            seed=args.seed,
        )
        _save(df, result_csv(pde, "nsga3", args.n_obs, args.noise))

    if args.mode in ("efs", "all"):
        run_efs_mode(pde, args.n_obs, args.noise, args.seed)

    if args.mode == "matrix":
        run_matrix(args)

    if args.mode in ("plot", "all"):
        plot_main(pde=None if args.pde == "all" else pde, epochs=args.epochs)
        aggregate_all_summaries()


if __name__ == "__main__":
    main()
