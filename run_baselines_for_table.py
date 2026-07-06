"""
Focused runner that produces the actual numbers for the baseline
comparison table demanded by Prof. Dr. Asiksoy.

It runs every baseline (Fixed-weight PINN, Grid, GradNorm, ReLoBRaLo,
NSGA-II, NSGA-III, MOEA/D) on Burgers 1D with the **same** set of
random seeds and the same training budget, then writes:

    results/multiseed_burgers_baseline/burgers_baseline_fixed_all_seeds.csv
    results/multiseed_burgers_grid_clean/burgers_grid_clean_all_seeds.csv
    results/multiseed_burgers_grid_gradnorm/burgers_grid_gradnorm_all_seeds.csv
    results/multiseed_burgers_grid_relobralo/burgers_grid_relobralo_all_seeds.csv
    results/multiseed_burgers_nsga2/burgers_nsga2_clean_all_seeds.csv
    results/multiseed_burgers_nsga3/burgers_nsga3_clean_all_seeds.csv
    results/multiseed_burgers_moead/burgers_moead_clean_all_seeds.csv

then runs baseline_comparison.py to produce the final table.

Default budget is small enough to finish on CPU within a few minutes;
override --epochs / --seeds / --pop / --gen for paper-grade runs.

Usage:
  python run_baselines_for_table.py             # default 3 seeds, 800 epochs
  python run_baselines_for_table.py --seeds 5   # 5 seeds
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Force UTF-8 stdout/stderr (Windows cp1252 console)
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
import torch

from moead_optimizer import run_moead
from nsga2_optimizer import run_nsga2, run_nsga3
from run_grid import run_grid
from train import train_pinn
from objectives import evaluate_objectives
from pde_registry import get_pde

RESULTS = ROOT / "results"


def _seed_dir(name: str) -> Path:
    p = RESULTS / f"multiseed_{name}"
    p.mkdir(parents=True, exist_ok=True)
    return p


# ---------------------------------------------------------------------------
# Per-baseline, per-seed runners. Each returns a DataFrame with the schema:
#   pde, algorithm, n_obs, obs_noise, lam_data, lam_pde, n_collocation,
#   f1_l2_error, f2_pde_residual, f3_data_budget, seed, ...
# ---------------------------------------------------------------------------

def _burgers_fixed_seed(seed: int, epochs: int) -> pd.DataFrame:
    """Single-seed fixed-weight PINN (lambda=(10, 1)) for a sanity reference."""
    pde = get_pde("burgers")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = train_pinn(
        pde_name="burgers",
        lam_data=10.0, lam_pde=1.0,
        n_collocation=200, n_obs=0, obs_noise=0.0,
        epochs=epochs, seed=seed, device=device,
    )
    tx_coll = pde.sample_collocation(200).to(device)
    m = evaluate_objectives(
        model=model, pde_name="burgers", n_collocation=200,
        tx_coll=tx_coll, device=device, n_obs=0,
    )
    return pd.DataFrame([{
        "pde": "burgers", "algorithm": "baseline_fixed",
        "n_obs": 0, "obs_noise": 0.0,
        "lam_data": 10.0, "lam_pde": 1.0, "n_collocation": 200,
        "balancer": "fixed", "hidden_dim": 50, "n_layers": 4,
        **m, "seed": seed,
    }])


def _burgers_grid_seed(seed: int, epochs: int) -> pd.DataFrame:
    """Grid sweep over (lam, n_coll). Includes balancer column for parity."""
    df = run_grid(
        pde_name="burgers",
        epochs=epochs,
        seed=seed,
        lam_values=[1.0, 10.0],
        n_values=[100, 200, 300],
        out_name=None,
        balancer=None,
    )
    df["seed"] = seed
    return df


def _burgers_gradnorm_seed(seed: int, epochs: int) -> pd.DataFrame:
    df = run_grid(
        pde_name="burgers",
        epochs=epochs,
        seed=seed,
        lam_values=[1.0],
        n_values=[100, 200, 300],
        out_name=None,
        balancer="gradnorm",
    )
    df["seed"] = seed
    return df


def _burgers_relobralo_seed(seed: int, epochs: int) -> pd.DataFrame:
    df = run_grid(
        pde_name="burgers",
        epochs=epochs,
        seed=seed,
        lam_values=[1.0],
        n_values=[100, 200, 300],
        out_name=None,
        balancer="relobralo",
    )
    df["seed"] = seed
    return df


def _burgers_nsga2_seed(seed: int, epochs: int, pop: int, gen: int) -> pd.DataFrame:
    df = run_nsga2(pde_name="burgers", pop_size=pop, n_gen=gen, epochs=epochs, seed=seed)
    df["seed"] = seed
    return df


def _burgers_nsga3_seed(seed: int, epochs: int, pop: int, gen: int, n_partitions: int = 4) -> pd.DataFrame:
    df = run_nsga3(
        pde_name="burgers", pop_size=pop, n_gen=gen,
        epochs=epochs, seed=seed, n_partitions=n_partitions,
    )
    df["seed"] = seed
    return df


def _burgers_moead_seed(seed: int, epochs: int, pop: int, gen: int, n_partitions: int = 4) -> pd.DataFrame:
    df = run_moead(
        pde_name="burgers", pop_size=pop, n_gen=gen,
        epochs=epochs, seed=seed, n_partitions=n_partitions,
    )
    df["seed"] = seed
    return df


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

EXPERIMENTS = [
    ("burgers_baseline_fixed", _burgers_fixed_seed,   {"epochs": "epochs"}, "Fixed-weight PINN"),
    ("burgers_grid_clean",     _burgers_grid_seed,    {"epochs": "epochs"}, "Grid (clean)"),
    ("burgers_grid_gradnorm",  _burgers_gradnorm_seed,{"epochs": "epochs"}, "GradNorm"),
    ("burgers_grid_relobralo", _burgers_relobralo_seed,{"epochs": "epochs"},"ReLoBRaLo"),
    ("burgers_nsga2_clean",    _burgers_nsga2_seed,   {"epochs": "epochs", "pop": "pop", "gen": "gen"}, "NSGA-II"),
    ("burgers_nsga3_clean",    _burgers_nsga3_seed,   {"epochs": "epochs", "pop": "pop", "gen": "gen"}, "NSGA-III"),
    ("burgers_moead_clean",    _burgers_moead_seed,   {"epochs": "epochs", "pop": "pop", "gen": "gen"}, "MOEA/D"),
]


def _resolve_kwargs(spec: dict, ctx: dict) -> dict:
    out = {}
    for k, ref in spec.items():
        out[k] = ctx[ref]
    return out


def _save_run(exp_name: str, frames: list[pd.DataFrame], ctx: dict) -> None:
    if not frames:
        return
    out = pd.concat(frames, ignore_index=True)
    out_dir = _seed_dir(exp_name)
    csv_path = out_dir / f"{exp_name}_all_seeds.csv"
    out.to_csv(csv_path, index=False)
    # Summary stats
    summary = (
        out.groupby(["pde", "algorithm", "n_obs", "obs_noise"], dropna=False)
        .agg(
            n_seeds=("seed", "nunique"),
            f1_l2_mean=("f1_l2_error", "mean"),
            f1_l2_std=("f1_l2_error", "std"),
            f2_res_mean=("f2_pde_residual", "mean"),
            f2_res_std=("f2_pde_residual", "std"),
            f3_budget_mean=("f3_data_budget", "mean"),
            f3_budget_std=("f3_data_budget", "std"),
        )
        .reset_index()
    )
    summary.to_csv(out_dir / f"{exp_name}_summary_stats.csv", index=False)
    meta = {
        "experiment": exp_name,
        "context": {k: v for k, v in ctx.items() if not isinstance(v, list)},
        "n_seeds": int(out["seed"].nunique()),
        "n_rows": int(len(out)),
        "csv": str(csv_path),
    }
    (out_dir / f"{exp_name}_metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"  [OK] {csv_path}  ({len(out)} rows, {out['seed'].nunique()} seeds)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0,1,2", help="Comma-separated seed list")
    ap.add_argument("--epochs", type=int, default=800, help="Per-PINN training epochs")
    ap.add_argument("--pop", type=int, default=8, help="NSGA/MOEA population")
    ap.add_argument("--gen", type=int, default=3, help="NSGA/MOEA generations")
    ap.add_argument("--n-partitions", type=int, default=4,
                    help="Das-Dennis partitions for NSGA-III / MOEA/D (lower => faster)")
    ap.add_argument("--only", default=None, help="Run only experiments whose name contains this substring")
    args = ap.parse_args()

    seed_list = [int(s) for s in args.seeds.split(",") if s.strip()]
    ctx = dict(
        seeds=seed_list,
        epochs=args.epochs,
        pop=args.pop,
        gen=args.gen,
        n_partitions=args.n_partitions,
    )

    print("=" * 78)
    print("BASELINE COMPARISON RUNNER (focused on Burgers 1D)")
    print(
        f"seeds={seed_list}  epochs={args.epochs}  pop={args.pop}  gen={args.gen}  "
        f"n_partitions={args.n_partitions}"
    )
    print("=" * 78)

    for exp_name, fn, kw_spec, label in EXPERIMENTS:
        if args.only and args.only not in exp_name:
            continue
        print(f"\n>> {label}  ({exp_name})")
        frames: list[pd.DataFrame] = []
        for seed in seed_list:
            t0 = time.time()
            try:
                df = fn(seed=seed, **_resolve_kwargs(kw_spec, ctx))
                dt = time.time() - t0
                print(f"   seed={seed}  rows={len(df)}  ({dt:.1f}s)")
                frames.append(df)
            except Exception as e:  # noqa: BLE001
                print(f"   seed={seed}  FAILED: {type(e).__name__}: {e}")
        _save_run(exp_name, frames, ctx)

    print("\n" + "=" * 78)
    print("Running baseline_comparison.py on the freshly-written CSVs ...")
    print("=" * 78)
    import subprocess
    subprocess.run([sys.executable, str(ROOT / "baseline_comparison.py")], check=False)


if __name__ == "__main__":
    main()