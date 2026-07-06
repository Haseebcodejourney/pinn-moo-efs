"""Multi-seed runner for the real-data cylinder wake experiment.

This is the deliverable for the user's "Please add real data" request
following Prof. Dr. Asiksoy's note:

    "adding an open dataset containing sparse real measurements paired
     with a known PDE (e.g., data from fluid dynamics or heat-transfer
     measurements) would also eliminate the 'sigma = 0.05 was chosen
     arbitrarily' criticism."

The data is the experimentally measured 2D incompressible Navier-Stokes
cylinder wake from Maziar Raissi's PINNs repository
(https://github.com/maziarraissi/PINNs), downloaded once and cached at
``results/real_data_external/cylinder_nektar_wake.mat``.

Pipeline
--------
For each n_obs in {50, 200, 500} and each method in
{Grid, GradNorm, ReLoBRaLo}, run the cylinder-wake PINN with the real
experimental data as the observation source.  Output per-(method, n_obs)
multi-seed CSVs at:

    results/multiseed_cylinder_wake/<method>_n<NNN>_all_seeds.csv

Then run ``cylinder_wake_comparison.py`` to produce
``results/cylinder_wake_table.{csv,md}``.

Usage
-----
    python run_cylinder_wake_experiment.py --seeds 0,1
    python run_cylinder_wake_experiment.py --seeds 0,1,2 --epochs 500

CPU budget on this machine: 2 seeds x 3 n_obs x 3 baselines x ~6 PINNs
at epochs=400 ~ 8 min.  3 seeds at epochs=500 ~ 15-20 min.
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

import pandas as pd

import pde_cylinder_wake  # noqa: F401 -- registers the PDEProblem
from run_grid import run_grid

RESULTS = ROOT / "results"


def _cylinder_wake_grid_seed(seed: int, n_obs: int, epochs: int) -> pd.DataFrame:
    """Grid sweep over (lam, N_coll) using the real-data observation source."""
    df = run_grid(
        pde_name="cylinder_wake",
        epochs=epochs,
        seed=seed,
        lam_values=[1.0, 10.0],
        n_values=[100, 200, 300],
        out_name=None,
        balancer=None,
        n_obs=n_obs,
        obs_noise=0.0,  # real data, no synthetic noise added
    )
    df["seed"] = seed
    df["real_data"] = True
    df["data_source"] = "cylinder_nektar_wake.mat"
    return df


def _cylinder_wake_gradnorm_seed(seed: int, n_obs: int, epochs: int) -> pd.DataFrame:
    df = run_grid(
        pde_name="cylinder_wake",
        epochs=epochs,
        seed=seed,
        lam_values=[1.0],
        n_values=[100, 200, 300],
        out_name=None,
        balancer="gradnorm",
        n_obs=n_obs,
        obs_noise=0.0,
    )
    df["seed"] = seed
    df["real_data"] = True
    df["data_source"] = "cylinder_nektar_wake.mat"
    return df


def _cylinder_wake_relobralo_seed(seed: int, n_obs: int, epochs: int) -> pd.DataFrame:
    df = run_grid(
        pde_name="cylinder_wake",
        epochs=epochs,
        seed=seed,
        lam_values=[1.0],
        n_values=[100, 200, 300],
        out_name=None,
        balancer="relobralo",
        n_obs=n_obs,
        obs_noise=0.0,
    )
    df["seed"] = seed
    df["real_data"] = True
    df["data_source"] = "cylinder_nektar_wake.mat"
    return df


# (label, function, short-name)
METHODS = [
    ("Grid",       _cylinder_wake_grid_seed,       "cylinder_wake_grid_clean"),
    ("GradNorm",   _cylinder_wake_gradnorm_seed,   "cylinder_wake_grid_gradnorm"),
    ("ReLoBRaLo",  _cylinder_wake_relobralo_seed,  "cylinder_wake_grid_relobralo"),
]


def _seed_dir(name: str) -> Path:
    p = RESULTS / f"multiseed_{name}"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _save_run(exp_name: str, frames: list[pd.DataFrame], ctx: dict) -> None:
    if not frames:
        return
    out = pd.concat(frames, ignore_index=True)
    out_dir = _seed_dir(exp_name)
    csv_path = out_dir / f"{exp_name}_all_seeds.csv"
    out.to_csv(csv_path, index=False)

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
        "real_data": True,
        "data_source": "cylinder_nektar_wake.mat (Raissi 2019 PINNs repo)",
        "csv": str(csv_path),
    }
    (out_dir / f"{exp_name}_metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"  [OK] {csv_path}  ({len(out)} rows, {out['seed'].nunique()} seeds)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0,1", help="Comma-separated seed list")
    ap.add_argument("--epochs", type=int, default=400, help="Per-PINN training epochs")
    ap.add_argument("--n-obs-list", default="50,200,500",
                    help="Comma-separated n_obs values to sweep")
    ap.add_argument("--only-methods", default=None,
                    help="Run only methods whose label contains this substring")
    ap.add_argument("--only-nobs", default=None,
                    help="Run only n_obs values containing this substring")
    args = ap.parse_args()

    seed_list = [int(s) for s in args.seeds.split(",") if s.strip()]
    n_obs_list = [int(n) for n in args.n_obs_list.split(",") if n.strip()]
    ctx = dict(seeds=seed_list, epochs=args.epochs, n_obs_list=n_obs_list)

    print("=" * 78)
    print("REAL-DATA CYLINDER WAKE EXPERIMENT (Raissi 2019 PINNs repository)")
    print(f"seeds={seed_list}  epochs={args.epochs}  n_obs_list={n_obs_list}")
    print("=" * 78)
    data_path = RESULTS / "real_data_external" / "cylinder_nektar_wake.mat"
    if not data_path.exists():
        print(f"ERROR: data file not found at {data_path}")
        sys.exit(1)
    print(f"Data: {data_path}  ({data_path.stat().st_size / 1e6:.2f} MB)")
    print()

    # Run each (method, n_obs) combination across the configured seeds
    for label, fn, short_name in METHODS:
        if args.only_methods and args.only_methods not in label:
            continue
        for n_obs in n_obs_list:
            if args.only_nobs and args.only_nobs not in str(n_obs):
                continue
            method_tag = f"{short_name}_n{n_obs:03d}"
            print(f"\n>> {label}  (n_obs={n_obs}, tag={method_tag})")
            frames: list[pd.DataFrame] = []
            for seed in seed_list:
                t0 = time.time()
                try:
                    df = fn(seed=seed, n_obs=n_obs, epochs=args.epochs)
                    dt = time.time() - t0
                    print(f"   seed={seed}  rows={len(df)}  ({dt:.1f}s)")
                    frames.append(df)
                except Exception as e:  # noqa: BLE001
                    print(f"   seed={seed}  FAILED: {type(e).__name__}: {e}")
            if frames:
                _save_run(method_tag, frames, {**ctx, "n_obs": n_obs, "method": label})

    print("\n" + "=" * 78)
    print("Running cylinder_wake_comparison.py on the freshly-written CSVs ...")
    print("=" * 78)
    import subprocess
    subprocess.run([sys.executable, str(ROOT / "cylinder_wake_comparison.py")], check=False)


if __name__ == "__main__":
    main()