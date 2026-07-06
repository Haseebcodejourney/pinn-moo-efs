"""
Focused runner for the real-data Burgers experiment.

What it does
------------
For each n_obs in {20, 50, 100} and each loss-balancing method in
{Grid, GradNorm, ReLoBRaLo}, runs the Burgers PINN with the
``PINN_USE_REAL_DATA=1`` flag set so observations come from the
high-fidelity Radau reference (1024 x 512) with sigma derived from
fine-grid interpolation error (see src/real_data_burgers.py).

Outputs
-------
    results/multiseed_burgers_real_data_n{020,050,100}/
        <exp>_all_seeds.csv
        <exp>_summary_stats.csv
        <exp>_metadata.json

Then it runs ``real_data_comparison.py`` to produce the aggregated
table at results/real_data_table.{csv,md}.

Usage
-----
    PINN_USE_REAL_DATA=1 python run_real_data_experiment.py --seeds 0,1,2
    PINN_USE_REAL_DATA=1 python run_real_data_experiment.py --seeds 5 --epochs 800

Notes
-----
The PINN_USE_REAL_DATA env var must be set BEFORE python starts (it is
read once when pde_burgers.py is imported). If you forget it, the
script will exit with an error.

CPU budget on this machine: 3 seeds x 3 n_obs x 3 baselines at
epochs=500 ~ 15 min. With --seeds 5 and --epochs 800 ~ 60 min.
"""

from __future__ import annotations

import argparse
import json
import os
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

# CRITICAL: must be set before importing pde_burgers, which reads it at import time.
if os.environ.get("PINN_USE_REAL_DATA", "0") != "1":
    print("ERROR: PINN_USE_REAL_DATA=1 must be set BEFORE running this script.")
    print("       Example:")
    print("         PINN_USE_REAL_DATA=1 python run_real_data_experiment.py --seeds 0,1,2")
    sys.exit(1)

import numpy as np
import pandas as pd

# Now safe to import pde_burgers -- it has read USE_REAL_DATA = True
import pde_burgers  # noqa: F401  -- registers PDEProblem and reads the flag
from run_baselines_for_table import (
    _burgers_grid_seed,
    _burgers_gradnorm_seed,
    _burgers_relobralo_seed,
)
from run_grid import run_grid
from train import train_pinn
from objectives import evaluate_objectives
from pde_registry import get_pde
from real_data_burgers import derive_measurement_noise, generate_high_fidelity_reference

RESULTS = ROOT / "results"

# Confirm the flag took effect at module-import time
assert pde_burgers.USE_REAL_DATA is True, (
    "pde_burgers.USE_REAL_DATA is False -- check PINN_USE_REAL_DATA env var."
)


def _burgers_realdata_grid_seed(seed: int, n_obs: int, epochs: int) -> pd.DataFrame:
    """Run a grid sweep using the real-data observation source."""
    df = run_grid(
        pde_name="burgers",
        epochs=epochs,
        seed=seed,
        lam_values=[1.0, 10.0],
        n_values=[100, 200, 300],
        out_name=None,
        balancer=None,
        n_obs=n_obs,
        obs_noise=0.0,  # sigma is derived inside sample_observations
    )
    df["seed"] = seed
    df["real_data"] = True
    df["real_data_sigma"] = derive_measurement_noise(n_obs, seed=seed)
    return df


def _burgers_realdata_gradnorm_seed(seed: int, n_obs: int, epochs: int) -> pd.DataFrame:
    df = run_grid(
        pde_name="burgers",
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
    df["real_data_sigma"] = derive_measurement_noise(n_obs, seed=seed)
    return df


def _burgers_realdata_relobralo_seed(seed: int, n_obs: int, epochs: int) -> pd.DataFrame:
    df = run_grid(
        pde_name="burgers",
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
    df["real_data_sigma"] = derive_measurement_noise(n_obs, seed=seed)
    return df


# (label, function, dir-suffix, short-name)
METHODS = [
    ("Grid",       _burgers_realdata_grid_seed,       "burgers_grid_clean"),
    ("GradNorm",   _burgers_realdata_gradnorm_seed,   "burgers_grid_gradnorm"),
    ("ReLoBRaLo",  _burgers_realdata_relobralo_seed,  "burgers_grid_relobralo"),
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
        "csv": str(csv_path),
    }
    (out_dir / f"{exp_name}_metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"  [OK] {csv_path}  ({len(out)} rows, {out['seed'].nunique()} seeds)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0,1,2", help="Comma-separated seed list")
    ap.add_argument("--epochs", type=int, default=500, help="Per-PINN training epochs")
    ap.add_argument("--n-obs-list", default="20,50,100",
                    help="Comma-separated n_obs values to sweep")
    ap.add_argument("--only-methods", default=None,
                    help="Run only methods whose label contains this substring")
    ap.add_argument("--only-nobs", default=None,
                    help="Run only n_obs values containing this substring")
    args = ap.parse_args()

    seed_list = [int(s) for s in args.seeds.split(",") if s.strip()]
    n_obs_list = [int(n) for n in args.n_obs_list.split(",") if n.strip()]
    ctx = dict(seeds=seed_list, epochs=args.epochs, n_obs_list=n_obs_list)

    # One-time: generate the high-fidelity reference NPZ
    print("=" * 78)
    print("REAL-DATA BURGERS EXPERIMENT")
    print(f"PINN_USE_REAL_DATA={os.environ.get('PINN_USE_REAL_DATA')}")
    print(f"seeds={seed_list}  epochs={args.epochs}  n_obs_list={n_obs_list}")
    print("=" * 78)
    print("Generating high-fidelity reference (one-time, cached afterwards) ...")
    ref_path = generate_high_fidelity_reference()
    print(f"  reference cached -> {ref_path}")
    print()
    print("Derived sigma for each n_obs:")
    for n in n_obs_list:
        s = derive_measurement_noise(n)
        print(f"  n_obs={n}: sigma={s:.6f}")
    print()

    # Run each (method, n_obs) combination across the configured seeds
    for label, fn, short_name in METHODS:
        if args.only_methods and args.only_methods not in label:
            continue
        for n_obs in n_obs_list:
            if args.only_nobs and args.only_nobs not in str(n_obs):
                continue
            tag = f"burgers_real_data_n{n_obs:03d}"
            method_tag = f"{short_name}_n{n_obs:03d}"
            print(f"\n>> {label}  (n_obs={n_obs}, tag={tag})")
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
            # Save into per-(n_obs) directory with one CSV per method
            if frames:
                _save_run(method_tag, frames, {**ctx, "n_obs": n_obs, "method": label})
            # Also stash in the per-n_obs aggregated directory for downstream consumers
            _stash_to_aggregated(tag, method_tag, frames)

    print("\n" + "=" * 78)
    print("Running real_data_comparison.py on the freshly-written CSVs ...")
    print("=" * 78)
    import subprocess
    subprocess.run([sys.executable, str(ROOT / "real_data_comparison.py")], check=False)


def _stash_to_aggregated(tag: str, method_tag: str, frames: list[pd.DataFrame]) -> None:
    """Append method_tag's CSV into the per-n_obs aggregated directory
    (results/multiseed_{tag}/). This is the directory real_data_comparison.py
    reads from.
    """
    if not frames:
        return
    out = pd.concat(frames, ignore_index=True)
    out_dir = _seed_dir(tag)
    csv_path = out_dir / f"{tag}_all_seeds.csv"
    if csv_path.exists():
        # Append to existing aggregated CSV
        existing = pd.read_csv(csv_path)
        out = pd.concat([existing, out], ignore_index=True)
    out.to_csv(csv_path, index=False)


if __name__ == "__main__":
    main()