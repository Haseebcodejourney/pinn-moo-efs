"""Assemble master outputs from the multi-seed experiments that have completed.

The comprehensive finish_all_multiseed.py pipeline takes ~10 hours on this
CPU-only machine at the chosen 5-seed budget (NSGA-II 15 pop / 8 gen at
epochs=500, NSGA-III 12/6, MOEA/D 12/6). To get usable results within a
realistic session wall-clock we let the non-MOO + NSGA-II experiments run
end-to-end, then aggregate:

  1. multiseed_all_runs.csv      -- every individual (seed, config) row
  2. multiseed_master_summary.csv-- mean/std per (algorithm, hyperparams)
  3. multiseed_pareto_metrics.csv-- per-seed hypervolume (with explicit nadir)
  4. multiseed_significance_tests.csv -- Wilcoxon vs baseline_fixed

Experiments with 5 seeds completed:
  - poisson_grid_clean        (Poisson 1D, 9 configs x 5 seeds = 45 rows)
  - burgers_baseline_fixed    (Burgers baseline, 1 PINN x 5 seeds = 5 rows)
  - burgers_grid_clean        (Burgers grid, 6 configs x 5 seeds = 30 rows)
  - burgers_grid_gradnorm     (Burgers grid, 3 configs x 5 seeds = 15 rows)
  - burgers_grid_relobralo    (Burgers grid, 3 configs x 5 seeds = 15 rows)
  - burgers_nsga2_clean       (NSGA-II 15/8 budget, ~14 Pareto-optimal per seed = 70 rows)

The NSGA-III / MOEA/D / 2D Poisson / noise-sweep at 5 seeds are intentionally
NOT re-run here -- they are already covered in:
  - results/baseline_comparison_table.md  (NSGA-III, MOEA/D at 3 seeds)
  - results/cylinder_wake_table.md        (2D Navier-Stokes real data, 3 seeds)
  - results/real_data_table.md            (Burgers noise robustness via n_obs)

This script produces the same four top-level CSVs that finish_all_multiseed.py
would produce, but only from the experiments that completed in this session.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

from multi_seed_runner import statistical_significance_test
from nsga2_optimizer import pareto_nondominated
from pareto_metrics import hypervolume, spacing

RESULTS = ROOT / "results"

# (display_name, actual_subdir_under_results, csv_filename_inside_subdir)
EXPERIMENTS = [
    ("poisson_grid_clean",          "multiseed_poisson_grid",                "poisson_grid_clean"),
    ("burgers_baseline_fixed",      "multiseed_burgers_baseline",           "burgers_baseline_fixed"),
    ("burgers_grid_clean",          "multiseed_burgers_grid_clean",          "burgers_grid_clean"),
    ("burgers_grid_gradnorm",       "multiseed_burgers_grid_gradnorm",       "burgers_grid_gradnorm"),
    ("burgers_grid_relobralo",      "multiseed_burgers_grid_relobralo",      "burgers_grid_relobralo"),
    ("burgers_nsga2_clean",         "multiseed_burgers_nsga2",               "burgers_nsga2_clean"),
    ("burgers_nsga3_clean",         "multiseed_burgers_nsga3_clean",         "burgers_nsga3_clean"),
    ("burgers_moead_clean",         "multiseed_burgers_moead_clean",         "burgers_moead_clean"),
    ("burgers_grid_noisy_s01",      "multiseed_burgers_grid_noisy_s01",      "burgers_grid_noisy_s01"),
    ("burgers_grid_noisy_s05",      "multiseed_burgers_grid_noisy",          "burgers_grid_noisy"),
    ("burgers_grid_noisy_s10",      "multiseed_burgers_grid_noisy_s10",      "burgers_grid_noisy_s10"),
    ("poisson2d_grid_clean",        "multiseed_poisson2d_grid",              "poisson2d_grid_clean"),
]


def load_experiment(display_name: str, subdir: str, filename: str) -> pd.DataFrame | None:
    csv = RESULTS / subdir / f"{filename}_all_seeds.csv"
    if csv.exists():
        df = pd.read_csv(csv)
        df["experiment"] = display_name
        return df
    return None


def compute_pareto_metrics(df: pd.DataFrame) -> dict[str, float]:
    """Per-(experiment, seed) HV with explicit nadir + spacing."""
    pareto = pareto_nondominated(df)
    if len(pareto) < 1:
        return {"hv": float("nan"), "spacing": float("nan"),
                "nadir_f1": float("nan"), "nadir_f2": float("nan"),
                "nadir_f3": float("nan")}
    pts = pareto[["f1_l2_error", "f2_pde_residual", "f3_data_budget"]].to_numpy()
    lo = pts.min(axis=0)
    hi = pts.max(axis=0)
    span = np.maximum(hi - lo, 1e-8)
    nadir = hi + 0.10 * span
    return {
        "hv": float(hypervolume(pts, ref_point=nadir)),
        "spacing": float(spacing(pts)),
        "nadir_f1": float(nadir[0]),
        "nadir_f2": float(nadir[1]),
        "nadir_f3": float(nadir[2]),
    }


def main() -> None:
    frames: list[pd.DataFrame] = []
    for display_name, subdir, filename in EXPERIMENTS:
        df = load_experiment(display_name, subdir, filename)
        if df is None:
            print(f"[WARN] {display_name}: not found at {subdir}/{filename}_all_seeds.csv, skipping")
            continue
        frames.append(df)
        print(f"[OK] {display_name}: {len(df)} rows, {df['seed'].nunique()} seeds ({subdir})")

    if not frames:
        print("No data to aggregate.")
        return

    all_runs = pd.concat(frames, ignore_index=True)
    all_runs_path = RESULTS / "multiseed_all_runs.csv"
    all_runs.to_csv(all_runs_path, index=False)
    print(f"\n[OK] {all_runs_path}  ({len(all_runs)} rows total)")

    # Master summary: mean / std / min / max per (experiment, algorithm, ...)
    metric_cols = ["f1_l2_error", "f2_pde_residual", "f3_data_budget"]
    grouping_cols = [c for c in all_runs.columns
                     if c not in metric_cols and c != "seed"]
    summary_rows = []
    for key, g in all_runs.groupby(grouping_cols, dropna=False):
        row = dict(zip(grouping_cols, key if isinstance(key, tuple) else [key]))
        row["n_seeds"] = int(g["seed"].nunique())
        for col in metric_cols:
            row[f"{col}_mean"] = float(g[col].mean())
            row[f"{col}_std"]  = float(g[col].std()) if g[col].size > 1 else 0.0
            row[f"{col}_min"]  = float(g[col].min())
            row[f"{col}_max"]  = float(g[col].max())
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    summary_path = RESULTS / "multiseed_master_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"[OK] {summary_path}  ({len(summary)} groups)")

    # Pareto metrics per (experiment, seed)
    pareto_rows = []
    for (exp, seed), g in all_runs.groupby(["experiment", "seed"]):
        m = compute_pareto_metrics(g)
        m["experiment"] = exp
        m["seed"] = int(seed)
        pareto_rows.append(m)
    pareto_df = pd.DataFrame(pareto_rows)
    pareto_path = RESULTS / "multiseed_pareto_metrics.csv"
    pareto_df.to_csv(pareto_path, index=False)
    print(f"[OK] {pareto_path}  ({len(pareto_df)} (experiment, seed) pairs)")

    # Significance tests: baseline vs each candidate (best per seed)
    baseline = all_runs[all_runs["experiment"] == "burgers_baseline_fixed"]
    if not baseline.empty:
        base_best = baseline.groupby("seed")["f1_l2_error"].mean()

        sig_rows = []
        for exp_name in ["burgers_grid_clean", "burgers_grid_gradnorm",
                         "burgers_grid_relobralo", "burgers_nsga2_clean",
                         "burgers_nsga3_clean", "burgers_moead_clean"]:
            cand = all_runs[all_runs["experiment"] == exp_name]
            if cand.empty:
                continue
            cand_best = cand.groupby("seed")["f1_l2_error"].min()
            common = base_best.index.intersection(cand_best.index)
            if len(common) < 3:
                continue
            t = statistical_significance_test(
                base_best.loc[common].values,
                cand_best.loc[common].values,
                metric_name="best_f1_l2",
            )
            sig_rows.append({
                "comparison": f"baseline_fixed_vs_{exp_name}",
                "metric": "best_f1_l2_per_seed",
                "baseline_mean": float(base_best.loc[common].mean()),
                "baseline_std":  float(base_best.loc[common].std()),
                "candidate_mean": float(cand_best.loc[common].mean()),
                "candidate_std":  float(cand_best.loc[common].std()),
                "p_value": t["p_value"],
                "significant": t["significant"],
                "n_seeds_paired": len(common),
                "interpretation": t["interpretation"],
            })

        if sig_rows:
            sig_df = pd.DataFrame(sig_rows)
            sig_path = RESULTS / "multiseed_significance_tests.csv"
            sig_df.to_csv(sig_path, index=False)
            print(f"[OK] {sig_path}  ({len(sig_rows)} comparisons)")

            print("\n=== Wilcoxon paired-by-seed (n=5) ===")
            for r in sig_rows:
                print(f"  {r['comparison']}: p={r['p_value']:.4f}  "
                      f"baseline={r['baseline_mean']:.3f}+/-{r['baseline_std']:.3f}  "
                      f"candidate={r['candidate_mean']:.3f}+/-{r['candidate_std']:.3f}  "
                      f"{'SIG' if r['significant'] else 'n.s.'}")

    print("\n=== Per-experiment hypervolume (5 seeds) ===")
    for exp, sub in pareto_df.groupby("experiment"):
        print(f"  {exp}: HV = {sub['hv'].mean():.4f} +/- {sub['hv'].std():.4f}  "
              f"spacing = {sub['spacing'].mean():.4f} +/- {sub['spacing'].std():.4f}  "
              f"(nadir f1={sub['nadir_f1'].mean():.3f}, f2={sub['nadir_f2'].mean():.3f}, "
              f"f3={sub['nadir_f3'].mean():.1f})")


if __name__ == "__main__":
    main()