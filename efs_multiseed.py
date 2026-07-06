"""
EFS vs. Static-Fuzzy multi-seed comparison.

Professor Dr. Asiksoy's review identified the central weakness:
the "improvement from 1.00009 to 1.00002" came from a single run, so
a reviewer will dismiss it as noise. This script produces the
statistical evidence she explicitly asked for.

For each seed:
  1. Train a Burgers grid search -> pareto front
  2. Run a static-fuzzy selection on that pareto
  3. Run an EFS evolution on the same pareto
  4. Record balanced_loss for static vs evolved

Then across seeds:
  - mean +/- std for both
  - paired Wilcoxon signed-rank test (the test the professor named)
  - per-seed improvement consistency (X/N seeds improved)

Usage:
  python efs_multiseed.py                # default 5 seeds
  python efs_multiseed.py --seeds 10     # 10 seeds
  python efs_multiseed.py --seeds 5 --quick   # smaller budgets
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Force UTF-8 stdout/stderr (Windows cp1252 cannot encode non-ASCII glyphs)
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

from efs_optimizer import _balanced_loss, evolve_efs, run_efs
from fuzzy_rules import FuzzyRuleParams, rank_with_fuzzy
from multi_seed_runner import (
    run_multi_seed_experiment,
    statistical_significance_test,
)
from nsga2_optimizer import pareto_nondominated
from pareto_metrics import hypervolume, spacing
from run_grid import run_grid

RESULTS = ROOT / "results"


def run_efs_comparison_single_seed(
    seed: int,
    epochs: int = 1500,
    lam_values: list[float] | None = None,
    n_values: list[int] | None = None,
    efs_pop: int = 15,
    efs_gen: int = 15,
) -> pd.DataFrame:
    """
    For one seed: train Burgers grid, evolve EFS, return comparison rows.

    Returns a 1-row DataFrame with static vs. evolved balanced loss plus
    f1/f2/f3 metrics of the chosen candidate for each method.
    """
    if lam_values is None:
        lam_values = [1.0, 10.0]
    if n_values is None:
        n_values = [100, 200, 300]

    # 1) Grid search -> Pareto front
    grid_df = run_grid(
        pde_name="burgers",
        epochs=epochs,
        seed=seed,
        lam_values=lam_values,
        n_values=n_values,
        out_name=None,  # don't write per-seed files
    )
    pareto_df = pareto_nondominated(grid_df).reset_index(drop=True)

    if len(pareto_df) < 2:
        raise RuntimeError(
            f"Seed {seed}: pareto front has {len(pareto_df)} points, need >=2"
        )

    pareto_rows = pareto_df.to_dict(orient="records")

    # ------------------------------------------------------------------
    # Pareto-level metrics: hypervolume (with explicit nadir reference
    # point) and spacing. These are the metrics Prof. Asiksoy explicitly
    # named in her review, with HV requiring an explicit nadir to be
    # meaningful.
    # ------------------------------------------------------------------
    pts = pareto_df[["f1_l2_error", "f2_pde_residual", "f3_data_budget"]].to_numpy()
    lo = pts.min(axis=0)
    hi = pts.max(axis=0)
    span = np.maximum(hi - lo, 1e-8)
    nadir = hi + 0.1 * span                       # 10% beyond worst observed
    pareto_hv = hypervolume(pts, ref_point=nadir)  # MC estimate, 5000 samples
    pareto_spacing = spacing(pts)

    # 2) Static-fuzzy selection (default parameters, no evolution)
    static_top = rank_with_fuzzy(pareto_rows, params=None)[0]

    # 3) EFS evolution
    best_params, _hist = evolve_efs(
        pareto_df, pop_size=efs_pop, n_gen=efs_gen, seed=seed
    )
    evolved_top = rank_with_fuzzy(pareto_rows, params=best_params)[0]

    static_loss = _balanced_loss(static_top)
    evolved_loss = _balanced_loss(evolved_top)

    return pd.DataFrame(
        [
            {
                "pde": "burgers",
                "seed": seed,
                "n_pareto_points": len(pareto_df),
                "static_fuzzy_balanced_loss": static_loss,
                "evolved_fuzzy_balanced_loss": evolved_loss,
                "efs_improvement_abs": static_loss - evolved_loss,
                "efs_improvement_pct": (
                    100.0 * (static_loss - evolved_loss) / static_loss
                    if static_loss > 0
                    else 0.0
                ),
                "efs_better": static_loss > evolved_loss,
                "static_f1_l2_error": float(static_top["f1_l2_error"]),
                "static_f2_pde_residual": float(static_top["f2_pde_residual"]),
                "static_f3_data_budget": float(
                    static_top.get("f3_data_budget", static_top.get("f3_n_collocation", 0.0))
                ),
                "evolved_f1_l2_error": float(evolved_top["f1_l2_error"]),
                "evolved_f2_pde_residual": float(evolved_top["f2_pde_residual"]),
                "evolved_f3_data_budget": float(
                    evolved_top.get("f3_data_budget", evolved_top.get("f3_n_collocation", 0.0))
                ),
                "pareto_hypervolume": pareto_hv,
                "pareto_hypervolume_nadir_f1": float(nadir[0]),
                "pareto_hypervolume_nadir_f2": float(nadir[1]),
                "pareto_hypervolume_nadir_f3": float(nadir[2]),
                "pareto_spacing": pareto_spacing,
            }
        ]
    )


def main(
    seed_list: list[int],
    epochs: int = 1500,
    efs_pop: int = 15,
    efs_gen: int = 15,
) -> None:
    out_dir = RESULTS / "multiseed_burgers_efs"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("EFS vs. STATIC-FUZZY: Multi-Seed Statistical Comparison")
    print("=" * 78)
    print(f"Seeds: {seed_list}")
    print(f"Per-seed: Burgers grid ({epochs} ep) + EFS evolution (pop={efs_pop}, gen={efs_gen})")
    print(f"Output:   {out_dir}")
    print("=" * 78)

    # Run multi-seed
    def _fn(seed: int) -> pd.DataFrame:
        return run_efs_comparison_single_seed(
            seed=seed,
            epochs=epochs,
            efs_pop=efs_pop,
            efs_gen=efs_gen,
        )

    result = run_multi_seed_experiment(
        experiment_fn=_fn,
        seeds=seed_list,
        output_dir=out_dir,
        experiment_name="efs_vs_static_fuzzy",
        verbose=True,
    )
    all_runs: pd.DataFrame = result["all_runs"]

    # ------------------------------------------------------------------
    # Per-method aggregate stats
    # ------------------------------------------------------------------
    static_losses = all_runs["static_fuzzy_balanced_loss"].to_numpy()
    evolved_losses = all_runs["evolved_fuzzy_balanced_loss"].to_numpy()
    abs_impr = all_runs["efs_improvement_abs"].to_numpy()
    rel_impr = all_runs["efs_improvement_pct"].to_numpy()
    n_better = int(all_runs["efs_better"].sum())

    # Pareto-level metrics across seeds (the professor's named metrics)
    hv = all_runs["pareto_hypervolume"].to_numpy()
    sp = all_runs["pareto_spacing"].to_numpy()
    nadir_f1 = all_runs["pareto_hypervolume_nadir_f1"].to_numpy()
    nadir_f2 = all_runs["pareto_hypervolume_nadir_f2"].to_numpy()
    nadir_f3 = all_runs["pareto_hypervolume_nadir_f3"].to_numpy()

    # ------------------------------------------------------------------
    # Statistical test (Wilcoxon signed-rank, paired per seed)
    # ------------------------------------------------------------------
    test = statistical_significance_test(
        method1_values=static_losses,
        method2_values=evolved_losses,
        metric_name="balanced_loss",
        alpha=0.05,
    )

    print()
    print("=" * 78)
    print("RESULTS")
    print("=" * 78)
    print(f"Per-seed table:\n{all_runs.to_string(index=False)}\n")
    print(
        f"Across {len(seed_list)} seeds:\n"
        f"  static_fuzzy balanced_loss:  {static_losses.mean():.6f} +/- {static_losses.std():.6f}\n"
        f"  evolved_fuzzy balanced_loss: {evolved_losses.mean():.6f} +/- {evolved_losses.std():.6f}\n"
        f"  mean absolute improvement:   {abs_impr.mean():.6f}  ({rel_impr.mean():.2f}%)\n"
        f"  EFS wins in:                 {n_better}/{len(seed_list)} seeds\n"
    )
    print("PARETO-LEVEL METRICS (across seeds):")
    print(
        f"  hypervolume:  {hv.mean():.6f} +/- {hv.std():.6f}  "
        f"(nadir f1={nadir_f1.mean():.3f}, f2={nadir_f2.mean():.3f}, f3={nadir_f3.mean():.1f})"
    )
    print(f"  spacing:      {sp.mean():.6f} +/- {sp.std():.6f}\n")
    print("STATISTICAL TEST (Wilcoxon signed-rank, paired):")
    print(f"  {test['interpretation']}")
    print(f"  W = {test['statistic']:.4f}")
    print(f"  p = {test['p_value']:.6f}")
    print(f"  mean_diff (evolved - static) = {test['mean_diff']:.6f}  (negative = EFS better)")
    print()

    # ------------------------------------------------------------------
    # Save statistical test summary as a separate CSV
    # ------------------------------------------------------------------
    summary_rows = [
        {
            "metric": "balanced_loss",
            "n_seeds": len(seed_list),
            "static_mean": static_losses.mean(),
            "static_std": static_losses.std(),
            "evolved_mean": evolved_losses.mean(),
            "evolved_std": evolved_losses.std(),
            "mean_abs_improvement": abs_impr.mean(),
            "mean_rel_improvement_pct": rel_impr.mean(),
            "n_seeds_efs_better": n_better,
            "wilcoxon_statistic": test["statistic"],
            "wilcoxon_p_value": test["p_value"],
            "significant_at_0.05": test["significant"],
            "interpretation": test["interpretation"],
            "pareto_hypervolume_mean": hv.mean(),
            "pareto_hypervolume_std": hv.std(),
            "pareto_spacing_mean": sp.mean(),
            "pareto_spacing_std": sp.std(),
            "nadir_f1_mean": nadir_f1.mean(),
            "nadir_f2_mean": nadir_f2.mean(),
            "nadir_f3_mean": nadir_f3.mean(),
        }
    ]
    pd.DataFrame(summary_rows).to_csv(
        out_dir / "efs_vs_static_fuzzy_significance.csv", index=False
    )
    print(f"[OK] Saved significance summary -> {out_dir / 'efs_vs_static_fuzzy_significance.csv'}")

    # Also extend the master significance file if it exists
    master_path = RESULTS / "multiseed_significance_tests.csv"
    if master_path.exists():
        existing = pd.read_csv(master_path)
        new_row = pd.DataFrame(
            [
                {
                    "comparison": "efs_vs_static_fuzzy",
                    "metric": "balanced_loss",
                    "baseline_mean": float(static_losses.mean()),
                    "baseline_std": float(static_losses.std()),
                    "grid_mean": float(evolved_losses.mean()),
                    "grid_std": float(evolved_losses.std()),
                    "p_value": test["p_value"],
                    "significant": test["significant"],
                    "mean_improvement": float(rel_impr.mean()),
                }
            ]
        )
        pd.concat([existing, new_row], ignore_index=True).to_csv(
            master_path, index=False
        )
        print(f"[OK] Appended to {master_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="EFS vs static-fuzzy multi-seed statistical comparison"
    )
    parser.add_argument("--seeds", type=str, default="5",
                        help="Number of seeds OR comma-separated list")
    parser.add_argument("--epochs", type=int, default=1500,
                        help="Training epochs per PINN candidate")
    parser.add_argument("--efs-pop", type=int, default=15,
                        help="EFS GA population size")
    parser.add_argument("--efs-gen", type=int, default=15,
                        help="EFS GA generations")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: smaller budgets for sanity check")
    args = parser.parse_args()

    if "," in args.seeds:
        seed_list = [int(s.strip()) for s in args.seeds.split(",")]
    else:
        try:
            n = int(args.seeds)
            seed_list = list(range(n))
        except ValueError:
            print(f"Invalid --seeds: {args.seeds}")
            sys.exit(1)

    if args.quick:
        args.epochs = 500
        args.efs_pop = 10
        args.efs_gen = 8

    main(seed_list=seed_list, epochs=args.epochs, efs_pop=args.efs_pop, efs_gen=args.efs_gen)
