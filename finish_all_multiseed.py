"""
Comprehensive multi-seed experiment pipeline.

Closes every gap named in Prof. Dr. Asiksoy's 2026-06-29 review:

  - Multiple seeds + statistical significance (Task 1)
  - 2D Poisson benchmark (Task 2)
  - GradNorm + ReLoBRaLo + MOEA/D baselines (Task 3)
  - Larger NSGA-II budget, configurable (Task 4)
  - Noise robustness sweep sigma in {0.01, 0.05, 0.1} (Task 5a)
  - Explicit nadir reference point for hypervolume (cross-cutting)

For each combination of (PDE, algorithm, noise level) we run the
experiment for N seeds, then compute Wilcoxon signed-rank tests across
seeds against the fixed-weight baseline.

Usage:
  python finish_all_multiseed.py --seeds 5
  python finish_all_multiseed.py --seeds 10 --quick
  python finish_all_multiseed.py --seeds 5 --nsga2-pop 40 --nsga2-gen 30
  python finish_all_multiseed.py --seeds 5 --skip-noise-sweep --skip-2d
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

# Register all PDEs (including the new poisson2d)
import pde_burgers         # noqa: F401
import pde_poisson         # noqa: F401
import pde_poisson_2d      # noqa: F401

from moead_optimizer import run_moead
from multi_seed_runner import (
    compare_methods_across_seeds,
    run_multi_seed_experiment,
    statistical_significance_test,
)
from nsga2_optimizer import pareto_nondominated, run_nsga2, run_nsga3
from pareto_metrics import hypervolume, spacing
from run_grid import run_grid
from train import train_pinn

RESULTS = ROOT / "results"


# ---------------------------------------------------------------------------
# Pareto-level metrics with explicit nadir
# ---------------------------------------------------------------------------

def compute_pareto_metrics(
    df: pd.DataFrame,
    ref_padding: float = 0.10,
) -> dict[str, float]:
    """Compute hypervolume (with explicit nadir) and spacing for a DataFrame.

    The nadir is taken per-Pareto as worst-observed value + 10% of span,
    so HV is always well-defined. We log the nadir explicitly so reviewers
    can reproduce the calculation -- the professor specifically flagged
    that nadir must be stated.
    """
    pareto = pareto_nondominated(df)
    if len(pareto) < 1:
        return {"pareto_n": 0, "hv": float("nan"), "spacing": float("nan")}

    pts = pareto[["f1_l2_error", "f2_pde_residual", "f3_data_budget"]].to_numpy()
    lo = pts.min(axis=0)
    hi = pts.max(axis=0)
    span = np.maximum(hi - lo, 1e-8)
    nadir = hi + ref_padding * span

    return {
        "pareto_n": len(pareto),
        "hv": hypervolume(pts, ref_point=nadir),
        "spacing": spacing(pts),
        "nadir_f1": float(nadir[0]),
        "nadir_f2": float(nadir[1]),
        "nadir_f3": float(nadir[2]),
        "lo_f1": float(lo[0]),
        "lo_f2": float(lo[1]),
        "lo_f3": float(lo[2]),
    }


# ---------------------------------------------------------------------------
# Single-seed experiment functions (one per row in the master CSV)
# ---------------------------------------------------------------------------

def run_poisson_grid_single_seed(seed: int, **kwargs) -> pd.DataFrame:
    """Poisson 1D grid: full lambda sweep across collocation sizes."""
    return run_grid(
        pde_name="poisson",
        epochs=kwargs.get("epochs", 2000),
        seed=seed,
        lam_values=[0.1, 1.0, 10.0],
        n_values=[100, 200, 300],
        out_name=None,
    )


def run_poisson2d_grid_single_seed(seed: int, **kwargs) -> pd.DataFrame:
    """Poisson 2D grid: 2D-spatial benchmark (Task 2)."""
    return run_grid(
        pde_name="poisson2d",
        epochs=kwargs.get("epochs", 2000),
        seed=seed,
        lam_values=[0.1, 1.0, 10.0],
        n_values=[100, 200, 300],
        out_name=None,
    )


def run_burgers_baseline_single_seed(seed: int, **kwargs) -> pd.DataFrame:
    """Burgers 1D with fixed lambda = (10, 1)."""
    _, metrics = train_pinn(
        "burgers",
        lam_data=10.0,
        lam_pde=1.0,
        epochs=kwargs.get("epochs", 2000),
        seed=seed,
        verbose=False,
    )
    return pd.DataFrame(
        [
            {
                **metrics,
                "pde": "burgers",
                "algorithm": "baseline_fixed",
                "n_obs": 0,
                "obs_noise": 0,
                "balancer": "fixed",
            }
        ]
    )


def run_burgers_grid_clean_single_seed(seed: int, **kwargs) -> pd.DataFrame:
    """Burgers 1D grid (clean)."""
    return run_grid(
        pde_name="burgers",
        epochs=kwargs.get("epochs", 2000),
        seed=seed,
        lam_values=[1.0, 10.0],
        n_values=[100, 200, 300],
        out_name=None,
    )


def run_burgers_grid_noisy_single_seed(
    seed: int,
    noise_sigma: float = 0.05,
    **kwargs,
) -> pd.DataFrame:
    """Burgers 1D grid (sparse noisy)."""
    return run_grid(
        pde_name="burgers",
        n_obs=20,
        obs_noise=noise_sigma,
        epochs=kwargs.get("epochs", 2000),
        seed=seed,
        lam_values=[1.0, 10.0],
        n_values=[100, 200, 300],
        out_name=None,
    )


def run_burgers_gradnorm_single_seed(seed: int, **kwargs) -> pd.DataFrame:
    """Burgers 1D grid using GradNorm adaptive loss balancing (Task 3a)."""
    return run_grid(
        pde_name="burgers",
        epochs=kwargs.get("epochs", 2000),
        seed=seed,
        lam_values=[1.0],
        n_values=[100, 200, 300],
        out_name=None,
        balancer="gradnorm",
    )


def run_burgers_relobralo_single_seed(seed: int, **kwargs) -> pd.DataFrame:
    """Burgers 1D grid using ReLoBRaLo adaptive loss balancing (Task 3a)."""
    return run_grid(
        pde_name="burgers",
        epochs=kwargs.get("epochs", 2000),
        seed=seed,
        lam_values=[1.0],
        n_values=[100, 200, 300],
        out_name=None,
        balancer="relobralo",
    )


def run_burgers_nsga2_single_seed(
    seed: int,
    pop_size: int = 6,
    n_gen: int = 4,
    **kwargs,
) -> pd.DataFrame:
    """Burgers 1D NSGA-II (Task 4: budget configurable via flags)."""
    return run_nsga2(
        pde_name="burgers",
        pop_size=pop_size,
        n_gen=n_gen,
        epochs=kwargs.get("epochs", 2500),
        seed=seed,
    )


def run_burgers_moead_single_seed(
    seed: int,
    pop_size: int = 15,
    n_gen: int = 8,
    **kwargs,
) -> pd.DataFrame:
    """Burgers 1D MOEA/D (Task 3b baseline)."""
    return run_moead(
        pde_name="burgers",
        pop_size=pop_size,
        n_gen=n_gen,
        epochs=kwargs.get("epochs", 2500),
        seed=seed,
    )


def run_burgers_nsga3_single_seed(
    seed: int,
    pop_size: int = 15,
    n_gen: int = 8,
    **kwargs,
) -> pd.DataFrame:
    """Burgers 1D NSGA-III (Task 3b baseline)."""
    return run_nsga3(
        pde_name="burgers",
        pop_size=pop_size,
        n_gen=n_gen,
        epochs=kwargs.get("epochs", 2500),
        seed=seed,
    )


# ---------------------------------------------------------------------------
# Master pipeline
# ---------------------------------------------------------------------------

def main(
    seed_list: list[int],
    quick: bool = False,
    epochs: int = 2000,
    nsga2_pop: int = 6,
    nsga2_gen: int = 4,
    moead_pop: int = 15,
    moead_gen: int = 8,
    skip_noise_sweep: bool = False,
    skip_2d: bool = False,
    noise_levels: list[float] | None = None,
    run_hv_per_seed: bool = True,
) -> None:
    """Execute the full multi-seed experiment pipeline.

    Parameters
    ----------
    seed_list : list[int]
        Independent random seeds to repeat each experiment under.
    quick : bool
        Reduced budget for sanity check.
    epochs, nsga2_pop, nsga2_gen, moead_pop, moead_gen : int
        Per-experiment budgets. Professor asks for nsga2_pop>=40, gen>=30
        once GPU is available; CPU use stays smaller (6/4).
    skip_noise_sweep, skip_2d : bool
        Disable specific experiments for time-constrained runs.
    noise_levels : list[float], optional
        Override the default {0.01, 0.05, 0.1} sweep.
    run_hv_per_seed : bool
        If True, attach per-seed hypervolume and spacing to the master CSV.
    """
    if quick:
        epochs = min(epochs, 500)
        nsga2_pop = min(nsga2_pop, 6)
        nsga2_gen = min(nsga2_gen, 2)
        moead_pop = min(moead_pop, 10)
        moead_gen = min(moead_gen, 4)
    if noise_levels is None:
        noise_levels = [0.01, 0.05, 0.10]

    print("=" * 78)
    print("MULTI-SEED PINN-MOO-EFS PIPELINE")
    print(f"Seeds: {seed_list}")
    print(f"epochs={epochs}  nsga2_pop={nsga2_pop}  nsga2_gen={nsga2_gen}")
    print(f"moead_pop={moead_pop}  moead_gen={moead_gen}")
    print(f"noise_levels={noise_levels}")
    print(f"Results -> {RESULTS}")
    print("=" * 78)

    # ------------------------------------------------------------------
    # Define the experiments to run
    # ------------------------------------------------------------------
    experiments: list[tuple[str, callable, Path, str]] = [
        ("poisson_grid_clean",     run_poisson_grid_single_seed,
         RESULTS / "multiseed_poisson_grid",     "Poisson 1D grid"),
        ("burgers_baseline_fixed", run_burgers_baseline_single_seed,
         RESULTS / "multiseed_burgers_baseline", "Burgers 1D baseline (fixed)"),
        ("burgers_grid_clean",     run_burgers_grid_clean_single_seed,
         RESULTS / "multiseed_burgers_grid_clean", "Burgers 1D grid (clean)"),
        ("burgers_grid_gradnorm",  run_burgers_gradnorm_single_seed,
         RESULTS / "multiseed_burgers_grid_gradnorm", "Burgers 1D grid (GradNorm)"),
        ("burgers_grid_relobralo", run_burgers_relobralo_single_seed,
         RESULTS / "multiseed_burgers_grid_relobralo", "Burgers 1D grid (ReLoBRaLo)"),
        ("burgers_nsga2_clean",    run_burgers_nsga2_single_seed,
         RESULTS / "multiseed_burgers_nsga2", "Burgers 1D NSGA-II"),
        ("burgers_nsga3_clean",    run_burgers_nsga3_single_seed,
         RESULTS / "multiseed_burgers_nsga3", "Burgers 1D NSGA-III"),
        ("burgers_moead_clean",    run_burgers_moead_single_seed,
         RESULTS / "multiseed_burgers_moead", "Burgers 1D MOEA/D"),
    ]
    if not skip_2d:
        experiments.append(
            ("poisson2d_grid_clean", run_poisson2d_grid_single_seed,
             RESULTS / "multiseed_poisson2d_grid", "Poisson 2D grid (NEW)"),
        )
    if not skip_noise_sweep:
        for sigma in noise_levels:
            tag = f"burgers_grid_noisy_s{int(round(sigma * 100)):02d}"
            fn = lambda seed, _sigma=sigma, **kw: run_burgers_grid_noisy_single_seed(
                seed=seed, noise_sigma=_sigma, **kw
            )
            experiments.append(
                (tag, fn,
                 RESULTS / f"multiseed_{tag}",
                 f"Burgers 1D grid (sigma={sigma})"),
            )

    # Wrap NSGA-II / MOEA-D experiments so they receive configured budgets.
# The runner (run_multi_seed_experiment) already forwards epochs via fn_kwargs,
# so we only inject pop_size / n_gen here -- DO NOT also pass epochs or it
# collides with the runner's epochs and Python raises "got multiple values".
    def _wrap(fn, **extra):
        def wrapped(seed: int, **kw) -> pd.DataFrame:
            return fn(seed=seed, **extra, **kw)
        return wrapped

    # Patch budgets into NSGA-II / NSGA-III / MOEA-D runners
    nsga2_fn = _wrap(run_burgers_nsga2_single_seed, pop_size=nsga2_pop, n_gen=nsga2_gen)
    nsga3_fn = _wrap(run_burgers_nsga3_single_seed, pop_size=nsga2_pop, n_gen=nsga2_gen)
    moead_fn = _wrap(run_burgers_moead_single_seed, pop_size=moead_pop, n_gen=moead_gen)
    # Replace the placeholder entries
    by_name = {exp[0]: exp for exp in experiments}
    by_name["burgers_nsga2_clean"] = (
        "burgers_nsga2_clean", nsga2_fn, by_name["burgers_nsga2_clean"][2],
        by_name["burgers_nsga2_clean"][3],
    )
    by_name["burgers_nsga3_clean"] = (
        "burgers_nsga3_clean", nsga3_fn, by_name["burgers_nsga3_clean"][2],
        by_name["burgers_nsga3_clean"][3],
    )
    by_name["burgers_moead_clean"] = (
        "burgers_moead_clean", moead_fn, by_name["burgers_moead_clean"][2],
        by_name["burgers_moead_clean"][3],
    )
    experiments = list(by_name.values())

    # ------------------------------------------------------------------
    # Run each experiment across seeds, aggregating into per-experiment dirs
    # ------------------------------------------------------------------
    all_runs_frames: list[pd.DataFrame] = []
    summary_frames: list[pd.DataFrame] = []
    pareto_metrics_per_seed: dict[str, list[dict]] = {}

    for i, (name, fn, out_dir, descr) in enumerate(experiments, 1):
        print(f"\n[{i}/{len(experiments)}] {descr}")
        result = run_multi_seed_experiment(
            experiment_fn=fn,
            seeds=seed_list,
            output_dir=out_dir,
            experiment_name=name,
            verbose=True,
            epochs=epochs,
        )
        all_runs = result["all_runs"]
        summary = result["summary_stats"]
        all_runs_frames.append(all_runs.assign(experiment=name))
        summary_frames.append(summary.assign(experiment=name))

        # Per-seed HV/spacing if requested
        if run_hv_per_seed and {"f1_l2_error", "f2_pde_residual", "f3_data_budget"} <= set(all_runs.columns):
            hv_rows = []
            for seed, gdf in all_runs.groupby("seed"):
                m = compute_pareto_metrics(gdf)
                m["seed"] = seed
                m["experiment"] = name
                hv_rows.append(m)
            pareto_metrics_per_seed[name] = hv_rows

    # ------------------------------------------------------------------
    # Master all-runs CSV
    # ------------------------------------------------------------------
    master_runs = pd.concat(all_runs_frames, ignore_index=True)
    master_runs.to_csv(RESULTS / "multiseed_all_runs.csv", index=False)
    print(f"\n[OK] Saved {len(master_runs)} total runs -> {RESULTS / 'multiseed_all_runs.csv'}")

    master_summary = pd.concat(summary_frames, ignore_index=True)
    master_summary.to_csv(RESULTS / "multiseed_master_summary.csv", index=False)
    print(f"[OK] Master summary -> {RESULTS / 'multiseed_master_summary.csv'}")

    # ------------------------------------------------------------------
    # Per-seed Pareto metrics (HV with explicit nadir + spacing)
    # ------------------------------------------------------------------
    if pareto_metrics_per_seed:
        flat = [row for rows in pareto_metrics_per_seed.values() for row in rows]
        pareto_df = pd.DataFrame(flat)
        pareto_df.to_csv(RESULTS / "multiseed_pareto_metrics.csv", index=False)
        print(
            f"[OK] Pareto metrics (hv, spacing, nadir) for {len(pareto_df)} "
            f"(experiment, seed) pairs -> {RESULTS / 'multiseed_pareto_metrics.csv'}"
        )

    # ------------------------------------------------------------------
    # Wilcoxon significance tests: baseline vs each candidate algorithm
    # ------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("STATISTICAL COMPARISON ANALYSIS (Wilcoxon signed-rank, paired)")
    print("=" * 78)

    baseline_runs = master_runs[master_runs["algorithm"] == "baseline_fixed"]
    if len(baseline_runs) == 0:
        baseline_runs = master_runs[master_runs["experiment"] == "burgers_baseline_fixed"]

    sig_rows = []
    for candidate_algo in ["grid", "grid_gradnorm", "grid_relobralo", "nsga2", "nsga3", "moead"]:
        c_runs = master_runs[master_runs["algorithm"] == candidate_algo]
        if len(c_runs) == 0:
            continue
        # Compute best-per-seed for each side (paired by seed)
        try:
            base_by_seed = baseline_runs.groupby("seed")["f1_l2_error"].mean()
            cand_by_seed = c_runs.groupby("seed")["f1_l2_error"].apply(
                lambda g: g.min()
            )
            common = base_by_seed.index.intersection(cand_by_seed.index)
            if len(common) >= 3:
                test = statistical_significance_test(
                    base_by_seed.loc[common].values,
                    cand_by_seed.loc[common].values,
                    metric_name="L2_error",
                )
                sig_rows.append({
                    "comparison": f"baseline_fixed_vs_{candidate_algo}",
                    "metric": "best_L2_per_seed",
                    "baseline_mean": float(base_by_seed.loc[common].mean()),
                    "baseline_std": float(base_by_seed.loc[common].std()),
                    "candidate_mean": float(cand_by_seed.loc[common].mean()),
                    "candidate_std": float(cand_by_seed.loc[common].std()),
                    "p_value": test["p_value"],
                    "significant": test["significant"],
                    "n_seeds_paired": len(common),
                })
                print(f"\n[{candidate_algo}] baseline vs. candidate (best L2 per seed):")
                print(f"  baseline: {base_by_seed.loc[common].mean():.4f} +/- {base_by_seed.loc[common].std():.4f}")
                print(f"  candidate: {cand_by_seed.loc[common].mean():.4f} +/- {cand_by_seed.loc[common].std():.4f}")
                print(f"  {test['interpretation']}")
        except Exception as e:
            print(f"  [WARN] Comparison vs {candidate_algo} failed: {e}")

    # ------------------------------------------------------------------
    # Pareto-quality comparison: HV across algorithms (paired by seed)
    # ------------------------------------------------------------------
    if pareto_metrics_per_seed:
        print("\n" + "=" * 78)
        print("PARETO QUALITY (hypervolume + spacing across seeds)")
        print("=" * 78)
        pareto_long = pd.DataFrame(
            [r for rows in pareto_metrics_per_seed.values() for r in rows]
        )
        for experiment, sub in pareto_long.groupby("experiment"):
            hv_mean = sub["hv"].mean()
            hv_std = sub["hv"].std()
            sp_mean = sub["spacing"].mean()
            sp_std = sub["spacing"].std()
            nadir_f1 = sub["nadir_f1"].mean()
            nadir_f2 = sub["nadir_f2"].mean()
            nadir_f3 = sub["nadir_f3"].mean()
            print(
                f"\n[{experiment}] across {len(sub)} seeds: "
                f"HV={hv_mean:.6f} +/- {hv_std:.6f}  "
                f"spacing={sp_mean:.4f} +/- {sp_std:.4f}  "
                f"(nadir f1={nadir_f1:.3f}, f2={nadir_f2:.3f}, f3={nadir_f3:.1f})"
            )

    # Save significance tests
    if sig_rows:
        sig_df = pd.DataFrame(sig_rows)
        sig_path = RESULTS / "multiseed_significance_tests.csv"
        sig_df.to_csv(sig_path, index=False)
        print(f"\n[OK] Significance tests -> {sig_path}")

    print("\n" + "=" * 78)
    print("MULTI-SEED EXPERIMENTS COMPLETE")
    print("=" * 78)
    print(f"\nKey outputs in {RESULTS}/:")
    print(f"  * multiseed_all_runs.csv                  -- all individual runs")
    print(f"  * multiseed_master_summary.csv            -- mean/std per config")
    print(f"  * multiseed_pareto_metrics.csv            -- HV + spacing per seed (with explicit nadir)")
    print(f"  * multiseed_significance_tests.csv        -- Wilcoxon comparisons")
    print(f"  * multiseed_<experiment>/                 -- per-experiment subdirs")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Comprehensive multi-seed PINN-MOO-EFS pipeline."
    )
    parser.add_argument("--seeds", type=str, default="5",
                        help="Number of seeds OR comma-separated list")
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--nsga2-pop", type=int, default=6,
                        help="NSGA-II population size (professor recommends >=40 with GPU)")
    parser.add_argument("--nsga2-gen", type=int, default=4,
                        help="NSGA-II generations (professor recommends >=30 with GPU)")
    parser.add_argument("--moead-pop", type=int, default=15)
    parser.add_argument("--moead-gen", type=int, default=8)
    parser.add_argument("--quick", action="store_true",
                        help="Small budgets for sanity check")
    parser.add_argument("--skip-noise-sweep", action="store_true")
    parser.add_argument("--skip-2d", action="store_true")
    parser.add_argument("--noise-levels", type=str, default="0.01,0.05,0.10",
                        help="Comma-separated noise sigmas for the noise sweep")
    args = parser.parse_args()

    if "," in args.seeds:
        seed_list = [int(s.strip()) for s in args.seeds.split(",")]
    else:
        try:
            seed_list = list(range(int(args.seeds)))
        except ValueError:
            print(f"Invalid --seeds: {args.seeds}")
            sys.exit(1)

    noise_levels = [float(s.strip()) for s in args.noise_levels.split(",")]
    main(
        seed_list=seed_list,
        quick=args.quick,
        epochs=args.epochs,
        nsga2_pop=args.nsga2_pop,
        nsga2_gen=args.nsga2_gen,
        moead_pop=args.moead_pop,
        moead_gen=args.moead_gen,
        skip_noise_sweep=args.skip_noise_sweep,
        skip_2d=args.skip_2d,
        noise_levels=noise_levels,
    )
