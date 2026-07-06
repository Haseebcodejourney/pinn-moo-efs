"""
Multi-seed experiment runner with statistical analysis.

Enables repetition of experiments across multiple seeds and computes:
- Mean and standard deviation for all metrics
- Statistical significance tests (Wilcoxon signed-rank)
- Confidence intervals

Usage:
  python -c "from multi_seed_runner import run_multi_seed_experiment; run_multi_seed_experiment(...)"
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

# Force UTF-8 stdout/stderr (Windows cp1252 cannot encode non-ASCII glyphs)
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

warnings.filterwarnings("ignore", category=DeprecationWarning)


def run_multi_seed_experiment(
    experiment_fn,
    seeds: list[int],
    output_dir: Path,
    experiment_name: str,
    verbose: bool = True,
    **fn_kwargs,
) -> dict[str, Any]:
    """
    Execute experiment_fn across multiple seeds and aggregate statistics.

    Parameters
    ----------
    experiment_fn : callable
        Function that takes (seed: int, **fn_kwargs) and returns pd.DataFrame.
    seeds : list[int]
        Random seeds to use (e.g., [0, 1, 2, 3, 4])
    output_dir : Path
        Directory to save multi-seed results
    experiment_name : str
        Descriptive name for this experiment
    verbose : bool
        Print progress
    fn_kwargs
        Extra keyword arguments forwarded to experiment_fn every call
        (e.g. epochs=500). All experiment_fn signatures in this project
        accept ``epochs`` via **kwargs.

    Returns
    -------
    dict with keys:
        - "all_runs": DataFrame with all individual runs (seed column added)
        - "summary_stats": DataFrame with mean ± std for each unique config
        - "seed_variance": DataFrame showing variance metrics
        - "metadata": Dict with seed list and experiment config
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_runs = []

    # Phase 1: Run experiment across all seeds
    if verbose:
        print(f"\n{'='*70}")
        print(f"Multi-Seed Experiment: {experiment_name}")
        print(f"Seeds: {seeds}")
        print(f"{'='*70}\n")

    for i, seed in enumerate(seeds, 1):
        if verbose:
            print(f"[{i}/{len(seeds)}] Running with seed={seed}")
        try:
            result_df = experiment_fn(seed=seed, **fn_kwargs)
            result_df["seed"] = seed
            all_runs.append(result_df)
        except Exception as e:
            if verbose:
                print(f"  [WARN] Error with seed={seed}: {e}")
            continue

    if not all_runs:
        raise RuntimeError(f"No successful runs for {experiment_name}")

    all_runs_df = pd.concat(all_runs, ignore_index=True)

    # Phase 2: Compute statistics by config group
    # Identify metric columns (numeric, not grouping keys)
    metric_cols = [
        c for c in all_runs_df.columns
        if c not in {"seed", "pde", "algorithm", "n_obs", "obs_noise", "balancer", "hidden_dim", "n_layers"}
        and pd.api.types.is_numeric_dtype(all_runs_df[c])
    ]

    # Group by all non-seed, non-metric columns
    groupby_cols = [
        c for c in all_runs_df.columns
        if c not in metric_cols and c != "seed"
    ]
    if not groupby_cols:
        groupby_cols = ["algorithm"]

    summary_stats = []
    seed_variances = []

    for group_key, group_df in all_runs_df.groupby(groupby_cols, dropna=False):
        row_dict = dict(zip(groupby_cols, group_key if isinstance(group_key, tuple) else [group_key]))
        row_dict["n_seeds"] = len(group_df)

        # Compute mean ± std for each metric
        for metric in metric_cols:
            values = group_df[metric].dropna()
            if len(values) > 0:
                row_dict[f"{metric}_mean"] = values.mean()
                row_dict[f"{metric}_std"] = values.std() if len(values) > 1 else 0.0
                row_dict[f"{metric}_min"] = values.min()
                row_dict[f"{metric}_max"] = values.max()
            else:
                row_dict[f"{metric}_mean"] = np.nan
                row_dict[f"{metric}_std"] = np.nan
                row_dict[f"{metric}_min"] = np.nan
                row_dict[f"{metric}_max"] = np.nan

        summary_stats.append(row_dict)
        seed_variances.append(row_dict)

    summary_stats_df = pd.DataFrame(summary_stats)
    seed_variance_df = pd.DataFrame(seed_variances)

    # Phase 3: Save results
    all_runs_path = output_dir / f"{experiment_name}_all_seeds.csv"
    summary_path = output_dir / f"{experiment_name}_summary_stats.csv"
    metadata_path = output_dir / f"{experiment_name}_metadata.json"

    all_runs_df.to_csv(all_runs_path, index=False)
    summary_stats_df.to_csv(summary_path, index=False)

    metadata = {
        "experiment_name": experiment_name,
        "n_seeds": len(seeds),
        "seeds": seeds,
        "successful_seeds": sorted(all_runs_df["seed"].unique().tolist()),
        "n_total_runs": len(all_runs_df),
        "metric_columns": metric_cols,
    }

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    if verbose:
        print(f"\n[OK] All runs: {all_runs_path}")
        print(f"[OK] Summary stats: {summary_path}")
        print(f"[OK] Metadata: {metadata_path}")
        print(f"\nStatistics computed for {len(summary_stats_df)} configurations across {len(seeds)} seeds.")

    return {
        "all_runs": all_runs_df,
        "summary_stats": summary_stats_df,
        "seed_variance": seed_variance_df,
        "metadata": metadata,
    }


def statistical_significance_test(
    method1_values: np.ndarray,
    method2_values: np.ndarray,
    metric_name: str = "metric",
    alpha: float = 0.05,
) -> dict[str, Any]:
    """
    Perform Wilcoxon signed-rank test between two methods.

    Assumes paired comparisons: method1_values[i] vs method2_values[i] for seed i.

    Parameters
    ----------
    method1_values : np.ndarray
        Metric values from method 1 (one per seed)
    method2_values : np.ndarray
        Metric values from method 2 (one per seed)
    metric_name : str
        Name of the metric being tested
    alpha : float
        Significance level (default 0.05)

    Returns
    -------
    dict with keys:
        - "statistic": Wilcoxon test statistic
        - "p_value": Two-tailed p-value
        - "significant": Whether p_value < alpha
        - "mean_diff": Mean difference (method2 - method1)
        - "interpretation": Plain-language result
    """
    if len(method1_values) != len(method2_values):
        raise ValueError("method1_values and method2_values must have same length")

    if len(method1_values) < 2:
        return {
            "statistic": np.nan,
            "p_value": np.nan,
            "significant": False,
            "mean_diff": np.mean(method2_values) - np.mean(method1_values),
            "interpretation": f"Cannot perform test with {len(method1_values)} pairs",
        }

    try:
        stat, p_val = wilcoxon(method1_values, method2_values, alternative="two-sided")
    except ValueError as e:
        return {
            "statistic": np.nan,
            "p_value": np.nan,
            "significant": False,
            "mean_diff": np.mean(method2_values) - np.mean(method1_values),
            "interpretation": f"Wilcoxon test failed: {e}",
        }

    mean_diff = np.mean(method2_values) - np.mean(method1_values)
    is_sig = p_val < alpha

    if is_sig:
        direction = "better" if mean_diff < 0 else "worse"
        interp = f"Method 2 is significantly {direction} (p={p_val:.4f} < {alpha})"
    else:
        interp = f"No significant difference (p={p_val:.4f} >= {alpha})"

    return {
        "statistic": float(stat),
        "p_value": float(p_val),
        "significant": bool(is_sig),
        "mean_diff": float(mean_diff),
        "interpretation": interp,
    }


def compare_methods_across_seeds(
    all_runs_df: pd.DataFrame,
    method1: str,
    method2: str,
    metric: str,
    groupby_cols: list[str] | None = None,
) -> pd.DataFrame:
    """
    Compare two methods across seeds using statistical tests.

    Parameters
    ----------
    all_runs_df : pd.DataFrame
        DataFrame with columns: seed, algorithm (or similar method indicator), metric column
    method1 : str
        First method name (value in algorithm column)
    method2 : str
        Second method name (value in algorithm column)
    metric : str
        Metric column to compare
    groupby_cols : list[str], optional
        Additional columns to group by (e.g., ["pde", "n_obs"])

    Returns
    -------
    pd.DataFrame
        Comparison results with statistical significance
    """
    if groupby_cols is None:
        groupby_cols = []

    results = []

    # Group by additional columns if specified
    if groupby_cols:
        for group_key, group_df in all_runs_df.groupby(groupby_cols, dropna=False):
            group_dict = dict(zip(groupby_cols, group_key if isinstance(group_key, tuple) else [group_key]))

            m1_vals = group_df[group_df["algorithm"] == method1][metric].values
            m2_vals = group_df[group_df["algorithm"] == method2][metric].values

            # Align by seed for paired test
            m1_df = group_df[group_df["algorithm"] == method1][["seed", metric]].rename(columns={metric: "m1"})
            m2_df = group_df[group_df["algorithm"] == method2][["seed", metric]].rename(columns={metric: "m2"})
            merged = m1_df.merge(m2_df, on="seed", how="inner")

            if len(merged) >= 2:
                test_result = statistical_significance_test(merged["m1"].values, merged["m2"].values, metric)
                group_dict.update({
                    f"{method1}_{metric}_mean": merged["m1"].mean(),
                    f"{method2}_{metric}_mean": merged["m2"].mean(),
                    **test_result,
                })
                results.append(group_dict)
    else:
        # No grouping, single comparison
        m1_df = all_runs_df[all_runs_df["algorithm"] == method1][["seed", metric]].rename(columns={metric: "m1"})
        m2_df = all_runs_df[all_runs_df["algorithm"] == method2][["seed", metric]].rename(columns={metric: "m2"})
        merged = m1_df.merge(m2_df, on="seed", how="inner")

        if len(merged) >= 2:
            test_result = statistical_significance_test(merged["m1"].values, merged["m2"].values, metric)
            results.append({
                f"{method1}_{metric}_mean": merged["m1"].mean(),
                f"{method2}_{metric}_mean": merged["m2"].mean(),
                **test_result,
            })

    return pd.DataFrame(results) if results else pd.DataFrame()
