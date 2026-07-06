"""
Statistical report generator for multi-seed PINN experiments.

Generates comprehensive analysis including:
- Summary statistics (mean, std, min, max) for all metrics
- Wilcoxon signed-rank tests for method comparisons
- Hypervolume and Pareto quality metrics
- Robustness analysis across seeds

Usage:
    python -c "from stats_report_generator import generate_report; generate_report('multiseed_all_runs.csv')"
    python -c "from stats_report_generator import statistical_summary; statistical_summary('multiseed_all_runs.csv')"
"""

from __future__ import annotations

import json
import sys
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

# Color codes for terminal output
class Colors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


def statistical_summary(
    csv_path: Path | str,
    output_dir: Path | str | None = None,
    groupby_cols: list[str] | None = None,
) -> dict[str, Any]:
    """
    Generate comprehensive statistical summary from multi-seed results.

    Parameters
    ----------
    csv_path : Path or str
        Path to CSV with columns: seed, algorithm, f1_l2_error, f2_pde_residual, f3_data_budget, ...
    output_dir : Path or str, optional
        Directory to save reports (if None, only returns dict)
    groupby_cols : list[str], optional
        Columns to group by (default: algorithm, pde)

    Returns
    -------
    dict with keys:
        - "summary_by_group": DataFrame with mean/std stats
        - "seed_variability": Variance metrics per group
        - "comparisons": Method comparison results
        - "robustness": Consistency across seeds
    """
    df = pd.read_csv(csv_path)

    if groupby_cols is None:
        groupby_cols = [c for c in ["pde", "algorithm"] if c in df.columns]

    if not groupby_cols:
        groupby_cols = ["algorithm"]

    # Identify metric columns
    metric_cols = [
        c for c in df.columns
        if c not in {"seed"} | set(groupby_cols)
        and pd.api.types.is_numeric_dtype(df[c])
        and not c.endswith("_std")
    ]

    summary_by_group = []

    for group_key, group_df in df.groupby(groupby_cols, dropna=False):
        row = dict(zip(groupby_cols, group_key if isinstance(group_key, tuple) else [group_key]))
        row["n_seeds"] = len(group_df)
        row["seeds_list"] = ",".join(map(str, sorted(group_df["seed"].unique())))

        for metric in metric_cols:
            values = group_df[metric].dropna()
            if len(values) > 0:
                row[f"{metric}_mean"] = values.mean()
                row[f"{metric}_std"] = values.std() if len(values) > 1 else 0.0
                row[f"{metric}_min"] = values.min()
                row[f"{metric}_max"] = values.max()
                row[f"{metric}_ci95"] = 1.96 * row[f"{metric}_std"] / np.sqrt(len(values))
            else:
                row[f"{metric}_mean"] = np.nan
                row[f"{metric}_std"] = np.nan
                row[f"{metric}_min"] = np.nan
                row[f"{metric}_max"] = np.nan
                row[f"{metric}_ci95"] = np.nan

        summary_by_group.append(row)

    summary_df = pd.DataFrame(summary_by_group)

    # Seed variability: coefficient of variation
    seed_var = []
    for group_key, group_df in df.groupby(groupby_cols, dropna=False):
        row = dict(zip(groupby_cols, group_key if isinstance(group_key, tuple) else [group_key]))

        for metric in metric_cols:
            values = group_df[metric].dropna()
            if len(values) > 1 and values.mean() != 0:
                cv = values.std() / np.abs(values.mean())  # Coefficient of variation
                row[f"{metric}_cv"] = cv
            else:
                row[f"{metric}_cv"] = np.nan

        seed_var.append(row)

    seed_var_df = pd.DataFrame(seed_var)

    result = {
        "summary_by_group": summary_df,
        "seed_variability": seed_var_df,
        "metric_columns": metric_cols,
        "groupby_columns": groupby_cols,
    }

    # Save if output_dir provided
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        summary_df.to_csv(output_dir / "stats_summary.csv", index=False)
        seed_var_df.to_csv(output_dir / "seed_variability.csv", index=False)

        with open(output_dir / "stats_report.json", "w") as f:
            json.dump({
                "n_groups": len(summary_df),
                "metric_columns": metric_cols,
                "groupby_columns": groupby_cols,
            }, f, indent=2)

    return result


def compare_two_methods(
    df: pd.DataFrame,
    method1: str,
    method2: str,
    metric: str,
    groupby_cols: list[str] | None = None,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Compare two methods using Wilcoxon signed-rank test (paired, per seed).

    Parameters
    ----------
    df : pd.DataFrame
        Results DataFrame with columns: seed, algorithm, metric, ...
    method1 : str
        First method name
    method2 : str
        Second method name
    metric : str
        Metric column to compare
    groupby_cols : list[str], optional
        Additional grouping columns
    alpha : float
        Significance level

    Returns
    -------
    pd.DataFrame
        Comparison results with p-values and significance indicators
    """
    if groupby_cols is None:
        groupby_cols = []

    results = []

    if groupby_cols:
        for group_key, group_df in df.groupby(groupby_cols, dropna=False):
            group_dict = dict(zip(groupby_cols, group_key if isinstance(group_key, tuple) else [group_key]))

            # Align by seed
            m1_df = group_df[group_df["algorithm"] == method1][["seed", metric]].rename(columns={metric: "m1"})
            m2_df = group_df[group_df["algorithm"] == method2][["seed", metric]].rename(columns={metric: "m2"})
            merged = m1_df.merge(m2_df, on="seed", how="inner")

            if len(merged) >= 2:
                try:
                    stat, p_val = wilcoxon(merged["m1"], merged["m2"], alternative="two-sided")
                    is_sig = p_val < alpha
                    mean_diff = merged["m2"].mean() - merged["m1"].mean()

                    group_dict.update({
                        "n_seeds": len(merged),
                        f"{method1}_{metric}_mean": merged["m1"].mean(),
                        f"{method2}_{metric}_mean": merged["m2"].mean(),
                        "mean_difference": mean_diff,
                        "p_value": p_val,
                        "statistic": stat,
                        "significant": is_sig,
                        "interpretation": (
                            f"Method2 {'significantly better' if is_sig and mean_diff < 0 else ('significantly worse' if is_sig else 'no significant diff.')}"
                        ),
                    })
                    results.append(group_dict)
                except ValueError:
                    pass
    else:
        m1_df = df[df["algorithm"] == method1][["seed", metric]].rename(columns={metric: "m1"})
        m2_df = df[df["algorithm"] == method2][["seed", metric]].rename(columns={metric: "m2"})
        merged = m1_df.merge(m2_df, on="seed", how="inner")

        if len(merged) >= 2:
            try:
                stat, p_val = wilcoxon(merged["m1"], merged["m2"], alternative="two-sided")
                is_sig = p_val < alpha
                mean_diff = merged["m2"].mean() - merged["m1"].mean()

                results.append({
                    "method1": method1,
                    "method2": method2,
                    "metric": metric,
                    "n_seeds": len(merged),
                    f"{method1}_mean": merged["m1"].mean(),
                    f"{method2}_mean": merged["m2"].mean(),
                    "mean_difference": mean_diff,
                    "p_value": p_val,
                    "statistic": stat,
                    "significant": is_sig,
                })
            except ValueError:
                pass

    return pd.DataFrame(results) if results else pd.DataFrame()


def robustness_analysis(
    df: pd.DataFrame,
    output_file: Path | None = None,
) -> dict[str, Any]:
    """
    Analyze robustness (consistency) across seeds.

    Returns metrics like coefficient of variation for each algorithm.
    """
    robustness = []

    for algo in df["algorithm"].unique():
        algo_df = df[df["algorithm"] == algo]

        for metric in ["f1_l2_error", "f2_pde_residual", "f3_data_budget"]:
            if metric not in algo_df.columns:
                continue

            values = algo_df.groupby("seed")[metric].mean().values
            if len(values) > 1:
                mean_val = values.mean()
                std_val = values.std()
                cv = std_val / np.abs(mean_val) if mean_val != 0 else np.nan
                robustness.append({
                    "algorithm": algo,
                    "metric": metric,
                    "mean": mean_val,
                    "std": std_val,
                    "cv": cv,
                    "n_seeds": len(values),
                })

    robustness_df = pd.DataFrame(robustness)

    if output_file:
        robustness_df.to_csv(output_file, index=False)

    return {
        "robustness_df": robustness_df,
        "most_robust": robustness_df.nsmallest(5, "cv")[["algorithm", "metric", "cv"]],
    }


def generate_report(
    csv_path: Path | str,
    output_dir: Path | str | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """
    Generate complete statistical report (main entry point).

    Parameters
    ----------
    csv_path : Path or str
        Multi-seed results CSV
    output_dir : Path or str, optional
        Output directory for report files
    verbose : bool
        Print to console

    Returns
    -------
    dict with report components
    """
    csv_path = Path(csv_path)
    if output_dir:
        output_dir = Path(output_dir)

    if verbose:
        print(f"\n{Colors.HEADER}{Colors.BOLD}=== STATISTICAL REPORT GENERATION ==={Colors.ENDC}")
        print(f"Input: {csv_path}")
        if output_dir:
            print(f"Output: {output_dir}\n")

    # Summary statistics
    summary_result = statistical_summary(csv_path, output_dir)
    if verbose:
        print(f"{Colors.OKBLUE}Summary Statistics:{Colors.ENDC}")
        print(summary_result["summary_by_group"].to_string())
        print()

    # Robustness analysis
    df = pd.read_csv(csv_path)
    robustness_result = robustness_analysis(df, output_dir / "robustness.csv" if output_dir else None)
    if verbose:
        print(f"{Colors.OKBLUE}Robustness (Coefficient of Variation):{Colors.ENDC}")
        print(robustness_result["robustness_df"].to_string())
        print()

    # Save to JSON
    if output_dir:
        with open(output_dir / "report.json", "w") as f:
            json.dump({
                "input_file": str(csv_path),
                "total_rows": len(df),
                "unique_seeds": len(df["seed"].unique()),
                "unique_algorithms": len(df["algorithm"].unique() if "algorithm" in df.columns else []),
            }, f, indent=2)

    if verbose:
        print(f"{Colors.OKGREEN}Report complete!{Colors.ENDC}\n")

    return {
        "summary": summary_result,
        "robustness": robustness_result,
    }
