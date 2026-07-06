"""
Baseline comparison table for the PINN-MOO-EFS paper.

This is the deliverable Prof. Dr. Asiksoy explicitly asked for:
   "You must actually implement at least these two [GradNorm, ReLoBRaLo]
    and compare them on the same problems."
   "On the multi-objective side NSGA-II alone is not enough; if you add
    a comparison with MOEA/D or NSGA-III your claims about front quality
    will be much stronger."

Reads the multi-seed outputs from `results/multiseed_*/` and produces:

  1. results/baseline_comparison_table.csv -- per-(baseline, pde) summary
  2. results/baseline_comparison_table.md  -- human-readable Markdown
  3. results/baseline_significance_tests.csv -- Wilcoxon signed-rank tests

Usage:
  python baseline_comparison.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Force UTF-8 stdout/stderr
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
from pareto_metrics import hypervolume, spacing

RESULTS = ROOT / "results"


# Mapping from experiment name in multiseed_* dirs to a clean baseline label.
BASELINE_LABELS: dict[str, str] = {
    "burgers_baseline_fixed":      "Fixed-weight PINN",
    "burgers_grid_clean":          "Grid search",
    "burgers_grid_gradnorm":       "GradNorm (Chen et al. 2018)",
    "burgers_grid_relobralo":      "ReLoBRaLo (Bischof & Kraus 2021)",
    "burgers_nsga2_clean":         "NSGA-II (Deb et al. 2002)",
    "burgers_nsga3_clean":         "NSGA-III (Jain & Deb 2014)",
    "burgers_moead_clean":         "MOEA/D (Zhang & Li 2007)",
}


def _best_per_seed(df: pd.DataFrame, metric: str) -> pd.Series:
    """For each seed, return the best (lowest) value of `metric`."""
    return df.groupby("seed")[metric].min()


def _hv_per_seed(df: pd.DataFrame) -> dict[int, float]:
    """Per-seed hypervolume with explicit nadir (worst + 10% span)."""
    out = {}
    for seed, g in df.groupby("seed"):
        pts = g[["f1_l2_error", "f2_pde_residual", "f3_data_budget"]].to_numpy()
        lo = pts.min(axis=0)
        hi = pts.max(axis=0)
        span = np.maximum(hi - lo, 1e-8)
        nadir = hi + 0.10 * span
        out[int(seed)] = float(hypervolume(pts, ref_point=nadir))
    return out


def _spacing_per_seed(df: pd.DataFrame) -> dict[int, float]:
    """Per-seed spacing on the Pareto front (only)."""
    out = {}
    from nsga2_optimizer import pareto_nondominated
    for seed, g in df.groupby("seed"):
        pareto = pareto_nondominated(g)
        if len(pareto) < 2:
            out[int(seed)] = float("nan")
        else:
            pts = pareto[["f1_l2_error", "f2_pde_residual", "f3_data_budget"]].to_numpy()
            out[int(seed)] = float(spacing(pts))
    return out


def collect_baseline(name: str) -> pd.DataFrame | None:
    """Read the per-experiment all_seeds CSV for a baseline."""
    candidates = [
        RESULTS / f"multiseed_{name}" / f"{name}_all_seeds.csv",
        RESULTS / f"{name}_all_seeds.csv",
    ]
    for c in candidates:
        if c.exists():
            return pd.read_csv(c)
    return None


def build_summary_table() -> pd.DataFrame:
    """Build the master baseline comparison summary table.

    Columns:
        baseline           -- human label
        pde                -- 'burgers' or 'poisson'
        n_seeds            -- count
        f1_l2_mean         -- best per seed, then mean across seeds
        f1_l2_std          -- std across seeds
        f2_res_mean        -- same for f2
        f2_res_std
        f3_budget_mean     -- mean total data budget of best point per seed
        hv_mean            -- hypervolume across seeds
        hv_std
        spacing_mean
        spacing_std
        nadir_f1, nadir_f2, nadir_f3   -- explicit reference point (mean across seeds)
    """
    rows = []
    for exp_name, label in BASELINE_LABELS.items():
        df = collect_baseline(exp_name)
        if df is None or df.empty:
            print(f"[skip] {label}: no data found for {exp_name}")
            continue

        # Detect PDE name(s) in this experiment (usually 1)
        pdes = df["pde"].unique().tolist() if "pde" in df.columns else ["burgers"]
        for pde in pdes:
            sub = df[df["pde"] == pde] if "pde" in df.columns else df

            f1_best = _best_per_seed(sub, "f1_l2_error")
            f2_best = _best_per_seed(sub, "f2_pde_residual")
            f3_best = _best_per_seed(sub, "f3_data_budget")
            hv_dict = _hv_per_seed(sub)
            sp_dict = _spacing_per_seed(sub)

            rows.append({
                "baseline": label,
                "experiment": exp_name,
                "pde": pde,
                "n_seeds": len(f1_best),
                "f1_l2_mean": float(f1_best.mean()),
                "f1_l2_std":  float(f1_best.std()),
                "f2_res_mean": float(f2_best.mean()),
                "f2_res_std":  float(f2_best.std()),
                "f3_budget_mean": float(f3_best.mean()),
                "f3_budget_std":  float(f3_best.std()),
                "hv_mean": float(np.mean(list(hv_dict.values()))),
                "hv_std":  float(np.std(list(hv_dict.values()))),
                "spacing_mean": float(np.nanmean(list(sp_dict.values()))),
                "spacing_std":  float(np.nanstd(list(sp_dict.values()))),
                "seeds": ",".join(str(s) for s in sorted(f1_best.index.tolist())),
            })
    return pd.DataFrame(rows)


def build_significance_table(summary: pd.DataFrame) -> pd.DataFrame:
    """Wilcoxon signed-rank tests: every baseline vs Fixed-weight PINN.

    Pairs: best f1 per seed (paired by seed) on Burgers (clean).
    """
    if summary.empty:
        return pd.DataFrame()

    base = summary[(summary["baseline"] == "Fixed-weight PINN") & (summary["pde"] == "burgers")]
    if base.empty:
        print("[skip] Fixed-weight PINN baseline not found for burgers; skipping Wilcoxon.")
        return pd.DataFrame()

    base_df = collect_baseline("burgers_baseline_fixed")
    if base_df is None:
        return pd.DataFrame()

    base_f1 = _best_per_seed(base_df, "f1_l2_error")
    base_f2 = _best_per_seed(base_df, "f2_pde_residual")

    sig_rows = []
    for exp_name, label in BASELINE_LABELS.items():
        if label == "Fixed-weight PINN":
            continue
        cand = collect_baseline(exp_name)
        if cand is None:
            continue
        cand_f1 = _best_per_seed(cand, "f1_l2_error")
        cand_f2 = _best_per_seed(cand, "f2_pde_residual")

        common = base_f1.index.intersection(cand_f1.index)
        if len(common) < 3:
            continue

        t_f1 = statistical_significance_test(
            base_f1.loc[common].values,
            cand_f1.loc[common].values,
            metric_name="best_f1_l2",
        )
        t_f2 = statistical_significance_test(
            base_f2.loc[common].values,
            cand_f2.loc[common].values,
            metric_name="best_f2_residual",
        )

        sig_rows.append({
            "candidate": label,
            "metric_f1_p_value": t_f1["p_value"],
            "metric_f1_significant": t_f1["significant"],
            "f1_candidate_mean": float(cand_f1.loc[common].mean()),
            "f1_candidate_std":  float(cand_f1.loc[common].std()),
            "metric_f2_p_value": t_f2["p_value"],
            "metric_f2_significant": t_f2["significant"],
            "f2_candidate_mean": float(cand_f2.loc[common].mean()),
            "f2_candidate_std":  float(cand_f2.loc[common].std()),
            "n_seeds_paired": len(common),
            "f1_interpretation": t_f1["interpretation"],
            "f2_interpretation": t_f2["interpretation"],
        })

    return pd.DataFrame(sig_rows)


def write_markdown(summary: pd.DataFrame, sig: pd.DataFrame, path: Path) -> None:
    """Write a human-readable Markdown comparison table."""
    lines = []
    lines.append("# Baseline Comparison Table")
    lines.append("")
    lines.append("Multi-seed results for the PINN-MOO-EFS paper.")
    lines.append("Each baseline was run across multiple seeds; metrics report")
    lines.append("**best (lowest) value per seed, then mean +/- std across seeds**.")
    lines.append("")
    if summary.empty:
        lines.append("(No baseline results found.)")
        path.write_text("\n".join(lines), encoding="utf-8")
        return

    burgers = summary[summary["pde"] == "burgers"]
    poisson = summary[summary["pde"] == "poisson"]

    if not burgers.empty:
        lines.append("## Burgers (clean, N_obs = 0)")
        lines.append("")
        lines.append("| Baseline | n_seeds | f1 L2 (mean +/- std) | f2 residual (mean +/- std) | f3 budget | HV (mean +/- std) | spacing (mean +/- std) |")
        lines.append("|----------|---------|----------------------|----------------------------|-----------|-------------------|------------------------|")
        for _, r in burgers.sort_values("f1_l2_mean").iterrows():
            lines.append(
                f"| {r['baseline']} | {r['n_seeds']} | "
                f"{r['f1_l2_mean']:.4f} +/- {r['f1_l2_std']:.4f} | "
                f"{r['f2_res_mean']:.4f} +/- {r['f2_res_std']:.4f} | "
                f"{r['f3_budget_mean']:.0f} +/- {r['f3_budget_std']:.0f} | "
                f"{r['hv_mean']:.4f} +/- {r['hv_std']:.4f} | "
                f"{r['spacing_mean']:.4f} +/- {r['spacing_std']:.4f} |"
            )
        lines.append("")

    if not poisson.empty:
        lines.append("## Poisson 1D (clean, N_obs = 0)")
        lines.append("")
        lines.append("| Baseline | n_seeds | f1 L2 (mean +/- std) | f2 residual (mean +/- std) | f3 budget | HV (mean +/- std) | spacing (mean +/- std) |")
        lines.append("|----------|---------|----------------------|----------------------------|-----------|-------------------|------------------------|")
        for _, r in poisson.sort_values("f1_l2_mean").iterrows():
            lines.append(
                f"| {r['baseline']} | {r['n_seeds']} | "
                f"{r['f1_l2_mean']:.4f} +/- {r['f1_l2_std']:.4f} | "
                f"{r['f2_res_mean']:.4f} +/- {r['f2_res_std']:.4f} | "
                f"{r['f3_budget_mean']:.0f} +/- {r['f3_budget_std']:.0f} | "
                f"{r['hv_mean']:.4f} +/- {r['hv_std']:.4f} | "
                f"{r['spacing_mean']:.4f} +/- {r['spacing_std']:.4f} |"
            )
        lines.append("")

    if not sig.empty:
        lines.append("## Statistical significance (Wilcoxon signed-rank, paired by seed)")
        lines.append("")
        lines.append("Reference baseline: Fixed-weight PINN (lambda = (10, 1)).")
        lines.append("Significance threshold: p < 0.05.")
        lines.append("")
        lines.append("| Candidate | f1 p-value | f1 sig? | f1 candidate mean | f2 p-value | f2 sig? | f2 candidate mean |")
        lines.append("|-----------|-----------|---------|-------------------|-----------|---------|-------------------|")
        for _, r in sig.iterrows():
            lines.append(
                f"| {r['candidate']} | "
                f"{r['metric_f1_p_value']:.4f} | "
                f"{'YES' if r['metric_f1_significant'] else 'no'} | "
                f"{r['f1_candidate_mean']:.4f} +/- {r['f1_candidate_std']:.4f} | "
                f"{r['metric_f2_p_value']:.4f} | "
                f"{'YES' if r['metric_f2_significant'] else 'no'} | "
                f"{r['f2_candidate_mean']:.4f} +/- {r['f2_candidate_std']:.4f} |"
            )
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    print("Building baseline comparison ...")
    summary = build_summary_table()
    print(f"Found {len(summary)} (baseline, pde) rows")
    if not summary.empty:
        out_csv = RESULTS / "baseline_comparison_table.csv"
        summary.to_csv(out_csv, index=False)
        print(f"[OK] {out_csv}")

        sig = build_significance_table(summary)
        if not sig.empty:
            sig_csv = RESULTS / "baseline_significance_tests.csv"
            sig.to_csv(sig_csv, index=False)
            print(f"[OK] {sig_csv}")
        else:
            sig = pd.DataFrame()
    else:
        sig = pd.DataFrame()

    md_path = RESULTS / "baseline_comparison_table.md"
    write_markdown(summary, sig, md_path)
    print(f"[OK] {md_path}")

    if not summary.empty:
        print("\n=== Summary ===")
        print(summary[["baseline", "pde", "n_seeds", "f1_l2_mean", "f1_l2_std",
                        "f2_res_mean", "f2_res_std", "hv_mean", "hv_std"]].to_string(index=False))


if __name__ == "__main__":
    main()