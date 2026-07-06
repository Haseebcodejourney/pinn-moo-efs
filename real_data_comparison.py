"""
Real-data Burgers comparison table for the PINN-MOO-EFS paper.

This is the deliverable for Prof. Dr. Asiksoy's request to add a real-data
experiment (otherwise the "data efficiency" third objective is weakly
motivated) and to eliminate the "sigma = 0.05 was chosen arbitrarily"
criticism.

Reads the multi-seed outputs from `results/multiseed_burgers_real_data_n*/`
and produces:

  1. results/real_data_table.csv -- per-(method, n_obs) summary
  2. results/real_data_table.md  -- human-readable Markdown
  3. results/real_data_significance.csv -- Wilcoxon signed-rank tests

Each method is Grid / GradNorm / ReLoBRaLo. The "real data" comes from a
high-fidelity Radau Burgers reference (1024 x 512, cached at
results/real_data/burgers_fine_nt1024_nx512.npz) with sigma derived from
the fine-grid bilinear interpolation residual (see
src/real_data_burgers.py::derive_measurement_noise).

Usage:
  PINN_USE_REAL_DATA=1 python run_real_data_experiment.py
  python real_data_comparison.py
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


# Mapping from per-method CSV (in per-n_obs aggregated dir) to a clean label.
METHOD_LABELS: dict[str, str] = {
    "burgers_grid_clean":      "Grid (real-data)",
    "burgers_grid_gradnorm":   "GradNorm (real-data)",
    "burgers_grid_relobralo":  "ReLoBRaLo (real-data)",
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


def collect_method(n_obs: int, method_csv: str) -> pd.DataFrame | None:
    """Read the per-method all_seeds CSV for a (n_obs, method) combination.

    The CSVs are written by run_real_data_experiment.py to
    results/multiseed_burgers_real_data_n<N>/<method>_n<N>_all_seeds.csv.
    We filter to that method within the aggregated n_obs CSV (which contains
    all methods concatenated together) -- this is the more robust path.
    """
    candidates = [
        RESULTS / f"multiseed_burgers_real_data_n{n_obs:03d}"
                 / f"burgers_{method_csv.split('_', 1)[1]}_n{n_obs:03d}_all_seeds.csv",
        RESULTS / f"multiseed_burgers_real_data_n{n_obs:03d}" / f"{method_csv}_all_seeds.csv",
    ]
    for c in candidates:
        if c.exists():
            return pd.read_csv(c)
    # Fallback: read aggregated CSV and filter by algorithm
    agg = RESULTS / f"multiseed_burgers_real_data_n{n_obs:03d}" / f"burgers_real_data_n{n_obs:03d}_all_seeds.csv"
    if agg.exists():
        df = pd.read_csv(agg)
        # algorithm names written by run_grid for each balancer
        algo_filter = {
            "burgers_grid_clean":     "grid",
            "burgers_grid_gradnorm":  "grid_gradnorm",
            "burgers_grid_relobralo": "grid_relobralo",
        }[method_csv]
        sub = df[df["algorithm"].astype(str) == algo_filter]
        if len(sub) > 0:
            return sub
    return None


def build_summary_table() -> pd.DataFrame:
    """Build the master real-data summary table.

    Columns:
        method            -- human label
        experiment        -- short name
        n_obs             -- number of observation points
        n_seeds           -- count
        f1_l2_mean/std    -- best per seed, then mean/std across seeds
        f2_res_mean/std   -- same for f2
        f3_budget_mean/std
        hv_mean/std
        spacing_mean/std
        real_data_sigma_mean
    """
    # Discover n_obs values from existing dirs
    n_obs_dirs = sorted(
        [d for d in RESULTS.glob("multiseed_burgers_real_data_n*") if d.is_dir()],
        key=lambda p: int(p.name.split("_n")[-1]),
    )
    if not n_obs_dirs:
        print("[skip] No multiseed_burgers_real_data_n* dirs found.")
        return pd.DataFrame()

    rows = []
    for n_dir in n_obs_dirs:
        n_obs = int(n_dir.name.split("_n")[-1])
        for exp_name, label in METHOD_LABELS.items():
            df = collect_method(n_obs, exp_name)
            if df is None or df.empty:
                print(f"[skip] {label} n_obs={n_obs}: no data")
                continue

            f1_best = _best_per_seed(df, "f1_l2_error")
            f2_best = _best_per_seed(df, "f2_pde_residual")
            f3_best = _best_per_seed(df, "f3_data_budget")
            hv_dict = _hv_per_seed(df)
            sp_dict = _spacing_per_seed(df)

            sigma_mean = float(df["real_data_sigma"].mean()) if "real_data_sigma" in df.columns else float("nan")

            rows.append({
                "method": label,
                "experiment": exp_name,
                "n_obs": n_obs,
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
                "real_data_sigma_mean": sigma_mean,
                "seeds": ",".join(str(s) for s in sorted(f1_best.index.tolist())),
            })
    return pd.DataFrame(rows)


def build_significance_table(summary: pd.DataFrame) -> pd.DataFrame:
    """Wilcoxon signed-rank: Grid (real-data) vs each other method, per n_obs."""
    if summary.empty:
        return pd.DataFrame()

    sig_rows = []
    for n_obs in sorted(summary["n_obs"].unique()):
        grid_row = summary[(summary["method"] == "Grid (real-data)") & (summary["n_obs"] == n_obs)]
        if grid_row.empty:
            continue
        grid_df = collect_method(int(n_obs), "burgers_grid_clean")
        if grid_df is None:
            continue
        grid_f1 = _best_per_seed(grid_df, "f1_l2_error")

        for exp_name, label in METHOD_LABELS.items():
            if label == "Grid (real-data)":
                continue
            cand = collect_method(int(n_obs), exp_name)
            if cand is None:
                continue
            cand_f1 = _best_per_seed(cand, "f1_l2_error")
            common = grid_f1.index.intersection(cand_f1.index)
            if len(common) < 3:
                continue
            t = statistical_significance_test(
                grid_f1.loc[common].values,
                cand_f1.loc[common].values,
                metric_name="best_f1_l2",
            )
            sig_rows.append({
                "n_obs": int(n_obs),
                "candidate": label,
                "p_value": t["p_value"],
                "significant": t["significant"],
                "candidate_mean": float(cand_f1.loc[common].mean()),
                "candidate_std":  float(cand_f1.loc[common].std()),
                "n_seeds_paired": len(common),
                "interpretation": t["interpretation"],
            })
    return pd.DataFrame(sig_rows)


def write_markdown(summary: pd.DataFrame, sig: pd.DataFrame, path: Path) -> None:
    """Write the human-readable Markdown table."""
    lines = []
    lines.append("# Real-Data Burgers Comparison Table")
    lines.append("")
    lines.append("Multi-seed results for the PINN-MOO-EFS paper.")
    lines.append("")
    lines.append("Observations are sampled from a high-fidelity Burgers reference")
    lines.append("(`src/real_data_burgers.py` -- 1024 x 512 Radau, rtol=1e-7, atol=1e-10)")
    lines.append("with sigma derived empirically from the fine-grid bilinear")
    lines.append("interpolation residual at random sample locations.")
    lines.append("")
    lines.append("This is the deliverable for Prof. Dr. Asiksoy's request:")
    lines.append("> 'If you have time, adding an open dataset containing sparse real")
    lines.append(">  measurements paired with a known PDE would also eliminate the")
    lines.append(">  sigma = 0.05 was chosen arbitrarily criticism.'")
    lines.append("")
    lines.append("Each baseline was run across multiple seeds; metrics report")
    lines.append("**best (lowest) value per seed, then mean +/- std across seeds**.")
    lines.append("")
    if summary.empty:
        lines.append("(No real-data results found.)")
        path.write_text("\n".join(lines), encoding="utf-8")
        return

    lines.append("## Per-method, per-n_obs results")
    lines.append("")
    lines.append("| n_obs | Method | n_seeds | sigma (derived) | f1 L2 (mean +/- std) | f2 residual (mean +/- std) | HV (mean +/- std) |")
    lines.append("|-------|--------|---------|------------------|----------------------|----------------------------|-------------------|")
    # Group by n_obs for readability
    for n_obs, g in summary.groupby("n_obs"):
        for _, r in g.sort_values("f1_l2_mean").iterrows():
            lines.append(
                f"| {int(r['n_obs'])} | {r['method']} | {r['n_seeds']} | "
                f"{r['real_data_sigma_mean']:.4f} | "
                f"{r['f1_l2_mean']:.4f} +/- {r['f1_l2_std']:.4f} | "
                f"{r['f2_res_mean']:.4f} +/- {r['f2_res_std']:.4f} | "
                f"{r['hv_mean']:.4f} +/- {r['hv_std']:.4f} |"
            )
        lines.append("")

    # Data-efficiency curve: per-method f1 vs n_obs
    lines.append("## Data efficiency: best f1 vs n_obs (lower is better)")
    lines.append("")
    lines.append("| Method | n_obs=20 | n_obs=50 | n_obs=100 |")
    lines.append("|--------|----------|----------|-----------|")
    methods_in_order = list(METHOD_LABELS.values())
    for method in methods_in_order:
        cells = []
        for n_obs in sorted(summary["n_obs"].unique()):
            row = summary[(summary["method"] == method) & (summary["n_obs"] == n_obs)]
            if not row.empty:
                cells.append(f"{row['f1_l2_mean'].iloc[0]:.4f} +/- {row['f1_l2_std'].iloc[0]:.4f}")
            else:
                cells.append("--")
        lines.append(f"| {method} | " + " | ".join(cells) + " |")
    lines.append("")

    # Significance tests
    if not sig.empty:
        lines.append("## Statistical significance (Wilcoxon signed-rank, paired by seed)")
        lines.append("")
        lines.append("Reference baseline: Grid (real-data). Significance threshold: p < 0.05.")
        lines.append("")
        lines.append("| n_obs | Candidate | p-value | Significant? | Candidate mean +/- std |")
        lines.append("|-------|-----------|---------|---------------|------------------------|")
        for _, r in sig.iterrows():
            lines.append(
                f"| {int(r['n_obs'])} | {r['candidate']} | "
                f"{r['p_value']:.4f} | {'YES' if r['significant'] else 'no'} | "
                f"{r['candidate_mean']:.4f} +/- {r['candidate_std']:.4f} |"
            )
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    print("Building real-data comparison ...")
    summary = build_summary_table()
    print(f"Found {len(summary)} (method, n_obs) rows")
    if not summary.empty:
        out_csv = RESULTS / "real_data_table.csv"
        summary.to_csv(out_csv, index=False)
        print(f"[OK] {out_csv}")

        sig = build_significance_table(summary)
        if not sig.empty:
            sig_csv = RESULTS / "real_data_significance.csv"
            sig.to_csv(sig_csv, index=False)
            print(f"[OK] {sig_csv}")
    else:
        sig = pd.DataFrame()

    md_path = RESULTS / "real_data_table.md"
    write_markdown(summary, sig, md_path)
    print(f"[OK] {md_path}")

    if not summary.empty:
        print("\n=== Summary ===")
        print(summary[["method", "n_obs", "n_seeds", "f1_l2_mean", "f1_l2_std",
                        "f2_res_mean", "f2_res_std", "hv_mean", "hv_std"]].to_string(index=False))


if __name__ == "__main__":
    main()