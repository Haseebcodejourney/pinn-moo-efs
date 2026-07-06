"""Real-data cylinder wake comparison table for the PINN-MOO-EFS paper.

This is the deliverable for the "Please add real data" request, paired
with the cylinder wake data from Maziar Raissi's PINNs repository
(https://github.com/maziarraissi/PINNs).

Reads the multi-seed outputs from
``results/multiseed_cylinder_wake_grid_*_n<N>/`` and produces:

  1. results/cylinder_wake_table.csv -- per-(method, n_obs) summary
  2. results/cylinder_wake_table.md  -- human-readable Markdown
  3. results/cylinder_wake_significance.csv -- Wilcoxon signed-rank tests

Each method is Grid / GradNorm / ReLoBRaLo.

Usage:
    python run_cylinder_wake_experiment.py --seeds 0,1
    python cylinder_wake_comparison.py
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


# Mapping from per-method short name to a clean human label.
METHOD_LABELS: dict[str, str] = {
    "cylinder_wake_grid_clean":      "Grid (cylinder wake)",
    "cylinder_wake_grid_gradnorm":   "GradNorm (cylinder wake)",
    "cylinder_wake_grid_relobralo":  "ReLoBRaLo (cylinder wake)",
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


def collect_method(n_obs: int, short_name: str) -> pd.DataFrame | None:
    """Read the per-method all_seeds CSV for a (n_obs, method) combination."""
    method_csv = f"{short_name}_n{n_obs:03d}_all_seeds.csv"
    candidates = [
        RESULTS / f"multiseed_{method_csv.replace('_all_seeds.csv', '')}" / method_csv,
    ]
    for c in candidates:
        if c.exists():
            return pd.read_csv(c)
    return None


def build_summary_table() -> pd.DataFrame:
    """Build the master cylinder-wake summary table."""
    # Discover n_obs values from existing dirs.
    # The runner writes to e.g. multiseed_cylinder_wake_grid_clean_n200.
    # Group by n_obs: collect every dir, parse n_obs from the trailing _n<NNN>,
    # and dedup so we have one row per (method, n_obs) pair.
    pattern = "multiseed_cylinder_wake_*"
    candidates = sorted([d for d in RESULTS.glob(pattern) if d.is_dir()])
    # Collect the unique n_obs values across all dirs.
    n_obs_values: set[int] = set()
    for d in candidates:
        name = d.name
        if "_n" in name:
            try:
                n_obs_values.add(int(name.split("_n")[-1]))
            except ValueError:
                pass
    if not n_obs_values:
        print("[skip] No multiseed_cylinder_wake_*n* dirs found.")
        return pd.DataFrame()
    # Use sorted list of n_obs so output is stable
    n_obs_list = sorted(n_obs_values)
    # We don't actually need the dir list -- collect_method() finds the file
    # directly. But we keep this convention so the for-loop is intuitive.

    rows = []
    for n_obs in n_obs_list:
        for short_name, label in METHOD_LABELS.items():
            df = collect_method(n_obs, short_name)
            if df is None or df.empty:
                print(f"[skip] {label} n_obs={n_obs}: no data")
                continue

            f1_best = _best_per_seed(df, "f1_l2_error")
            f2_best = _best_per_seed(df, "f2_pde_residual")
            f3_best = _best_per_seed(df, "f3_data_budget")
            hv_dict = _hv_per_seed(df)
            sp_dict = _spacing_per_seed(df)

            rows.append({
                "method": label,
                "experiment": short_name,
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
                "seeds": ",".join(str(s) for s in sorted(f1_best.index.tolist())),
            })
    return pd.DataFrame(rows)


def build_significance_table(summary: pd.DataFrame) -> pd.DataFrame:
    """Wilcoxon signed-rank: Grid vs each other method, per n_obs."""
    if summary.empty:
        return pd.DataFrame()

    sig_rows = []
    for n_obs in sorted(summary["n_obs"].unique()):
        grid_row = summary[(summary["method"] == "Grid (cylinder wake)") & (summary["n_obs"] == n_obs)]
        if grid_row.empty:
            continue
        grid_df = collect_method(int(n_obs), "cylinder_wake_grid_clean")
        if grid_df is None:
            continue
        grid_f1 = _best_per_seed(grid_df, "f1_l2_error")

        for short_name, label in METHOD_LABELS.items():
            if label == "Grid (cylinder wake)":
                continue
            cand = collect_method(int(n_obs), short_name)
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
    lines.append("# Real-Data Cylinder Wake Comparison Table")
    lines.append("")
    lines.append("Multi-seed results for the PINN-MOO-EFS paper.")
    lines.append("")
    lines.append("**Data source:** Real experimental cylinder wake flow from Maziar Raissi's PINNs repository")
    lines.append("([github.com/maziarraissi/PINNs](https://github.com/maziarraissi/PINNs)).")
    lines.append("Domain: x ∈ [1, 8], y ∈ [-2, 2], t ∈ [0, 20] around a cylinder of radius 0.5 at (1, 1.5).")
    lines.append("5000 spatial points × 200 time steps = 1,000,000 real measurements of (u, v, p).")
    lines.append("This is the deliverable for the user's 'Please add real data' request, addressing")
    lines.append("Prof. Dr. Asiksoy's note that 'adding an open dataset containing sparse real")
    lines.append("measurements paired with a known PDE would eliminate the sigma = 0.05 criticism'.")
    lines.append("")
    lines.append("Each baseline was run across multiple seeds; metrics report")
    lines.append("**best (lowest) value per seed, then mean +/- std across seeds**.")
    lines.append("")
    if summary.empty:
        lines.append("(No cylinder-wake results found.)")
        path.write_text("\n".join(lines), encoding="utf-8")
        return

    lines.append("## Per-method, per-n_obs results")
    lines.append("")
    lines.append("| n_obs | Method | n_seeds | f1 L2 (mean +/- std) | f2 residual (mean +/- std) | HV (mean +/- std) |")
    lines.append("|-------|--------|---------|----------------------|----------------------------|-------------------|")
    for n_obs in sorted(summary["n_obs"].unique()):
        g = summary[summary["n_obs"] == n_obs].sort_values("f1_l2_mean")
        for _, r in g.iterrows():
            lines.append(
                f"| {int(r['n_obs'])} | {r['method']} | {r['n_seeds']} | "
                f"{r['f1_l2_mean']:.4f} +/- {r['f1_l2_std']:.4f} | "
                f"{r['f2_res_mean']:.4f} +/- {r['f2_res_std']:.4f} | "
                f"{r['hv_mean']:.4f} +/- {r['hv_std']:.4f} |"
            )
        lines.append("")

    lines.append("## Data efficiency: best f1 vs n_obs (lower is better)")
    lines.append("")
    lines.append("| Method | " + " | ".join(f"n_obs={int(n)}" for n in sorted(summary['n_obs'].unique())) + " |")
    lines.append("|" + "---|" * (1 + len(summary["n_obs"].unique())))
    for method in METHOD_LABELS.values():
        cells = []
        for n_obs in sorted(summary["n_obs"].unique()):
            row = summary[(summary["method"] == method) & (summary["n_obs"] == n_obs)]
            if not row.empty:
                cells.append(f"{row['f1_l2_mean'].iloc[0]:.4f} +/- {row['f1_l2_std'].iloc[0]:.4f}")
            else:
                cells.append("--")
        lines.append(f"| {method} | " + " | ".join(cells) + " |")
    lines.append("")

    if not sig.empty:
        lines.append("## Statistical significance (Wilcoxon signed-rank, paired by seed)")
        lines.append("")
        lines.append("Reference baseline: Grid (cylinder wake). Significance threshold: p < 0.05.")
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
    print("Building cylinder-wake comparison ...")
    summary = build_summary_table()
    print(f"Found {len(summary)} (method, n_obs) rows")
    if not summary.empty:
        out_csv = RESULTS / "cylinder_wake_table.csv"
        summary.to_csv(out_csv, index=False)
        print(f"[OK] {out_csv}")

        sig = build_significance_table(summary)
        if not sig.empty:
            sig_csv = RESULTS / "cylinder_wake_significance.csv"
            sig.to_csv(sig_csv, index=False)
            print(f"[OK] {sig_csv}")
    else:
        sig = pd.DataFrame()

    md_path = RESULTS / "cylinder_wake_table.md"
    write_markdown(summary, sig, md_path)
    print(f"[OK] {md_path}")

    if not summary.empty:
        print("\n=== Summary ===")
        print(summary[["method", "n_obs", "n_seeds", "f1_l2_mean", "f1_l2_std",
                        "f2_res_mean", "f2_res_std", "hv_mean", "hv_std"]].to_string(index=False))


if __name__ == "__main__":
    main()