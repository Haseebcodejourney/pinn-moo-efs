"""Generate comparison figures and summary_table.csv."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from efs_optimizer import run_efs
from experiment_log import FIG_DIR, RESULTS_DIR
from fuzzy_rules import FuzzyRuleParams, select_compromise
from nsga2_optimizer import pareto_nondominated
from pareto_metrics import OBJECTIVE_COLS, hypervolume, sanitize_objectives_df, spacing
from pde_registry import get_pde
import pde_burgers  # noqa: F401
import pde_poisson  # noqa: F401
from selection_methods import compare_selections
from train import train_pinn


def _load_latest_moo(pde: str | None = None) -> pd.DataFrame:
    patterns = ["*nsga2*.csv", "*nsga3*.csv", "*grid*.csv", "moo_results.csv"]
    files = []
    for pat in patterns:
        files.extend(RESULTS_DIR.glob(pat))
    files = sorted(set(files), key=lambda p: p.stat().st_mtime, reverse=True)
    if pde:
        files = [f for f in files if pde in f.name]
    if not files:
        raise FileNotFoundError("No MOO/grid results found in results/.")
    return pd.read_csv(files[0]), files[0]


def plot_solution(model, pde_name: str, device: torch.device, out_path: Path) -> None:
    pde = get_pde(pde_name)
    coords = pde.evaluation_grid().to(device)

    with torch.no_grad():
        pred = model(coords).cpu().numpy()

    if pde.input_dim == 2:
        from pde_burgers import T_MAX, T_MIN, X_MAX, X_MIN

        nt, nx = 100, 256
        t = np.linspace(T_MIN, T_MAX, nt)
        x = np.linspace(X_MIN, X_MAX, nx)
        tt, xx = np.meshgrid(t, x, indexing="ij")
        u_pred = pred.reshape(nt, nx)
        u_true = pde.reference_solution(tt, xx)

        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        for ax, data, title in zip(axes, [u_true, u_pred, np.abs(u_pred - u_true)], ["Reference", "PINN", "|Error|"]):
            im = ax.imshow(data, origin="lower", aspect="auto", extent=[X_MIN, X_MAX, T_MIN, T_MAX])
            ax.set_title(title)
            ax.set_xlabel("x")
            ax.set_ylabel("t")
            plt.colorbar(im, ax=ax, fraction=0.046)
    else:
        from pde_poisson import X_MAX, X_MIN

        x = np.linspace(X_MIN, X_MAX, pred.shape[0])
        u_true = pde.reference_solution(x, None)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(x, u_true, label="Reference", lw=2)
        ax.plot(x, pred.ravel(), "--", label="PINN")
        ax.set_xlabel("x")
        ax.set_ylabel("u(x)")
        ax.legend()
        ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_pareto(df: pd.DataFrame, out_dir: Path, prefix: str = "") -> None:
    pareto = pareto_nondominated(df)
    f3 = "f3_data_budget" if "f3_data_budget" in df.columns else "f3_n_collocation"

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(df["f1_l2_error"], df["f2_pde_residual"], c="lightgray", label="All", s=35)
    ax.scatter(pareto["f1_l2_error"], pareto["f2_pde_residual"], c="crimson", label="Pareto", s=55)
    ax.set_xlabel("f1: Relative L2 error")
    ax.set_ylabel("f2: Mean PDE residual")
    ax.set_title("Accuracy vs Physics Consistency")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / f"{prefix}fig_pareto_f1_f2.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(df["f1_l2_error"], df[f3], c="lightgray", label="All", s=35)
    ax.scatter(pareto["f1_l2_error"], pareto[f3], c="steelblue", label="Pareto", s=55)
    ax.set_xlabel("f1: Relative L2 error")
    ax.set_ylabel("f3: Data budget")
    ax.set_title("Accuracy vs Data Cost")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / f"{prefix}fig_pareto_f1_f3.png", dpi=200)
    plt.close(fig)


def plot_efs_comparison(static_loss: float, evolved_loss: float, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(["Static fuzzy", "Evolved fuzzy (EFS)"], [static_loss, evolved_loss], color=["gray", "seagreen"])
    ax.set_ylabel("Balanced loss (lower better)")
    ax.set_title("EFS vs Static Fuzzy Selection")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_data_scarcity(summary: pd.DataFrame, out_path: Path) -> None:
    if "n_obs" not in summary.columns:
        return
    sub = summary.groupby(["pde", "n_obs"], as_index=False)["f1_l2_error"].min()
    fig, ax = plt.subplots(figsize=(6, 4))
    for pde in sub["pde"].unique():
        s = sub[sub["pde"] == pde]
        ax.plot(s["n_obs"], s["f1_l2_error"], marker="o", label=pde)
    ax.set_xlabel("N_obs (sparse observations)")
    ax.set_ylabel("Best L2 error")
    ax.set_title("Data Scarcity Effect")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def build_summary_table(df: pd.DataFrame, efs_params: FuzzyRuleParams | None = None) -> pd.DataFrame:
    df = sanitize_objectives_df(df)
    pareto_df = pareto_nondominated(df)
    if pareto_df.empty:
        raise ValueError("No valid Pareto points (all runs missing f1/f2/f3). Re-run grid or nsga2.")
    pareto_rows = pareto_df.to_dict(orient="records")
    selections = compare_selections(pareto_rows, efs_params=efs_params)

    hv = hypervolume(pareto_rows)
    sp = spacing(pareto_rows)

    rows = []
    meta = {
        "pde": df["pde"].iloc[0] if "pde" in df.columns else "burgers",
        "algorithm": df["algorithm"].iloc[0] if "algorithm" in df.columns else "unknown",
        "n_obs": int(df["n_obs"].iloc[0]) if "n_obs" in df.columns else 0,
        "obs_noise": float(df["obs_noise"].iloc[0]) if "obs_noise" in df.columns else 0.0,
        "hypervolume": hv,
        "spacing": sp,
        "n_pareto": len(pareto_rows),
        "n_runs": len(df),
    }

    if "lam_data" in df.columns:
        baseline_mask = (df["lam_data"] - 1.0).abs() + (df["lam_pde"] - 1.0).abs()
        baseline = df.iloc[baseline_mask.idxmin()].to_dict()
        rows.append({"selection_method": "baseline_fixed_weights", **meta, **baseline})

    for sel in selections:
        rows.append({**meta, **sel})

    return pd.DataFrame(rows)


def aggregate_all_summaries() -> pd.DataFrame:
    parts = []
    for path in RESULTS_DIR.glob("*_summary.csv"):
        parts.append(pd.read_csv(path))
    if (RESULTS_DIR / "summary_table.csv").exists():
        return pd.read_csv(RESULTS_DIR / "summary_table.csv")
    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts, ignore_index=True)
    out.to_csv(RESULTS_DIR / "summary_table.csv", index=False)
    return out


def main(pde: str | None = None, run_efs_flag: bool = True, epochs: int = 2500) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df, src_path = _load_latest_moo(pde)
    df = sanitize_objectives_df(df)
    if df.empty:
        raise ValueError(f"No complete runs in {src_path}. Re-run grid/nsga2 (some rows had missing metrics).")
    pde_name = str(df["pde"].iloc[0]) if "pde" in df.columns else (pde or "burgers")
    prefix = f"{pde_name}_"

    plot_pareto(df, FIG_DIR, prefix=prefix)

    efs_params = None
    if run_efs_flag:
        pareto_path = src_path.parent / src_path.name.replace(".csv", "_pareto.csv")
        if not pareto_path.exists():
            pareto_path = src_path
        efs_out = run_efs(pareto_path, RESULTS_DIR, pop_size=16, n_gen=20, seed=42)
        efs_params = efs_out["params"]
        plot_efs_comparison(
            efs_out["result"]["static_fuzzy_balanced_loss"],
            efs_out["result"]["evolved_fuzzy_balanced_loss"],
            FIG_DIR / f"{prefix}fig_efs_comparison.png",
        )

    summary = build_summary_table(df, efs_params=efs_params)
    summary_path = RESULTS_DIR / f"{pde_name}_summary.csv"
    summary.to_csv(summary_path, index=False)

    pareto = pareto_nondominated(df)
    best = select_compromise(pareto.to_dict(orient="records"), use_fuzzy=True)
    model, _ = train_pinn(
        pde_name=pde_name,
        lam_data=best.get("lam_data", 1.0),
        lam_pde=best.get("lam_pde", 1.0),
        n_collocation=int(best.get("n_collocation", 200)),
        n_obs=int(best.get("n_obs", 0)),
        obs_noise=float(best.get("obs_noise", 0.0)),
        epochs=epochs,
        seed=42,
        device=device,
    )
    plot_solution(model, pde_name, device, FIG_DIR / f"{prefix}fig_solution.png")

    aggregate_all_summaries()
    all_summary = pd.read_csv(RESULTS_DIR / "summary_table.csv") if (RESULTS_DIR / "summary_table.csv").exists() else summary
    plot_data_scarcity(all_summary, FIG_DIR / "fig_data_scarcity.png")

    print(f"Figures -> {FIG_DIR}")
    print(f"Summary -> {summary_path}")
    print(f"Master table -> {RESULTS_DIR / 'summary_table.csv'}")


if __name__ == "__main__":
    main()
