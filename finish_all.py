"""One-shot pipeline to finish experiments, figures, and summary_table.csv."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd
import pde_burgers as burgers_mod
from efs_optimizer import run_efs
from nsga2_optimizer import pareto_nondominated, run_nsga2
from plot_results import build_summary_table, main as plot_main
from run_grid import run_grid
from train import train_pinn

RESULTS = ROOT / "results"
SEED = 42
EPOCHS = 2000


def save(df: pd.DataFrame, name: str) -> Path:
    path = RESULTS / name
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    pareto_nondominated(df).to_csv(path.with_name(path.stem + "_pareto.csv"), index=False)
    print(f"saved {path}")
    return path


def main() -> None:
    burgers_mod.clear_reference_cache()
    print("=== 1/6 Poisson grid ===")
    save(
        run_grid(
            pde_name="poisson",
            epochs=EPOCHS,
            seed=SEED,
            lam_values=[0.1, 1.0, 10.0],
            n_values=[100, 200, 300],
            out_name="poisson_grid_n0_s0p0.csv",
        ),
        "poisson_grid_n0_s0p0.csv",
    )

    print("=== 2/6 Burgers baseline + grid (clean) ===")
    _, b0 = train_pinn("burgers", lam_data=10.0, lam_pde=1.0, epochs=EPOCHS, seed=SEED, verbose=True)
    save(pd.DataFrame([{**b0, "pde": "burgers", "algorithm": "baseline_fixed", "n_obs": 0, "obs_noise": 0}]),
         "burgers_baseline_fixed_n0_s0p0.csv")

    bgrid = run_grid(
        pde_name="burgers",
        epochs=EPOCHS,
        seed=SEED,
        lam_values=[1.0, 10.0],
        n_values=[100, 200, 300],
        out_name="burgers_grid_n0_s0p0.csv",
    )
    save(bgrid, "burgers_grid_n0_s0p0.csv")

    print("=== 3/6 Burgers sparse noisy grid ===")
    bgrid2 = run_grid(
        pde_name="burgers",
        n_obs=20,
        obs_noise=0.05,
        epochs=EPOCHS,
        seed=SEED,
        lam_values=[1.0, 10.0],
        n_values=[100, 200, 300],
        out_name="burgers_grid_n20_s0p05.csv",
    )
    save(bgrid2, "burgers_grid_n20_s0p05.csv")

    print("=== 4/6 Burgers NSGA-II ===")
    moo = run_nsga2(pde_name="burgers", pop_size=6, n_gen=4, epochs=2500, seed=SEED)
    save(moo, "burgers_nsga2_n0_s0p0.csv")

    print("=== 5/6 EFS on Burgers Pareto ===")
    pareto_path = RESULTS / "burgers_grid_n0_s0p0_pareto.csv"
    efs_out = run_efs(pareto_path, RESULTS, pop_size=16, n_gen=20, seed=SEED)
    print("EFS improves:", efs_out["result"]["efs_improves"])

    print("=== 6/6 Figures + master summary ===")
    plot_main(pde="burgers", epochs=EPOCHS)
    plot_main(pde="poisson", epochs=EPOCHS)

    summaries = []
    for path in [
        RESULTS / "burgers_grid_n0_s0p0.csv",
        RESULTS / "burgers_grid_n20_s0p05.csv",
        RESULTS / "burgers_nsga2_n0_s0p0.csv",
        RESULTS / "poisson_grid_n0_s0p0.csv",
    ]:
        if path.exists():
            df = pd.read_csv(path)
            efs_params = efs_out["params"] if "burgers" in path.name else None
            summaries.append(build_summary_table(df, efs_params=efs_params))

    master = pd.concat(summaries, ignore_index=True)
    master.to_csv(RESULTS / "summary_table.csv", index=False)
    print("DONE ->", RESULTS / "summary_table.csv")


if __name__ == "__main__":
    main()
