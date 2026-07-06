"""Run the missing multi-seed experiments so the master CSVs are at n=5 across the board.

What this script does (4 missing experiments):

  1. NSGA-III seeds 3, 4  -- extends existing 3-seed run (pop=5/gen=3/epochs=500)
  2. MOEA/D  seeds 3, 4  -- extends existing 3-seed run (pop=5/gen=3/epochs=500)
  3. Noise sweep sigma in {0.01, 0.10}  at 5 seeds (sigma=0.05 already at 5 seeds)
  4. 2D Poisson grid    at 5 seeds (currently zero seeds)

After this script:
  - Re-merge each extended experiment with the existing 3-seed CSVs.
  - Re-run assemble_master_outputs.py to rebuild the 4 master CSVs.

The script uses the existing run_multi_seed_experiment() helper so the
per-experiment CSVs have the same schema as the other 5-seed experiments.

Approximate wall-clock on this CPU-only machine (epochs=500):
  NSGA-III 2 seeds   ~ 15-25 min
  MOEA/D  2 seeds    ~ 15-25 min
  Noise sweep 2*5    ~ 10-15 min
  Poisson 2D grid    ~ 30-45 min
                          ----------
  Total              ~ 70-110 min
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from finish_all_multiseed import (
    run_burgers_grid_noisy_single_seed,
    run_burgers_moead_single_seed,
    run_burgers_nsga3_single_seed,
    run_poisson2d_grid_single_seed,
)
from multi_seed_runner import run_multi_seed_experiment
from nsga2_optimizer import run_nsga3
from moead_optimizer import run_moead

RESULTS = ROOT / "results"
EPOCHS = 500
POP, GEN = 5, 3  # match the existing 3-seed NSGA-III / MOEA-D budget

# Seeds to add to NSGA-III / MOEA-D (existing 3-seed run had 0, 1, 2)
EXTEND_SEEDS = [3, 4]


def append_seeds_to_existing(
    experiment_name: str,
    subdir: str,
    new_seeds: list[int],
    fn,
    *,
    fn_kwargs: dict,
) -> Path:
    """Run `fn` for each new seed, append the rows to the existing all_seeds.csv."""
    out_dir = RESULTS / subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{experiment_name}_all_seeds.csv"

    if csv_path.exists():
        existing = pd.read_csv(csv_path)
        existing_seeds = set(int(s) for s in existing["seed"].unique())
    else:
        existing = pd.DataFrame()
        existing_seeds = set()

    to_run = [s for s in new_seeds if s not in existing_seeds]
    if not to_run:
        print(f"[skip] {experiment_name}: seeds {new_seeds} already present")
        return csv_path

    print(f"\n[extend] {experiment_name}: existing seeds = {sorted(existing_seeds)}; "
          f"running new seeds = {to_run}")
    new_frames = []
    for s in to_run:
        t0 = time.time()
        df = fn(seed=s, **fn_kwargs)
        df["seed"] = s
        new_frames.append(df)
        print(f"  seed={s}: {len(df)} rows ({time.time()-t0:.1f}s)")

    combined = pd.concat([existing, *new_frames], ignore_index=True)
    combined.to_csv(csv_path, index=False)
    print(f"[OK] {csv_path} now has {len(combined)} rows "
          f"({combined['seed'].nunique()} seeds)")
    return csv_path


def run_fresh(name: str, subdir: str, fn, seeds: list[int], **fn_kwargs) -> Path:
    """Run a fresh n=5 multi-seed experiment."""
    out_dir = RESULTS / subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{name}_all_seeds.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        if df["seed"].nunique() >= len(seeds):
            print(f"[skip] {name}: already has {df['seed'].nunique()} seeds")
            return csv_path

    print(f"\n[fresh] {name}: running {len(seeds)} seeds at "
          f"{fn_kwargs}")
    t0 = time.time()
    result = run_multi_seed_experiment(
        experiment_fn=fn,
        seeds=seeds,
        output_dir=out_dir,
        experiment_name=name,
        verbose=True,
        **fn_kwargs,
    )
    print(f"[OK] {name} done in {(time.time()-t0)/60:.1f} min")
    return csv_path


def main() -> None:
    # ------------------------------------------------------------------
    # 1. NSGA-III: extend 3 -> 5 seeds (existing budget pop=5/gen=3)
    # ------------------------------------------------------------------
    def _nsga3(seed: int) -> pd.DataFrame:
        return run_nsga3(
            pde_name="burgers",
            pop_size=POP,
            n_gen=GEN,
            epochs=EPOCHS,
            seed=seed,
        )

    append_seeds_to_existing(
        "burgers_nsga3_clean",
        "multiseed_burgers_nsga3_clean",
        EXTEND_SEEDS,
        _nsga3,
        fn_kwargs={},
    )

    # ------------------------------------------------------------------
    # 2. MOEA/D: extend 3 -> 5 seeds (existing budget pop=5/gen=3)
    # ------------------------------------------------------------------
    def _moead(seed: int) -> pd.DataFrame:
        return run_moead(
            pde_name="burgers",
            pop_size=POP,
            n_gen=GEN,
            epochs=EPOCHS,
            seed=seed,
        )

    append_seeds_to_existing(
        "burgers_moead_clean",
        "multiseed_burgers_moead_clean",
        EXTEND_SEEDS,
        _moead,
        fn_kwargs={},
    )

    # ------------------------------------------------------------------
    # 3. Noise sweep sigma in {0.01, 0.10} at 5 seeds
    #    (sigma=0.05 already at 5 seeds in multiseed_burgers_grid_noisy)
    # ------------------------------------------------------------------
    for sigma in (0.01, 0.10):
        tag = f"burgers_grid_noisy_s{int(round(sigma * 100)):02d}"
        subdir = f"multiseed_{tag}"
        def _noisy(seed: int, _sigma: float = sigma) -> pd.DataFrame:
            return run_burgers_grid_noisy_single_seed(
                seed=seed, noise_sigma=_sigma, epochs=EPOCHS,
            )
        run_fresh(tag, subdir, _noisy, [0, 1, 2, 3, 4])

    # ------------------------------------------------------------------
    # 4. 2D Poisson grid: 5 seeds (currently zero seeds)
    # ------------------------------------------------------------------
    run_fresh(
        "poisson2d_grid_clean",
        "multiseed_poisson2d_grid",
        run_poisson2d_grid_single_seed,
        [0, 1, 2, 3, 4],
        epochs=EPOCHS,
    )

    print("\n" + "=" * 78)
    print("ALL MISSING 5-SEED RUNS COMPLETE")
    print("=" * 78)
    print("Now re-run: python assemble_master_outputs.py")


if __name__ == "__main__":
    main()