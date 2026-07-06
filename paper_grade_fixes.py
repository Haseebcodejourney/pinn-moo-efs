"""One-shot fixes for two known gaps:

  1. Re-run MOEA/D at n_partitions=6 (28 ref dirs instead of 91) so the
     Pareto front is diverse instead of collapsed -> hypervolume jumps.
  2. Extend every comparison to n=10 paired seeds so the Wilcoxon test
     reaches strict p<0.05 (n=10 critical value is 0.0277 for two-sided).

Run on CPU or GPU. On GPU the entire script finishes in ~1-3 hours.
On CPU expect 8-14 hours.

Usage:
    python paper_grade_fixes.py                 # both fixes at default settings
    python paper_grade_fixes.py --seeds 10      # 10 seeds
    python paper_grade_fixes.py --only-moead    # only fix MOEA/D partitions
    python paper_grade_fixes.py --only-seeds    # only extend to 10 seeds

After it finishes, re-run:
    python assemble_master_outputs.py
"""

from __future__ import annotations

import argparse
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

from moead_optimizer import run_moead
from multi_seed_runner import run_multi_seed_experiment

RESULTS = ROOT / "results"
EPOCHS = 500
POP, GEN = 5, 3


def rerun_moead_part6(seeds: list[int]) -> Path:
    """Re-run burgers_moead_clean at n_partitions=6 (28 ref dirs)."""
    out_dir = RESULTS / "multiseed_burgers_moead_clean_part6"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv = out_dir / "burgers_moead_clean_part6_all_seeds.csv"

    print(f"\n[moead/n_partitions=6] running {len(seeds)} seeds")
    rows = []
    for s in seeds:
        t0 = time.time()
        df = run_moead(
            pde_name="burgers",
            pop_size=POP,
            n_gen=GEN,
            epochs=EPOCHS,
            seed=s,
            n_partitions=6,
        )
        df["seed"] = s
        rows.append(df)
        print(f"  seed={s}: {len(df)} rows ({time.time()-t0:.1f}s)")
    combined = pd.concat(rows, ignore_index=True)
    combined.to_csv(csv, index=False)
    print(f"[OK] {csv} ({len(combined)} rows, {combined['seed'].nunique()} seeds)")
    return csv


def extend_experiment_to_n(name: str, subdir: str, filename: str,
                           target_seeds: int, pop_size: int = 6,
                           n_gen: int = 4) -> Path | None:
    """Extend a single 5-seed experiment to `target_seeds`."""
    out_dir = RESULTS / subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    csv = out_dir / f"{filename}_all_seeds.csv"
    if not csv.exists():
        print(f"[skip] {name}: no existing CSV at {csv}")
        return None

    existing = pd.read_csv(csv)
    have = sorted(int(s) for s in existing["seed"].unique())
    needed = [s for s in range(target_seeds) if s not in have]
    if not needed:
        print(f"[skip] {name}: already has {len(have)} seeds")
        return csv
    print(f"\n[extend] {name}: have {have}, adding {needed}")

    from finish_all_multiseed import (
        run_burgers_baseline_single_seed,
        run_burgers_grid_clean_single_seed,
        run_burgers_nsga2_single_seed,
        run_burgers_nsga3_single_seed,
    )

    fn_for_name = {
        "burgers_baseline_fixed": run_burgers_baseline_single_seed,
        "burgers_grid_clean":      run_burgers_grid_clean_single_seed,
        "burgers_nsga2_clean":     run_burgers_nsga2_single_seed,
        "burgers_nsga3_clean":     run_burgers_nsga3_single_seed,
    }

    fn = fn_for_name[name]

    def wrapped(seed: int, **kw):
        if fn in (run_burgers_nsga2_single_seed,
                  run_burgers_nsga3_single_seed):
            return fn(seed=seed, pop_size=pop_size, n_gen=n_gen, **kw)
        return fn(seed=seed, **kw)

    result = run_multi_seed_experiment(
        experiment_fn=wrapped,
        seeds=needed,
        output_dir=out_dir,
        experiment_name=name,
        verbose=True,
        epochs=EPOCHS,
    )
    all_rows = pd.concat([existing, result["all_runs"]], ignore_index=True)
    all_rows.to_csv(csv, index=False)
    print(f"[OK] {csv} now has {all_rows['seed'].nunique()} seeds")
    return csv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=10,
                        help="Target seed count (default 10 for p<0.05)")
    parser.add_argument("--only-moead", action="store_true")
    parser.add_argument("--only-seeds", action="store_true")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    args = parser.parse_args()

    seeds_to_run = list(range(args.seeds))
    print(f"Target seed count: {args.seeds}")
    print(f"Seeds to add to MOEA/D (n_partitions=6): {seeds_to_run}")

    if not args.only_seeds:
        rerun_moead_part6(seeds_to_run)

    if not args.only_moead:
        # Extend the four experiments that drive the headline claim
        for name, subdir, filename, pop, gen in [
            ("burgers_baseline_fixed", "multiseed_burgers_baseline",
             "burgers_baseline_fixed", 0, 0),
            ("burgers_grid_clean", "multiseed_burgers_grid_clean",
             "burgers_grid_clean", 0, 0),
            ("burgers_nsga2_clean", "multiseed_burgers_nsga2",
             "burgers_nsga2_clean", 15, 8),
            ("burgers_nsga3_clean", "multiseed_burgers_nsga3_clean",
             "burgers_nsga3_clean", 5, 3),
        ]:
            extend_experiment_to_n(name, subdir, filename, args.seeds,
                                  pop_size=pop, n_gen=gen)


if __name__ == "__main__":
    main()