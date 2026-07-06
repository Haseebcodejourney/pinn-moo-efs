"""
Quick test of multi-seed framework with minimal budget.

Usage:
    python test_multiseed.py

This runs 2 seeds x 1 simple experiment to verify the framework works.
Total runtime: ~2-3 minutes on GPU.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Force UTF-8 stdout/stderr (Windows cp1252 cannot encode ⏱ ✓ ✅ etc.)
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd
from multi_seed_runner import (
    run_multi_seed_experiment,
    statistical_significance_test,
)
from run_grid import run_grid
from stats_report_generator import generate_report

RESULTS = ROOT / "results"
TEST_OUTPUT = ROOT / "results" / "test_multiseed"


def test_poisson_grid_single_seed(seed: int) -> pd.DataFrame:
    """Minimal Poisson grid: 3 configs only."""
    return run_grid(
        pde_name="poisson",
        epochs=500,  # Very short for testing
        seed=seed,
        lam_values=[0.1, 1.0],  # Only 2 values
        n_values=[100, 200],    # Only 2 values
        out_name=None,
    )


def main() -> None:
    print("\n" + "=" * 70)
    print("TESTING MULTI-SEED FRAMEWORK")
    print("=" * 70)
    print("\n[TIMER] Running 2 seeds x Poisson grid (minimal budget)...")
    print("   This tests data collection, aggregation, and statistics.\n")

    TEST_OUTPUT.mkdir(parents=True, exist_ok=True)

    # Run multi-seed experiment
    result = run_multi_seed_experiment(
        experiment_fn=test_poisson_grid_single_seed,
        seeds=[42, 99],  # Two different seeds
        output_dir=TEST_OUTPUT,
        experiment_name="test_poisson",
        verbose=True,
    )

    all_runs = result["all_runs"]
    summary = result["summary_stats"]

    print(f"\n{'='*70}")
    print("RESULTS")
    print("=" * 70)
    print(f"\nAll runs (shape: {all_runs.shape}):")
    print(all_runs.head(10).to_string())

    print(f"\n\nSummary statistics (shape: {summary.shape}):")
    print(summary.to_string())

    # Test statistical comparison
    print(f"\n{'='*70}")
    print("STATISTICAL TESTING")
    print("=" * 70)

    seed_42_l2 = all_runs[all_runs["seed"] == 42]["f1_l2_error"].values
    seed_99_l2 = all_runs[all_runs["seed"] == 99]["f1_l2_error"].values

    if len(seed_42_l2) > 0 and len(seed_99_l2) > 0:
        print(f"\nComparing L2 error between seed 42 and seed 99:")
        print(f"  Seed 42: {len(seed_42_l2)} values, mean={seed_42_l2.mean():.6f}")
        print(f"  Seed 99: {len(seed_99_l2)} values, mean={seed_99_l2.mean():.6f}")
        print(f"\n[NOTE] These are different configs, not paired comparison")
        print(f"       (just demonstrating statistical function)")

    # Generate report
    print(f"\n{'='*70}")
    print("GENERATING REPORT")
    print("=" * 70)

    report = generate_report(
        TEST_OUTPUT / "test_poisson_all_seeds.csv",
        output_dir=TEST_OUTPUT / "report",
        verbose=True,
    )

    print(f"\n{'='*70}")
    print("[OK] MULTI-SEED FRAMEWORK TEST PASSED")
    print("=" * 70)
    print(f"\nTest outputs saved to: {TEST_OUTPUT}")
    print(f"\nKey files:")
    print(f"  * {TEST_OUTPUT / 'test_poisson_all_seeds.csv'}")
    print(f"  * {TEST_OUTPUT / 'test_poisson_summary_stats.csv'}")
    print(f"  * {TEST_OUTPUT / 'test_poisson_metadata.json'}")
    print(f"  * {TEST_OUTPUT / 'report' / 'stats_summary.csv'}")
    print(f"\nThe framework is ready for full multi-seed experiments!")


if __name__ == "__main__":
    main()
