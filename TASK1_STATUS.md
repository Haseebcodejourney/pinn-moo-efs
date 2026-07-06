# Task 1 Status — Multi-Seed + Statistical Significance

**Date:** 2026-06-29
**Addresses:** Prof. Dr. Asiksoy's "MOST CRITICAL GAP" (single-seed, no statistics)

---

## What Was Already Built (Before Today)

| File | Status |
|---|---|
| `src/multi_seed_runner.py` | Working: `run_multi_seed_experiment`, `statistical_significance_test` (Wilcoxon), `compare_methods_across_seeds` |
| `src/stats_report_generator.py` | Working: `statistical_summary`, `compare_two_methods`, `robustness_analysis`, `generate_report` (mean, std, CI95, CV) |
| `finish_all_multiseed.py` | Working: 5-experiment pipeline (Poisson grid, Burgers baseline, Burgers grid clean/noisy, Burgers NSGA-II) + baseline-vs-grid Wilcoxon |
| `test_multiseed.py` | Working: 2-seed smoke test |

**But:** Never ran successfully on this machine. First run died at the
`print("\n⏱️ Running ...")` line with `UnicodeEncodeError: 'charmap' codec
can't encode characters in position 2-3` — Windows `cp1252` cannot encode
the ⏱ ✓ ✅ ⚡ ⚠️ ± ≥ • glyphs.

---

## What Was Fixed Today

### 1. UTF-8 stdout/stderr enforcement (root cause of every prior run failure)

Added at the top of every script that prints:

```python
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass
```

Plus replaced non-ASCII glyphs (⏱ ✓ ✅ ⚡ ⚠️ ± ≥ • ×) with ASCII
equivalents in print statements (kept UTF-8 in matplotlib config and
code comments — those don't print to stdout).

Files touched:
- `test_multiseed.py`
- `src/multi_seed_runner.py`
- `src/stats_report_generator.py`
- `finish_all_multiseed.py`

### 2. New focused script: `efs_multiseed.py`

This is the deliverable that directly addresses the professor's
specific complaint: *"the improvement from 1.00009 to 1.00002 comes from
a single run ... You must demonstrate the statistical significance of
this improvement (e.g., the Wilcoxon signed-rank test)."*

For each seed:
1. Train Burgers grid -> Pareto front
2. Run static-fuzzy selection (default params)
3. Run EFS evolution (pop=15, gen=15) on same Pareto
4. Compute `balanced_loss` for both selections

Across seeds:
- Mean +/- std for static and evolved `balanced_loss`
- Mean +/- std for **hypervolume** (with explicit nadir point)
- Mean +/- std for **spacing**
- Paired **Wilcoxon signed-rank test** (the test she named)
- Per-seed consistency: `n_seeds_efs_better / n_seeds`
- Per-method f1, f2, f3 of the chosen candidate

Outputs:
- `results/multiseed_burgers_efs/efs_vs_static_fuzzy_all_seeds.csv` — per-seed raw
- `results/multiseed_burgers_efs/efs_vs_static_fuzzy_summary_stats.csv` — mean/std
- `results/multiseed_burgers_efs/efs_vs_static_fuzzy_significance.csv` — Wilcoxon
- `results/multiseed_burgers_efs/efs_vs_static_fuzzy_metadata.json`
- Appends row to `results/multiseed_significance_tests.csv`

**Smoke-tested with 2 seeds** — works end-to-end. Wilcoxon p=0.5 is
expected with n=2 (Wilcoxon needs >=5 pairs for statistical power).

---

## Verification (Run Today)

```
$ python test_multiseed.py
[OK] MULTI-SEED FRAMEWORK TEST PASSED
   16 runs (8 configs x 2 seeds)
   summary stats computed, report generated

$ python efs_multiseed.py --seeds "42,99" --quick
EFS vs. STATIC-FUZZY: Multi-Seed Statistical Comparison
Across 2 seeds:
  static_fuzzy balanced_loss:  1.006444 +/- 0.000047
  evolved_fuzzy balanced_loss: 1.004805 +/- 0.000432
  EFS wins in: 2/2 seeds
PARETO-LEVEL METRICS:
  hypervolume: 2.47 +/- 2.47 (nadir f1=0.64, f2=0.17, f3=155.0)
  spacing:     21.63 +/- 21.63
Wilcoxon W=0.0, p=0.50 (n too small)
```

---

## How to Run the Real Experiment (5 or 10 Seeds)

```bash
# Smoke test (2 seeds, tiny budget)
python efs_multiseed.py --seeds "42,99" --quick

# Recommended (5 seeds, full budget)
python efs_multiseed.py --seeds 5

# Stronger (10 seeds)
python efs_multiseed.py --seeds 10

# Or with explicit list
python efs_multiseed.py --seeds "0,1,2,3,4,5,6,7,8,9"

# Combine with the broader pipeline
python finish_all_multiseed.py --seeds 5
```

**Estimated runtime (CPU torch 2.12):**
- 5 seeds: ~25-40 minutes (each seed = 12 PINN trainings + 15x15 EFS evolution)
- 10 seeds: ~50-80 minutes

For GPU torch, divide by ~3-5x.

---

## What Task 1 STILL Does NOT Cover (Honest Gaps)

These are still on the "Task 1" critical list but were NOT fixed today:

1. **`finish_all_multiseed.py` does not yet compute HV + spacing per Pareto.**
   It computes mean +/- std of candidate-level metrics (f1, f2, f3) per
   (algorithm, n_obs, obs_noise) group, but does not aggregate Pareto-
   level metrics (HV, spacing) across seeds.
   - Fix: add 5-line loop computing `pareto_nondominated(df).pipe(lambda x: hypervolume(...))` per seed, then `groupby("seed").mean()`.

2. **`finish_all_multiseed.py` does not yet compare EFS-vs-static-fuzzy.**
   Only `efs_multiseed.py` does. To unify into one pipeline, call
   `efs_multiseed.run_efs_comparison_single_seed` from
   `finish_all_multiseed.main()` as experiment [6/6].

3. **HV reference point currently uses 10%-beyond-worst per seed**, which
   is defensible but not standardised. If the professor wants a
   *fixed* reference point across all seeds, that's a 3-line change.

---

## Recommended Next Steps

In order:

1. **Run** `python efs_multiseed.py --seeds 10` and send the
   `efs_vs_static_fuzzy_significance.csv` to Prof. Asiksoy.
   This is the smallest demonstration of Task 1.

2. **Extend `finish_all_multiseed.py`** with HV + spacing per Pareto
   per seed (15 lines), then re-run with 5-10 seeds.

3. **Then move to Task 2** (2D problem) — she said that's #2 priority.

4. **Then Task 3** (GradNorm/ReLoBRaLo/MOEA-D baselines) — code for
   GradNorm and ReLoBRaLo already exists in `src/adaptive_balancers.py`;
   just need to wire them into the grid runner.