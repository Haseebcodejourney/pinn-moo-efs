# Multi-Seed PINN Experiment Framework — Implementation Guide

## Overview

I have implemented **Phase 1: Multi-Seed Framework + Statistical Analysis** as requested by Prof. Dr. Aşıksoy. This addresses the **MOST CRITICAL GAP**: single seed experiments with no statistical rigor.

### What Has Been Implemented

#### 1. **Multi-Seed Runner Framework** (`src/multi_seed_runner.py`)
- `run_multi_seed_experiment()` — Orchestrates experiments across N seeds
- `statistical_significance_test()` — Wilcoxon signed-rank test (paired)
- `compare_methods_across_seeds()` — Cross-seed method comparisons
- Automatically computes: mean, std, min, max, confidence intervals for all metrics
- Aggregates results with seed tracking

#### 2. **Complete Multi-Seed Pipeline** (`finish_all_multiseed.py`)
Replaces `finish_all.py` with full multi-seed execution:

```bash
# Run with 5 seeds (default):
python finish_all_multiseed.py

# Run with 10 seeds:
python finish_all_multiseed.py --seeds 10

# Run with specific seeds:
python finish_all_multiseed.py --seeds "0,1,2,3,4"

# Quick test mode:
python finish_all_multiseed.py --seeds 2 --quick
```

**Six Experiments (each repeated N times):**
1. **Poisson Grid** (clean) → 27 configurations × N seeds
2. **Burgers Baseline** → 1 baseline × N seeds
3. **Burgers Grid** (clean) → 36 configurations × N seeds
4. **Burgers Grid** (sparse noisy: n_obs=20, σ=0.05) → 36 configurations × N seeds
5. **Burgers NSGA-II** → 6-10 Pareto candidates × N seeds
6. **Statistical analysis** — Reports Wilcoxon tests + significance

#### 3. **Statistical Report Generator** (`src/stats_report_generator.py`)
Produces comprehensive analysis:
- Summary statistics (mean ± std) grouped by algorithm/PDE
- Coefficient of variation (robustness across seeds)
- Wilcoxon signed-rank test results
- 95% confidence intervals
- CSV and JSON output

**Usage:**
```python
from stats_report_generator import generate_report
generate_report('multiseed_all_runs.csv', output_dir='reports/', verbose=True)
```

#### 4. **Quick Test Script** (`test_multiseed.py`)
Minimal end-to-end test (2 seeds, tiny budget):
```bash
python test_multiseed.py
# Runtime: ~2-3 minutes (GPU) or ~5-10 minutes (CPU)
```

Verifies:
- Multi-seed data collection works
- Statistics aggregation functions
- Report generation pipeline

---

## How to Use: Step-by-Step

### **Step 1: Setup (One Time)**

Ensure you have the research project with updated code:

```bash
cd "Data Science and Artificial Intelligence\research project"
pip install -r requirements.txt
```

### **Step 2: Run Quick Test** (Recommended First)

```bash
python test_multiseed.py
```

**Output:**
- `results/test_multiseed/test_poisson_all_seeds.csv` — All runs
- `results/test_multiseed/test_poisson_summary_stats.csv` — Aggregated stats
- `results/test_multiseed/report/stats_summary.csv` — Report
- `results/test_multiseed/test_poisson_metadata.json` — Metadata

This confirms everything works before running full experiments.

### **Step 3: Run Full Multi-Seed Experiments**

**Option A: 5 seeds (recommended starting point):**
```bash
python finish_all_multiseed.py --seeds 5
```

**Option B: 10 seeds (stronger statistical power):**
```bash
python finish_all_multiseed.py --seeds 10
```

**Estimated runtime:**
- 5 seeds: 4–8 hours (GPU), 16–24 hours (CPU)
- 10 seeds: 8–16 hours (GPU), 32–48 hours (CPU)

**Full control:**
```bash
python finish_all_multiseed.py --seeds "0,1,2,3,4,5,6,7,8,9" --epochs 2000
```

### **Step 4: Analyze Results**

Results are automatically saved to:

```
results/
├── multiseed_master_summary.csv          # Master summary (all experiments)
├── multiseed_all_runs.csv                # All individual runs with seeds
├── multiseed_significance_tests.csv      # Wilcoxon test results
├── multiseed_poisson_grid/
│   ├── poisson_grid_clean_all_seeds.csv
│   ├── poisson_grid_clean_summary_stats.csv
│   └── poisson_grid_clean_metadata.json
├── multiseed_burgers_baseline/
├── multiseed_burgers_grid_clean/
├── multiseed_burgers_grid_noisy/
└── multiseed_burgers_nsga2/
```

### **Step 5: Generate Custom Report**

```python
from stats_report_generator import generate_report, compare_two_methods
import pandas as pd

# Generate full report
report = generate_report(
    'results/multiseed_all_runs.csv',
    output_dir='results/my_report/',
    verbose=True
)

# Compare two specific methods
df = pd.read_csv('results/multiseed_all_runs.csv')
comparison = compare_two_methods(
    df,
    method1="baseline_fixed",
    method2="grid",
    metric="f1_l2_error",
    groupby_cols=["pde"]
)
print(comparison)
```

---

## Output Format & Interpretation

### **Results CSV Columns**

`multiseed_all_runs.csv`:
```
seed, pde, algorithm, n_obs, obs_noise, f1_l2_error, f2_pde_residual, f3_data_budget, ...
42,   burgers, baseline_fixed, 0, 0.0, 0.00123, 0.456, 200, ...
99,   burgers, baseline_fixed, 0, 0.0, 0.00127, 0.421, 200, ...
42,   burgers, grid, 0, 0.0, 0.00098, 0.234, 300, ...
99,   burgers, grid, 0, 0.0, 0.00102, 0.198, 300, ...
...
```

### **Summary Statistics CSV**

`multiseed_poisson_grid/poisson_grid_clean_summary_stats.csv`:
```
pde,algorithm,n_obs,obs_noise,n_seeds,seeds_list,
f1_l2_error_mean,f1_l2_error_std,f1_l2_error_min,f1_l2_error_max,f1_l2_error_ci95,
f2_pde_residual_mean,f2_pde_residual_std,...
poisson,grid,0,0.0,5,"0,1,2,3,4",
0.000234,0.000015,0.000210,0.000253,0.000013,
0.123,0.008,0.112,0.132,0.007,
...
```

**Key columns:**
- `*_mean` — Average value across seeds
- `*_std` — Standard deviation (variability)
- `*_ci95` — 95% confidence interval
- `n_seeds` — Number of successful runs

### **Significance Test Results**

`multiseed_significance_tests.csv`:
```
comparison,metric,baseline_mean,grid_mean,p_value,significant,mean_improvement
baseline_vs_grid,L2_error,0.00250,0.00185,0.0032,True,26%
```

**Interpretation:**
- `p_value < 0.05` → Result is **statistically significant**
- `mean_improvement > 0` → Grid search improves baseline
- `significant=True` → Not just noise, reproducible improvement

---

## Key Metrics Explained

### **f1_l2_error** (Accuracy, lower is better)
Relative L2 error vs. reference solution: $\text{f1} = \frac{\|\hat{u} - u_{\text{ref}}\|_2}{\|u_{\text{ref}}\|_2}$

- Good: < 0.01
- Acceptable: 0.01–0.1
- Poor: > 0.1

### **f2_pde_residual** (Physics Consistency, lower is better)
Mean absolute PDE residual at collocation points: $\text{f2} = \frac{1}{N}\sum_{i=1}^{N}|r_i|$

- Good: < 0.1
- Acceptable: 0.1–0.5
- Poor: > 0.5

### **f3_data_budget** (Data Efficiency, lower is better)
Total collocation + sparse observation points: $\text{f3} = n_{\text{coll}} + n_{\text{obs}}$

- Efficient: 100–200
- Standard: 200–500
- Inefficient: > 500

---

## Comparing with Prof. Dr. Aşıksoy's Requirements

| Requirement | ✓ Implemented | How |
|---|---|---|
| 5–10 independent seeds | ✓ | `--seeds N` parameter |
| Mean ± std for all metrics | ✓ | Auto-aggregated in `*_summary_stats.csv` |
| Statistical significance tests | ✓ | Wilcoxon test in `*_significance_tests.csv` |
| Seed tracking | ✓ | Seed column in all results |
| Robustness analysis | ✓ | Coefficient of variation computed |
| 95% CI | ✓ | `*_ci95` columns |

---

## Next Steps (Phase 2+)

After validating Phase 1 with 5 seeds:

### **Phase 2: Scale MOO Budget**
- Increase NSGA-II: 40–50 pop × 30+ gen
- Adjust training epochs
- See: `modify_nsga2_budget.md` (TODO)

### **Phase 3: Add 2D Benchmarks**
- Implement 2D Poisson (`pde_poisson_2d.py`)
- Implement heat equation (`pde_heat.py`)
- Repeat multi-seed experiments
- See: `add_2d_benchmarks.md` (TODO)

### **Phase 4: Baseline Comparisons**
- Enable GradNorm/ReLoBRaLo in grid search
- Add MOEA/D vs. NSGA-II
- Compare against published methods
- See: `implement_baselines.md` (TODO)

### **Phase 5: Robustness Analysis**
- Noise sweep: σ ∈ {0.01, 0.05, 0.1}
- Real data integration
- Figure/table polish

---

## Troubleshooting

### **Issue: "ModuleNotFoundError: No module named 'scipy'"**
```bash
pip install scipy
```

### **Issue: "CUDA out of memory"**
Set epochs lower or reduce batch sizes in `train.py`:
```bash
python finish_all_multiseed.py --seeds 2 --epochs 1000
```

### **Issue: Results not appearing**
Check `results/multiseed_all_seeds.csv` in subdirectories:
```bash
ls results/multiseed_*/
```

### **Issue: Test script hangs**
Press Ctrl+C and check GPU/memory with:
```bash
nvidia-smi  # (on NVIDIA GPU systems)
```

---

## File Inventory

| File | Purpose | Status |
|---|---|---|
| `src/multi_seed_runner.py` | Core multi-seed framework | ✅ Created |
| `src/stats_report_generator.py` | Statistical analysis utilities | ✅ Created |
| `finish_all_multiseed.py` | Main multi-seed pipeline | ✅ Created |
| `test_multiseed.py` | Quick validation script | ✅ Created |
| `README_MULTISEED.md` | This document | ✅ Created |

---

## Example: Running with 5 Seeds

```bash
# Terminal 1: Start experiments
cd research project
python finish_all_multiseed.py --seeds 5
# [Progress shown every experiment]
# Total time: ~4–8 hours (GPU)

# Terminal 2 (while waiting): Check intermediate results
tail -f results/multiseed_poisson_grid/poisson_grid_clean_all_seeds.csv

# After complete: View summary
cat results/multiseed_master_summary.csv

# Generate custom analysis
python
>>> from stats_report_generator import generate_report
>>> generate_report('results/multiseed_all_runs.csv', output_dir='analysis/')
```

---

## Contact & Support

For issues or questions about the multi-seed framework:
1. Check test_multiseed.py output first
2. Review individual seed logs in `multiseed_*/` directories
3. Verify requirements.txt is installed

---

**Phase 1 Complete!** ✅

You now have:
- ✅ Multi-seed execution framework
- ✅ Statistical analysis (mean, std, significance tests)
- ✅ Wilcoxon signed-rank testing
- ✅ Reproducible, defensible results

Next: Run with 5 seeds, then move to Phase 2 (2D benchmarks).
