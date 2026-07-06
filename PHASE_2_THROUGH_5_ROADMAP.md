# Roadmap: Phases 2–5 Implementation Plan

## Overview

After Phase 1 is validated with 5 seeds, implement these phases in order of priority (per Prof. Dr. Aşıksoy).

---

## **Phase 2: Scale MOO Budget** (Estimated: 1–2 weeks)

**Objective:** Increase NSGA-II from (6–10 pop, 4–6 gen) → (40–50 pop, 30+ gen)

### Why This Matters
- Current budget is too small; Pareto front hasn't converged
- Reviewer will reject as "results indistinguishable from noise"
- Need ~1200 PINN trainings per seed (not 60)

### Implementation Tasks

**Task 2.1: Modify NSGA-II Parameters**
```python
# File: src/nsga2_optimizer.py
# Change default values:
DEFAULT_POP_SIZE = 40  # was 10
DEFAULT_N_GEN = 30      # was 6
```

**Task 2.2: GPU Optimization**
- Reduce epochs per PINN from 2500 → 1500 (benchmark on validation error plateau)
- Use batch processing (already in `train.py`)
- Verify GPU memory usage

**Task 2.3: Validation**
```bash
# Test with one seed first
python -c "
from src.nsga2_optimizer import run_nsga2
import pandas as pd
result = run_nsga2('poisson', pop_size=40, n_gen=30, epochs=1500, seed=42)
print(result.shape)  # Should be ~1200 rows
"
```

**Task 2.4: Run Multi-Seed Experiments**
```bash
python finish_all_multiseed.py --seeds 5 --epochs 1500
```

### Expected Output
- `burgers_nsga2_n0_s0p0_summary_stats.csv` now with 40×30 = 1200+ evaluations
- Pareto front should be visibly smoother (less jagged)
- Hypervolume should increase (more diverse solutions)

### Success Criteria
✓ Front shape is smooth (no isolated outliers)  
✓ Hypervolume improves vs. Phase 1  
✓ Multiple non-dominated solutions clearly visible  

---

## **Phase 3: Add 2D Benchmarks** (Estimated: 2–3 weeks)

**Objective:** Eliminate "1D only" rejection ground by adding 2D problems

### Why This Matters
- Current: Poisson 1D (trivial), Burgers 1D (non-linear but 1D)
- Needed: At least one 2D problem
- Recommended: 2D Poisson (easier, good baseline) + Heat equation (time-dependent)

### Implementation Tasks

**Task 3.1: Implement 2D Poisson PDE**
File: `src/pde_poisson_2d.py`

```python
"""
2D Poisson equation: -∇²u = -2π² sin(πx) sin(πy)
Domain: [0,1] × [0,1]
BC: u = 0 (Dirichlet homogeneous)
Reference: u(x,y) = sin(πx)sin(πy)
"""

class PoissonPDE2D:
    def __init__(self):
        self.name = "poisson_2d"
        
    def ic_bc_data(self, n_ic, n_bc, rng):
        # Random (x, y) pairs for interior
        # Edge points for BC
        
    def residual(self, u_model, x, y):
        # Compute -u_xx - u_yy + source
        
    def exact_solution(self, x, y):
        # Return sin(πx)sin(πy)
```

**Task 3.2: Implement Heat Equation (Optional, Recommended)**
File: `src/pde_heat.py`

```python
"""
Heat equation: u_t - α∇²u = 0, α = 0.1
Domain: t∈[0,1], x∈[0,1], y∈[0,1]
IC: u(0,x,y) = sin(πx)sin(πy)
BC: u(t,0,y) = u(t,1,y) = u(t,x,0) = u(t,x,1) = 0
"""
```

**Task 3.3: Update PDE Registry**
```python
# File: src/pde_registry.py
PDE_CLASSES = {
    "poisson": PoissonPDE,
    "poisson_2d": PoissonPDE2D,      # NEW
    "burgers": BurgersPDE,
    "heat": HeatPDE,                  # NEW (optional)
}
```

**Task 3.4: Extend finish_all_multiseed.py**
```python
# Add new experiments:
# [7/8] Poisson 2D Grid (Multi-Seed)
# [8/8] Heat Equation Grid (Multi-Seed)

def run_poisson_2d_grid_single_seed(seed: int) -> pd.DataFrame:
    return run_grid(
        pde_name="poisson_2d",
        epochs=2000,
        seed=seed,
        lam_values=[0.1, 1.0, 10.0],
        n_values=[200, 400, 600],  # Larger collocation for 2D
    )
```

**Task 3.5: Run Multi-Seed Experiments on 2D Problems**
```bash
python finish_all_multiseed.py --seeds 5
# Now includes 2D results
```

### Expected Output
- `poisson_2d_grid_clean_summary_stats.csv` (new)
- `poisson_2d_grid_clean_all_seeds.csv` (all runs)
- Comparison: 1D vs 2D problem difficulty

### Success Criteria
✓ 2D problem runs without errors  
✓ PINN successfully trains on 2D domain  
✓ Accuracy within reasonable bounds (f1 < 0.1)  
✓ Multi-seed statistics computed successfully  

---

## **Phase 4: Implement Missing Baselines** (Estimated: 1–2 weeks)

**Objective:** Compare against GradNorm/ReLoBRaLo and MOEA/D algorithms

### Why This Matters
- Current only compares fixed weights vs. grid search vs. NSGA-II
- Missing: Adaptive loss balancing (GradNorm, ReLoBRaLo)
- Missing: Alternative MOO algorithms (MOEA/D, NSGA-III)
- Reviewer will say: "You avoided comparing with known rivals"

### Implementation Tasks

**Task 4.1: Enable GradNorm in Baseline**
Already implemented! Just add to experiments:

```bash
python run_experiment.py --mode baseline --pde burgers --balancer gradnorm
python run_experiment.py --mode baseline --pde burgers --balancer relobralo
```

**Task 4.2: Add Baseline Experiments to Pipeline**
```python
# In finish_all_multiseed.py, add:
def run_burgers_gradnorm_single_seed(seed: int):
    _, metrics = train_pinn(
        "burgers",
        balancer="gradnorm",
        epochs=2000,
        seed=seed,
    )
    return pd.DataFrame([{...}])

def run_burgers_relobralo_single_seed(seed: int):
    _, metrics = train_pinn(
        "burgers",
        balancer="relobralo",
        epochs=2000,
        seed=seed,
    )
    return pd.DataFrame([{...}])
```

**Task 4.3: Add MOEA/D Comparison**
```python
# File: src/moead_optimizer.py (new)
from pymoo.algorithms.moo.moead import MOEAD

def run_moead(pde_name, pop_size=40, n_gen=30, epochs=2500, seed=42):
    problem = PINNHyperparamProblem(...)
    algorithm = MOEAD(
        ref_dirs=...,
        sampling=...,
        crossover=...,
        mutation=...,
    )
    result = minimize(problem, algorithm, ...)
    return format_results(result)
```

**Task 4.4: Add NSGA-III Comparison**
```python
# Already exists in src/nsga2_optimizer.py, just enable:
def run_nsga3_comparison(pde_name, seed=42):
    # Run both NSGA-II and NSGA-III on same seed
    # Compare Pareto front quality
```

### Expected Output
```
Comparison CSV:
algorithm,pde,hypervolume_mean,hypervolume_std,p_value_vs_nsga2
baseline_fixed,burgers,0.234,0.012,—
gradnorm,burgers,0.445,0.015,0.0001 *
relobralo,burgers,0.418,0.018,0.0023 *
moead,burgers,0.456,0.014,0.0089 *
nsga2,burgers,0.412,0.016,—
nsga3,burgers,0.478,0.012,0.0034 *
```

### Success Criteria
✓ All baselines run without errors  
✓ Statistical comparisons computed  
✓ Clear ranking of algorithms visible  

---

## **Phase 5: Robustness & Polish** (Estimated: 1 week)

**Objective:** Robustness analysis, real data integration, figure/table fixes

### Why This Matters
- Prof. Dr. Aşıksoy mentioned: "σ = 0.05 chosen arbitrarily"
- Need robustness sweep: σ ∈ {0.01, 0.05, 0.1}
- Fix table/figure inconsistencies (Table 2 discrepancy mentioned)

### Implementation Tasks

**Task 5.1: Noise Robustness Sweep**
```bash
# Run experiments with multiple noise levels
python finish_all_multiseed.py --seeds 5 --noise-sweep
```

New sweep:
```python
NOISE_LEVELS = [0.0, 0.01, 0.05, 0.1]
for noise in NOISE_LEVELS:
    result = run_grid(..., obs_noise=noise, ...)
```

**Task 5.2: Real Data Integration** (Optional but recommended)
```python
# File: src/pde_real_data.py
"""
Option 1: FitzHugh-Nagumo with real neuroscience data
Option 2: Heat transfer from engineering dataset
Option 3: Fluid dynamics from OpenFOAM simulations
"""
```

**Task 5.3: Fix Figure/Table Issues**
- Table 2 inconsistency: 3.90×10⁻² vs 5.15×10⁻³ → clarify which metric
- Pareto scatter: Make "All" points visible (not just Pareto)
- Add reference point documentation for hypervolume
- Improve resolution/labels

**Task 5.4: Enhanced Report Generation**
```python
from stats_report_generator import generate_robust_report

report = generate_robust_report(
    'results/multiseed_all_runs.csv',
    noise_levels=[0.01, 0.05, 0.1],
    output_dir='final_report/',
)
```

### Expected Output
```
Robustness Analysis:
noise_sigma,f1_l2_error_mean,f1_l2_error_std,cv
0.0,0.00123,0.00008,6.5%
0.01,0.00156,0.00011,7.1%
0.05,0.00289,0.00019,6.6%
0.1,0.00512,0.00035,6.8%
# ✓ Consistent performance across noise levels
```

### Success Criteria
✓ Robustness metrics computed  
✓ All figures at publication quality  
✓ All tables are consistent  
✓ All claims backed by statistics  

---

## Timeline & Resource Estimation

| Phase | Tasks | Effort | GPU Time | Total Time |
|-------|-------|--------|----------|-----------|
| **1** | Multi-seed framework | ✅ Done | 0h | Complete |
| **2** | Scale MOO | 1–2 weeks | 16–32h | 1–2 weeks |
| **3** | 2D benchmarks | 2–3 weeks | 20–40h | 2–3 weeks |
| **4** | Baselines | 1–2 weeks | 10–20h | 1–2 weeks |
| **5** | Robustness | 1 week | 5–10h | 1 week |
| **Total** | | | 51–102h | 5–9 weeks |

---

## Success Metrics (Post-Phase 5)

After all phases, your paper should have:

✅ **Phase 1 Results:** 5–10 seeds, mean ± std, Wilcoxon tests  
✅ **Phase 2 Results:** 40–50 pop × 30 gen, smooth Pareto fronts  
✅ **Phase 3 Results:** 2D problems (Poisson 2D + Heat equation)  
✅ **Phase 4 Results:** GradNorm, ReLoBRaLo, MOEA/D comparisons  
✅ **Phase 5 Results:** Noise robustness, publication-quality figures  

**Paper Strength:** From "automatic rejection" → "strong candidate for publication"

---

## Decision Points After Each Phase

- **After Phase 1:** Validate statistics are reproducible, get Prof. Dr. Aşıksoy feedback
- **After Phase 2:** Confirm Pareto fronts are stable, hypervolumes reasonable
- **After Phase 3:** Decide whether to add Heat equation or focus on phase 4
- **After Phase 4:** Rank algorithms, identify best performing combination
- **After Phase 5:** Ready to submit to journal/conference

---

## Quick Reference: Commands for Each Phase

```bash
# Phase 1 (complete)
python test_multiseed.py
python finish_all_multiseed.py --seeds 5

# Phase 2 (after editing nsga2_optimizer.py)
python finish_all_multiseed.py --seeds 5 --epochs 1500

# Phase 3 (after adding pde_poisson_2d.py)
python finish_all_multiseed.py --seeds 5

# Phase 4 (after adding baseline methods)
python run_experiment.py --mode baseline --pde burgers --balancer gradnorm

# Phase 5 (after adding robustness)
python finish_all_multiseed.py --seeds 5 --noise-sweep
```

---

**Ready to move forward?** Start with Phase 1 validation, then proceed to Phase 2!
