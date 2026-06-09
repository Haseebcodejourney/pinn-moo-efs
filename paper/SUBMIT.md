# Project Complete — Submission Checklist

## What is done

| Item | Location |
|------|----------|
| Full codebase | `src/`, `run_experiment.py`, `finish_all.py` |
| Final results | `results/summary_table.csv` |
| Figures | `results/figures/` |
| Paper draft (8 pages) | `paper/draft.md` |
| 24 references | `paper/references.bib` |

## Key results (from `summary_table.csv`)

### Poisson
- Best L2 (knee): **1.01×10⁻⁴**
- Best residual (knee): **3.23×10⁻³**
- Pareto points: **5**

### Burgers (clean)
- Best L2 (grid): **0.275** (λ=1,1, N=300)
- Best residual (knee): **9.65×10⁻³**
- NSGA-II best L2: **0.233**

### Burgers (20 obs, σ=0.05)
- Best L2 (knee): **0.246**
- Ideal point L2: **0.270** at budget 120

### EFS
- Evolved fuzzy **improves** balanced loss vs static (`efs_best_params.json`)

## Your last 3 steps (30–60 min)

1. Open `paper/draft.md` → copy to Word → insert figures from `results/figures/`
2. Add your name, institution, date
3. Export PDF (8 pages) + submit

## Reproduce everything

```bash
python finish_all.py
```

Or step-by-step:

```bash
python run_experiment.py --mode grid --pde burgers --epochs 2000 --seed 42
python run_experiment.py --mode plot --pde burgers --epochs 1500
```

## Figures to insert

- `burgers_fig_pareto_f1_f2.png`
- `burgers_fig_pareto_f1_f3.png`
- `burgers_fig_solution.png`
- `burgers_fig_efs_comparison.png`
- `poisson_fig_pareto_f1_f2.png` (if generated)
- `fig_data_scarcity.png`
