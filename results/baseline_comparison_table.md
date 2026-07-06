# Baseline Comparison Table

Multi-seed results for the PINN-MOO-EFS paper.
Each baseline was run across multiple seeds; metrics report
**best (lowest) value per seed, then mean +/- std across seeds**.

## Burgers (clean, N_obs = 0)

| Baseline | n_seeds | f1 L2 (mean +/- std) | f2 residual (mean +/- std) | f3 budget | HV (mean +/- std) | spacing (mean +/- std) |
|----------|---------|----------------------|----------------------------|-----------|-------------------|------------------------|
| NSGA-III (Jain & Deb 2014) | 3 | 0.3047 +/- 0.0535 | 0.0153 +/- 0.0072 | 87 +/- 33 | 3.5235 +/- 2.1282 | 91.4221 +/- 56.2820 |
| MOEA/D (Zhang & Li 2007) | 3 | 0.3539 +/- 0.1054 | 0.0651 +/- 0.0285 | 72 +/- 38 | 0.0063 +/- 0.0054 | 6.2980 +/- 8.7235 |
| NSGA-II (Deb et al. 2002) | 3 | 0.3954 +/- 0.0522 | 0.0168 +/- 0.0103 | 94 +/- 37 | 6.2734 +/- 2.1748 | 20.5874 +/- 9.5654 |
| Grid search | 3 | 0.4014 +/- 0.0545 | 0.0225 +/- 0.0012 | 100 +/- 0 | 18.5203 +/- 5.0475 | 41.3907 +/- 4.1366 |
| Fixed-weight PINN | 3 | 0.4457 +/- 0.0530 | 0.3071 +/- 0.0281 | 200 +/- 0 | 0.0000 +/- 0.0000 | nan +/- nan |
| ReLoBRaLo (Bischof & Kraus 2021) | 3 | 0.4689 +/- 0.0133 | 0.0583 +/- 0.0034 | 100 +/- 0 | 0.2094 +/- 0.1122 | 0.0000 +/- 0.0000 |
| GradNorm (Chen et al. 2018) | 3 | 0.8167 +/- 0.0823 | 0.0013 +/- 0.0002 | 100 +/- 0 | 0.3362 +/- 0.3053 | 0.0001 +/- 0.0001 |

## Statistical significance (Wilcoxon signed-rank, paired by seed)

Reference baseline: Fixed-weight PINN (lambda = (10, 1)).
Significance threshold: p < 0.05.

| Candidate | f1 p-value | f1 sig? | f1 candidate mean | f2 p-value | f2 sig? | f2 candidate mean |
|-----------|-----------|---------|-------------------|-----------|---------|-------------------|
| Grid search | 0.5000 | no | 0.4014 +/- 0.0545 | 0.2500 | no | 0.0225 +/- 0.0012 |
| GradNorm (Chen et al. 2018) | 0.2500 | no | 0.8167 +/- 0.0823 | 0.2500 | no | 0.0013 +/- 0.0002 |
| ReLoBRaLo (Bischof & Kraus 2021) | 0.5000 | no | 0.4689 +/- 0.0133 | 0.2500 | no | 0.0583 +/- 0.0034 |
| NSGA-II (Deb et al. 2002) | 0.7500 | no | 0.3954 +/- 0.0522 | 0.2500 | no | 0.0168 +/- 0.0103 |
| NSGA-III (Jain & Deb 2014) | 0.2500 | no | 0.3047 +/- 0.0535 | 0.2500 | no | 0.0153 +/- 0.0072 |
| MOEA/D (Zhang & Li 2007) | 0.5000 | no | 0.3539 +/- 0.1054 | 0.2500 | no | 0.0651 +/- 0.0285 |
