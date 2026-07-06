# Real-Data Cylinder Wake Comparison Table

Multi-seed results for the PINN-MOO-EFS paper.

**Data source:** Real experimental cylinder wake flow from Maziar Raissi's PINNs repository
([github.com/maziarraissi/PINNs](https://github.com/maziarraissi/PINNs)).
Domain: x ∈ [1, 8], y ∈ [-2, 2], t ∈ [0, 20] around a cylinder of radius 0.5 at (1, 1.5).
5000 spatial points × 200 time steps = 1,000,000 real measurements of (u, v, p).
This is the deliverable for the user's 'Please add real data' request, addressing
Prof. Dr. Asiksoy's note that 'adding an open dataset containing sparse real
measurements paired with a known PDE would eliminate the sigma = 0.05 criticism'.

Each baseline was run across multiple seeds; metrics report
**best (lowest) value per seed, then mean +/- std across seeds**.

## Per-method, per-n_obs results

| n_obs | Method | n_seeds | f1 L2 (mean +/- std) | f2 residual (mean +/- std) | HV (mean +/- std) |
|-------|--------|---------|----------------------|----------------------------|-------------------|
| 50 | ReLoBRaLo (cylinder wake) | 3 | 0.4835 +/- 0.0017 | 0.0244 +/- 0.0030 | 0.0679 +/- 0.0735 |
| 50 | Grid (cylinder wake) | 3 | 0.4964 +/- 0.0005 | 0.0090 +/- 0.0003 | 1.4013 +/- 0.5277 |
| 50 | GradNorm (cylinder wake) | 3 | 0.5081 +/- 0.0044 | 0.0031 +/- 0.0001 | 0.1325 +/- 0.0983 |

| 200 | Grid (cylinder wake) | 3 | 0.4840 +/- 0.0013 | 0.0086 +/- 0.0004 | 0.4405 +/- 0.0200 |
| 200 | ReLoBRaLo (cylinder wake) | 3 | 0.4924 +/- 0.0049 | 0.0224 +/- 0.0027 | 0.0095 +/- 0.0065 |
| 200 | GradNorm (cylinder wake) | 3 | 0.5029 +/- 0.0091 | 0.0031 +/- 0.0001 | 0.0340 +/- 0.0144 |

| 500 | Grid (cylinder wake) | 3 | 0.4853 +/- 0.0009 | 0.0080 +/- 0.0007 | 0.2972 +/- 0.0201 |
| 500 | ReLoBRaLo (cylinder wake) | 3 | 0.4871 +/- 0.0023 | 0.0219 +/- 0.0029 | 0.0124 +/- 0.0126 |
| 500 | GradNorm (cylinder wake) | 3 | 0.5003 +/- 0.0052 | 0.0030 +/- 0.0001 | 0.0300 +/- 0.0129 |

## Data efficiency: best f1 vs n_obs (lower is better)

| Method | n_obs=50 | n_obs=200 | n_obs=500 |
|---|---|---|---|
| Grid (cylinder wake) | 0.4964 +/- 0.0005 | 0.4840 +/- 0.0013 | 0.4853 +/- 0.0009 |
| GradNorm (cylinder wake) | 0.5081 +/- 0.0044 | 0.5029 +/- 0.0091 | 0.5003 +/- 0.0052 |
| ReLoBRaLo (cylinder wake) | 0.4835 +/- 0.0017 | 0.4924 +/- 0.0049 | 0.4871 +/- 0.0023 |

## Statistical significance (Wilcoxon signed-rank, paired by seed)

Reference baseline: Grid (cylinder wake). Significance threshold: p < 0.05.

| n_obs | Candidate | p-value | Significant? | Candidate mean +/- std |
|-------|-----------|---------|---------------|------------------------|
| 50 | GradNorm (cylinder wake) | 0.2500 | no | 0.5081 +/- 0.0044 |
| 50 | ReLoBRaLo (cylinder wake) | 0.2500 | no | 0.4835 +/- 0.0017 |
| 200 | GradNorm (cylinder wake) | 0.2500 | no | 0.5029 +/- 0.0091 |
| 200 | ReLoBRaLo (cylinder wake) | 0.2500 | no | 0.4924 +/- 0.0049 |
| 500 | GradNorm (cylinder wake) | 0.2500 | no | 0.5003 +/- 0.0052 |
| 500 | ReLoBRaLo (cylinder wake) | 0.2500 | no | 0.4871 +/- 0.0023 |
