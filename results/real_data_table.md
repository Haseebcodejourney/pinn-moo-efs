# Real-Data Burgers Comparison Table

Multi-seed results for the PINN-MOO-EFS paper.

Observations are sampled from a high-fidelity Burgers reference
(`src/real_data_burgers.py` -- 1024 x 512 Radau, rtol=1e-7, atol=1e-10)
with sigma derived empirically from the fine-grid bilinear
interpolation residual at random sample locations.

This is the deliverable for Prof. Dr. Asiksoy's request:
> 'If you have time, adding an open dataset containing sparse real
>  measurements paired with a known PDE would also eliminate the
>  sigma = 0.05 was chosen arbitrarily criticism.'

Each baseline was run across multiple seeds; metrics report
**best (lowest) value per seed, then mean +/- std across seeds**.

## Per-method, per-n_obs results

| n_obs | Method | n_seeds | sigma (derived) | f1 L2 (mean +/- std) | f2 residual (mean +/- std) | HV (mean +/- std) |
|-------|--------|---------|------------------|----------------------|----------------------------|-------------------|
| 20 | Grid (real-data) | 3 | 0.0227 | 0.3504 +/- 0.0447 | 0.0210 +/- 0.0017 | 20.4673 +/- 2.4220 |
| 20 | ReLoBRaLo (real-data) | 3 | 0.0227 | 0.4833 +/- 0.0471 | 0.0531 +/- 0.0014 | 0.1483 +/- 0.0872 |
| 20 | GradNorm (real-data) | 3 | 0.0227 | 0.6950 +/- 0.1131 | 0.0006 +/- 0.0001 | 0.8427 +/- 0.6914 |

| 50 | Grid (real-data) | 3 | 0.0199 | 0.2875 +/- 0.0354 | 0.0246 +/- 0.0021 | 16.4796 +/- 2.2357 |
| 50 | ReLoBRaLo (real-data) | 3 | 0.0199 | 0.4480 +/- 0.0074 | 0.0563 +/- 0.0038 | 0.3758 +/- 0.3087 |
| 50 | GradNorm (real-data) | 3 | 0.0199 | 0.7639 +/- 0.1248 | 0.0014 +/- 0.0004 | 0.4225 +/- 0.4278 |

| 100 | Grid (real-data) | 3 | 0.0212 | 0.2815 +/- 0.0231 | 0.0219 +/- 0.0014 | 17.8641 +/- 4.1619 |
| 100 | ReLoBRaLo (real-data) | 3 | 0.0212 | 0.4250 +/- 0.0124 | 0.0518 +/- 0.0005 | 0.2996 +/- 0.3286 |
| 100 | GradNorm (real-data) | 3 | 0.0212 | 0.7586 +/- 0.1230 | 0.0025 +/- 0.0012 | 0.4015 +/- 0.4661 |

## Data efficiency: best f1 vs n_obs (lower is better)

| Method | n_obs=20 | n_obs=50 | n_obs=100 |
|--------|----------|----------|-----------|
| Grid (real-data) | 0.3504 +/- 0.0447 | 0.2875 +/- 0.0354 | 0.2815 +/- 0.0231 |
| GradNorm (real-data) | 0.6950 +/- 0.1131 | 0.7639 +/- 0.1248 | 0.7586 +/- 0.1230 |
| ReLoBRaLo (real-data) | 0.4833 +/- 0.0471 | 0.4480 +/- 0.0074 | 0.4250 +/- 0.0124 |

## Statistical significance (Wilcoxon signed-rank, paired by seed)

Reference baseline: Grid (real-data). Significance threshold: p < 0.05.

| n_obs | Candidate | p-value | Significant? | Candidate mean +/- std |
|-------|-----------|---------|---------------|------------------------|
| 20 | GradNorm (real-data) | 0.2500 | no | 0.6950 +/- 0.1131 |
| 20 | ReLoBRaLo (real-data) | 0.2500 | no | 0.4833 +/- 0.0471 |
| 50 | GradNorm (real-data) | 0.2500 | no | 0.7639 +/- 0.1248 |
| 50 | ReLoBRaLo (real-data) | 0.2500 | no | 0.4480 +/- 0.0074 |
| 100 | GradNorm (real-data) | 0.2500 | no | 0.7586 +/- 0.1230 |
| 100 | ReLoBRaLo (real-data) | 0.2500 | no | 0.4250 +/- 0.0124 |
