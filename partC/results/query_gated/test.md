Test results with configs frozen from validation (3 seeds; Δ = paired difference to Original, pp).

| dataset | K | config | accuracy | macro-F1 | Δ accuracy | seeds improved |
|---|---|---|---|---|---|---|
| eurosat | 5 | Original | 74.09 ± 2.13 | 74.05 ± 2.23 | — | — |
| eurosat | 5 | Uniform x1.5 | 75.54 ± 1.84 | 75.37 ± 1.87 | +1.45 ± 0.32 | 3/3 |
| eurosat | 5 | Uniform x2.0 | 76.67 ± 1.32 | 76.32 ± 1.29 | +2.58 ± 0.82 | 3/3 |
| eurosat | 5 | Gated x1.5, margin<q75 | 75.47 ± 1.85 | 75.30 ± 1.87 | +1.37 ± 0.28 | 3/3 |
| eurosat | 10 | Original | 82.07 ± 0.99 | 81.83 ± 1.14 | — | — |
| eurosat | 10 | Uniform x1.5 | 83.47 ± 1.01 | 83.05 ± 1.14 | +1.41 ± 0.45 | 3/3 |
| eurosat | 10 | Uniform x2.0 | 83.47 ± 1.01 | 83.05 ± 1.14 | +1.41 ± 0.45 | 3/3 |
| eurosat | 10 | Gated x1.5, margin<q75 | 83.23 ± 1.15 | 82.85 ± 1.25 | +1.16 ± 0.28 | 3/3 |
| galaxy10 | 5 | Original | 38.61 ± 1.44 | 36.93 ± 1.27 | — | — |
| galaxy10 | 5 | Uniform x1.5 | 38.58 ± 1.35 | 36.89 ± 0.98 | -0.04 ± 0.54 | 1/3 |
| galaxy10 | 5 | Uniform x2.0 | 37.82 ± 2.25 | 36.30 ± 1.85 | -0.79 ± 1.25 | 1/3 |
| galaxy10 | 5 | Gated x1.5, margin<q75 | 38.56 ± 1.37 | 36.89 ± 1.02 | -0.06 ± 0.51 | 1/3 |
| galaxy10 | 10 | Original | 41.06 ± 0.61 | 39.35 ± 0.72 | — | — |
| galaxy10 | 10 | Uniform x1.5 | 39.74 ± 0.65 | 37.95 ± 0.62 | -1.32 ± 0.59 | 0/3 |
| galaxy10 | 10 | Uniform x2.0 | 39.74 ± 0.65 | 37.95 ± 0.62 | -1.32 ± 0.59 | 0/3 |
| galaxy10 | 10 | Gated x1.5, margin<q75 | 40.27 ± 0.76 | 38.49 ± 0.70 | -0.79 ± 0.48 | 0/3 |
