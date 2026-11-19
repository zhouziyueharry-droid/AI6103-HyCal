Validation accuracy (%), mean over 3 seeds. Selection = mean of the four columns.

| config | eurosat K=5 | eurosat K=10 | galaxy10 K=5 | galaxy10 K=10 | domain-mean |
|---|---|---|---|---|---|
| Original | 73.28 | 81.54 | 39.48 | 42.97 | 59.32 |
| Uniform x1.5 | 74.38 | 82.75 | 39.72 | 41.07 | 59.48 |
| Gated x1.5, margin<q25 | 73.75 | 81.54 | 39.31 | 42.63 | 59.31 |
| Gated x1.5, margin<q50 | 74.17 | 82.01 | 39.53 | 42.26 | 59.49 |
| Gated x1.5, margin<q75 | 74.27 | 82.56 | 39.72 | 41.58 | 59.53 |
| Uniform x2.0 | 75.43 | 82.75 | 38.73 | 41.07 | 59.50 |
| Gated x2.0, margin<q25 | 73.80 | 81.54 | 39.06 | 42.63 | 59.26 |
| Gated x2.0, margin<q50 | 74.55 | 82.01 | 39.25 | 42.26 | 59.52 |
| Gated x2.0, margin<q75 | 75.16 | 82.56 | 38.76 | 41.58 | 59.52 |
| Uniform x3.0 | 75.43 | 82.75 | 38.73 | 41.07 | 59.50 |
| Gated x3.0, margin<q25 | 73.80 | 81.54 | 39.06 | 42.63 | 59.26 |
| Gated x3.0, margin<q50 | 74.55 | 82.01 | 39.25 | 42.26 | 59.52 |
| Gated x3.0, margin<q75 | 75.16 | 82.56 | 38.76 | 41.58 | 59.52 |

Selected on validation: **Gated x1.5, margin<q75**; best uniform control: **Uniform x2.0**.
tau grid (absolute cosine-margin values): q25=0.0337, q50=0.0748, q75=0.1282
