Validation accuracy (%), mean over 3 seeds.  `val mean` is the selection criterion.

| configuration | eurosat K=1 | eurosat K=5 | eurosat K=10 | galaxy10 K=1 | galaxy10 K=5 | galaxy10 K=10 | val mean |
|---|---|---|---|---|---|---|---|
| **Official (n_min_p=20)**<br>fusion=sum, lambda=0.0001, alpha=0, weighted(n_min_p=20, scale=5) | 52.67 | 70.78 | 75.03 | 25.23 | 37.52 | 43.42 | 50.78 |
| **Baseline (n_min_p=5)**<br>fusion=sum, lambda=0.0001, alpha=0, weighted(n_min_p=5, scale=5) | 52.67 | 73.28 | 81.54 | 25.23 | 39.48 | 42.97 | 52.53 |
| **Tier 1: best official parameters**<br>fusion=sum, lambda=0.001, alpha=0, weighted(n_min_p=2, scale=1) | 52.67 | 75.54 | 82.67 | 25.23 | 39.48 | 42.82 | 53.07 |
| **Tier 2: + pooled shrinkage**<br>fusion=sum, lambda=0.0001, alpha=0.75, weighted(n_min_p=2, scale=1) | 52.67 | 77.96 | 86.11 | 25.23 | 39.67 | 46.11 | 54.63 |

Best value of each parameter on validation (other parameters free; alpha=0 unless stated):

| parameter | value → best val mean |
|---|---|
| fusion | sum: 53.07, concat: 52.93, None: 52.90 |
| lambda_reg | 1e-05: 52.69, 0.0001: 52.71, 0.001: 53.07, 0.01: 52.90, 0.1: 51.59 |
| sim_metric | cosine: 50.78, md: 53.06, avg: 52.27, weighted: 53.07 |
| n_min_p (weighted) | 2: 53.07, 5: 52.64, 10: 51.51, 20: 50.78 |
| shrink_alpha (Tier 2) | 0: 53.07, 0.25: 54.56, 0.5: 54.62, 0.75: 54.63, 1: 54.49 |

Best official-parameter setting if chosen per dataset (not a unified setting):
- eurosat: fusion=sum, lambda=0.001, alpha=0, md
- galaxy10: fusion=concat, lambda=0.001, alpha=0, weighted(n_min_p=2, scale=5)
