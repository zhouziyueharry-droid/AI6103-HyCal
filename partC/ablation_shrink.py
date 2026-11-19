"""Controlled ablation: change ONE thing at a time between the Baseline and the shrinkage version.

param_search.py picked its Tier-1 and Tier-2 settings independently, so they differ in lambda as
well as in alpha.  This script separates the three changes on the TEST split (fusion=sum, weighted):

  A  Baseline               lambda=1e-4  n_min_p=5 scale=5  alpha=0
  B  lower threshold only   lambda=1e-4  n_min_p=2 scale=1  alpha=0
  C  B + lambda 1e-3        lambda=1e-3  n_min_p=2 scale=1  alpha=0      (= Tier 1)
  D  B + shrinkage          lambda=1e-4  n_min_p=2 scale=1  alpha=0.75   (= Tier 2)
  E  C + shrinkage          lambda=1e-3  n_min_p=2 scale=1  alpha=0.75
plus an alpha sweep at B's other parameters.  D vs B and E vs C change alpha only.

These are post-hoc comparisons on the test split (exploratory); the configurations themselves
were chosen on validation in param_search.py.
Output: partC/results/param_search/ablation_shrink.md
"""
import os

import numpy as np
import torch

import common as C
from param_search import KS, SEEDS, build, combine, ms

CONFIGS = {
    "A Baseline": (("sum", 1e-4, 0.0), ("weighted", 5, 5.0)),
    "B threshold only": (("sum", 1e-4, 0.0), ("weighted", 2, 1.0)),
    "C B + λ=1e-3": (("sum", 1e-3, 0.0), ("weighted", 2, 1.0)),
    "D B + shrinkage": (("sum", 1e-4, 0.75), ("weighted", 2, 1.0)),
    "E C + shrinkage": (("sum", 1e-3, 0.75), ("weighted", 2, 1.0)),
}
ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]


def main():
    data = {}
    for ds in C.DATASETS:
        d = np.load(C.feature_cache(ds))
        ids = [str(c) for c in d["class_ids"]]
        t = torch.from_numpy(d["text_feats"])
        data[ds] = dict(train=torch.from_numpy(d["train_feats"]), text={c: t[i] for i, c in enumerate(ids)},
                        test=torch.from_numpy(d["test_feats"]), y=d["test_labels"])

    def run(cfg, rule, ds, k):
        out = []
        for seed in SEEDS:
            m, counts = build(data, ds, k, seed, cfg)
            out.append(100.0 * (combine(m.score(data[ds]["test"]), counts, rule).argmax(1).numpy() == data[ds]["y"]).mean())
        return np.array(out)

    cells = [(ds, k) for ds in C.DATASETS for k in KS if k > 1]   # K=1 is identical in every configuration
    res = {name: {c: run(cfg, rule, *c) for c in cells} for name, (cfg, rule) in CONFIGS.items()}

    lines = ["Test accuracy (%), mean ± std over seeds 0/1/42.  K=1 omitted (identical in all configurations).\n",
             "| dataset | K | " + " | ".join(CONFIGS) + " |", "|---|---|" + "---|" * len(CONFIGS)]
    for c in cells:
        lines.append(f"| {c[0]} | {c[1]} | " + " | ".join(ms(res[n][c]) for n in CONFIGS) + " |")
    lines.append("| **mean of the four rows** | | " + " | ".join(f"{np.mean([res[n][c].mean() for c in cells]):.2f}" for n in CONFIGS) + " |")

    lines += ["\nPaired differences (pp, mean ± std over seeds; in brackets: seeds improved).\n",
              "| dataset | K | threshold only (B−A) | λ only (C−B) | **α only at λ=1e-4 (D−B)** | **α only at λ=1e-3 (E−C)** |",
              "|---|---|---|---|---|---|"]
    for c in cells:
        row = []
        for a, b in [("B threshold only", "A Baseline"), ("C B + λ=1e-3", "B threshold only"),
                     ("D B + shrinkage", "B threshold only"), ("E C + shrinkage", "C B + λ=1e-3")]:
            d = res[a][c] - res[b][c]
            row.append(f"{ms(d, True)} ({int((d > 0).sum())}/3)")
        lines.append(f"| {c[0]} | {c[1]} | " + " | ".join(row) + " |")

    lines += ["\nα sweep with λ=1e-4, n_min_p=2, scale=1 fixed (test accuracy %, mean ± std).\n",
              "| dataset | K | " + " | ".join(f"α={a:g}" for a in ALPHAS) + " |", "|---|---|" + "---|" * len(ALPHAS)]
    for c in cells:
        lines.append(f"| {c[0]} | {c[1]} | " + " | ".join(ms(run(("sum", 1e-4, a), ("weighted", 2, 1.0), *c)) for a in ALPHAS) + " |")

    out = os.path.join(C.RESULT_DIR, "param_search", "ablation_shrink.md")
    open(out, "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
