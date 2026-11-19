"""Parameter search for HyCal on Part C data, selection on VALIDATION, one evaluation on TEST.

Tier 1 -- parameters that already exist as official command-line arguments (method untouched):
    --fusion      sum | concat | None
    --lambda_reg  1e-5 ... 1e-1
    --sim_metric  cosine | md | avg | weighted,  with --n_min_p and --scale
  (--lambda_bg and the background statistics are not searched: the background Mahalanobis term is
   constant across classes for a query and is removed by the per-query z-score, so it never changes
   a prediction.)
Tier 2 -- one small change to the statistics (NOT official HyCal):
    shrink_alpha  class covariance <- (1-a)*cov_c + a*cov_pooled(session), a in {0.25, 0.5, 0.75, 1}

Protocol: one configuration for both datasets and all K.  Selection criterion = mean validation
accuracy over {EuroSAT, Galaxy10} x K in {1, 5, 10} x seeds {0, 1, 42}.  Test is evaluated once,
only for the reference configurations and the selected ones.

Outputs: partC/results/param_search/{validation.md, test.md, incremental.md, selected.json, val_grid.csv}
"""
import csv
import itertools
import json
import os
import time

import numpy as np
import torch

import common as C
from hycal_cached import HyCalCached
from run_incremental import load_domain

SEEDS, KS = [0, 1, 42], [1, 5, 10]
FUSIONS = ["sum", "concat", None]
LAMBDAS = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]
RULES = ([("cosine", 0, 0.0), ("md", 0, 0.0)]
         + [("weighted", n, s) for n in (2, 5, 10, 20) for s in (1.0, 2.0, 5.0)]
         + [("avg", n, 0.0) for n in (2, 5)])
OUT = os.path.join(C.RESULT_DIR, "param_search")


def model_cfgs():
    for f in FUSIONS:
        for lam in LAMBDAS:
            for a in (ALPHAS if f != "concat" else [0.0]):   # concat is 1024-d: alpha not searched
                yield (f, lam, a)


def combine(s, counts, rule):
    metric, n_min_p, scale = rule
    if metric == "cosine":
        return s["cos"]
    if metric == "md":
        return s["md"]
    w = torch.sigmoid((counts - n_min_p) / scale) if metric == "weighted" else torch.full_like(counts, 0.5)
    w = torch.where(counts >= n_min_p, w, torch.zeros_like(w))
    return (1 - w) * s["cos"] + w * s["md"]


def build(data, ds, k, seed, cfg):
    f, lam, a = cfg
    d = data[ds]
    idx = json.load(open(C.index_file(ds, k, seed)))
    m = HyCalCached(fusion=f, lambda_reg=lam, shrink_alpha=a)
    m.add_session(d["train"][idx["train_indices"]], idx["class_ids"], d["text"])
    return m, torch.tensor([m.N[c] for c in m.class_ids], dtype=torch.float32)


def name(cfg, rule):
    f, lam, a = cfg
    r = rule[0] if rule[0] in ("cosine", "md") else f"{rule[0]}(n_min_p={rule[1]}" + (f", scale={rule[2]:g})" if rule[0] == "weighted" else ")")
    return f"fusion={f}, lambda={lam:g}, alpha={a:g}, {r}"


def ms(v, signed=False):
    v = np.array(v)
    return (f"{v.mean():+.2f}" if signed else f"{v.mean():.2f}") + f" ± {v.std(ddof=1):.2f}"


def main():
    os.makedirs(OUT, exist_ok=True)
    data = {}
    for ds in C.DATASETS:
        d = np.load(C.feature_cache(ds))
        ids = [str(c) for c in d["class_ids"]]
        t = torch.from_numpy(d["text_feats"])
        data[ds] = dict(train=torch.from_numpy(d["train_feats"]), text={c: t[i] for i, c in enumerate(ids)},
                        val=torch.from_numpy(d["val_feats"]), y_val=d["val_labels"],
                        test=torch.from_numpy(d["test_feats"]), y_test=d["test_labels"])

    # ------------------------------------------------ pass 1: validation grid
    cfgs = list(model_cfgs())
    val = {}   # (cfg, rule) -> {(ds, k): [acc per seed]}
    t0 = time.time()
    for ci, cfg in enumerate(cfgs):
        for ds in C.DATASETS:
            for k in KS:
                for seed in SEEDS:
                    m, counts = build(data, ds, k, seed, cfg)
                    s = m.score(data[ds]["val"])
                    for rule in RULES:
                        acc = 100.0 * (combine(s, counts, rule).argmax(1).numpy() == data[ds]["y_val"]).mean()
                        val.setdefault((cfg, rule), {}).setdefault((ds, k), []).append(acc)
        print(f"[val] {ci + 1}/{len(cfgs)} {cfg}  {time.time() - t0:.0f}s", flush=True)

    score = {key: float(np.mean([np.mean(v) for v in cells.values()])) for key, cells in val.items()}
    with open(os.path.join(OUT, "val_grid.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["fusion", "lambda_reg", "shrink_alpha", "metric", "n_min_p", "scale", "val_mean"]
                   + [f"{ds}_K{k}" for ds in C.DATASETS for k in KS])
        for (cfg, rule), cells in sorted(val.items(), key=lambda kv: -score[kv[0]]):
            w.writerow([cfg[0], cfg[1], cfg[2], rule[0], rule[1], rule[2], f"{score[(cfg, rule)]:.3f}"]
                       + [f"{np.mean(cells[(ds, k)]):.2f}" for ds in C.DATASETS for k in KS])

    official = (("sum", 1e-4, 0.0), ("weighted", 20, 5.0))
    baseline = (("sum", 1e-4, 0.0), ("weighted", 5, 5.0))
    tier1 = max((k for k in val if k[0][2] == 0.0), key=lambda k: score[k])
    tier2 = max(val, key=lambda k: score[k])
    per_ds = {ds: max((k for k in val if k[0][2] == 0.0),
                      key=lambda k: np.mean([np.mean(val[k][(ds, kk)]) for kk in KS])) for ds in C.DATASETS}
    chosen = {"Official (n_min_p=20)": official, "Baseline (n_min_p=5)": baseline,
              "Tier 1: best official parameters": tier1, "Tier 2: + pooled shrinkage": tier2}

    lines = ["Validation accuracy (%), mean over 3 seeds.  `val mean` is the selection criterion.\n",
             "| configuration | " + " | ".join(f"{ds} K={k}" for ds in C.DATASETS for k in KS) + " | val mean |",
             "|---|" + "---|" * 7]
    for label, key in chosen.items():
        lines.append(f"| **{label}**<br>{name(*key)} | " + " | ".join(f"{np.mean(val[key][(ds, k)]):.2f}" for ds in C.DATASETS for k in KS)
                     + f" | {score[key]:.2f} |")
    lines.append("\nBest value of each parameter on validation (other parameters free; alpha=0 unless stated):\n")
    lines.append("| parameter | value → best val mean |")
    lines.append("|---|---|")
    t1 = [k for k in val if k[0][2] == 0.0]
    lines.append("| fusion | " + ", ".join(f"{f}: {max(score[k] for k in t1 if k[0][0] == f):.2f}" for f in FUSIONS) + " |")
    lines.append("| lambda_reg | " + ", ".join(f"{l:g}: {max(score[k] for k in t1 if k[0][1] == l):.2f}" for l in LAMBDAS) + " |")
    lines.append("| sim_metric | " + ", ".join(f"{m}: {max(score[k] for k in t1 if k[1][0] == m):.2f}" for m in ("cosine", "md", "avg", "weighted")) + " |")
    lines.append("| n_min_p (weighted) | " + ", ".join(f"{n}: {max(score[k] for k in t1 if k[1][0] == 'weighted' and k[1][1] == n):.2f}" for n in (2, 5, 10, 20)) + " |")
    lines.append("| shrink_alpha (Tier 2) | " + ", ".join(f"{a:g}: {max(score[k] for k in val if k[0][2] == a):.2f}" for a in ALPHAS) + " |")
    lines.append("\nBest official-parameter setting if chosen per dataset (not a unified setting):")
    for ds, key in per_ds.items():
        lines.append(f"- {ds}: {name(*key)}")
    open(os.path.join(OUT, "validation.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)
    json.dump({label: dict(fusion=k[0][0], lambda_reg=k[0][1], shrink_alpha=k[0][2], metric=k[1][0],
                           n_min_p=k[1][1], scale=k[1][2], val_mean=score[k]) for label, k in chosen.items()},
              open(os.path.join(OUT, "selected.json"), "w"), indent=1)

    # ------------------------------------------------ pass 2: test, frozen
    test = {label: {} for label in chosen}
    for label, (cfg, rule) in chosen.items():
        for ds in C.DATASETS:
            for k in KS:
                for seed in SEEDS:
                    m, counts = build(data, ds, k, seed, cfg)
                    pred = combine(m.score(data[ds]["test"]), counts, rule).argmax(1).numpy()
                    test[label].setdefault((ds, k), []).append(100.0 * (pred == data[ds]["y_test"]).mean())
    labels = list(chosen)
    lines = ["Test accuracy (%), 3 seeds, configurations frozen from validation.  Δ = paired difference to "
             "Baseline (n_min_p=5), pp; in brackets: seeds improved.\n",
             "| dataset | K | " + " | ".join(labels) + " | Δ Tier 1 | Δ Tier 2 |", "|---|---|" + "---|" * (len(labels) + 2)]
    for ds in C.DATASETS:
        for k in KS:
            base = np.array(test[labels[1]][(ds, k)])
            cells = [ms(test[l][(ds, k)]) for l in labels]
            deltas = []
            for l in labels[2:]:
                d = np.array(test[l][(ds, k)]) - base
                deltas.append(f"{ms(d, True)} ({int((d > 0).sum())}/3)")
            lines.append(f"| {ds} | {k} | " + " | ".join(cells) + " | " + " | ".join(deltas) + " |")
    lines.append("| **mean** | all | " + " | ".join(f"{np.mean([np.mean(v) for v in test[l].values()]):.2f}" for l in labels) + " | | |")
    open(os.path.join(OUT, "test.md"), "w").write("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines), flush=True)

    # ------------------------------------------------ cross-domain check (EuroSAT + Galaxy10, 20 candidates)
    doms = {n: load_domain(n) for n in C.DATASETS}
    lines = ["EuroSAT + Galaxy10, final session, 20 candidate classes (3 seeds).\n",
             "| K_E / K_G | configuration | EuroSAT acc | Galaxy10 acc | EuroSAT→galaxy class (%) | Galaxy10→eurosat class (%) |",
             "|---|---|---|---|---|---|"]
    for ke, kg in [(5, 5), (10, 10), (10, 1), (1, 10), (10, 5), (5, 10)]:
        for label, (cfg, rule) in chosen.items():
            res = {"eurosat": [], "galaxy10": [], "leak_eurosat": [], "leak_galaxy10": []}
            for seed in SEEDS:
                m = HyCalCached(fusion=cfg[0], lambda_reg=cfg[1], shrink_alpha=cfg[2])
                for n, k in [("eurosat", ke), ("galaxy10", kg)]:
                    idx = json.load(open(C.index_file(n, k, seed)))
                    m.add_session(doms[n]["train"][idx["train_indices"]], idx["class_ids"], doms[n]["text"])
                counts = torch.tensor([m.N[c] for c in m.class_ids], dtype=torch.float32)
                col = np.array([c.split(":")[0] for c in m.class_ids])
                for n in C.DATASETS:
                    pred = combine(m.score(doms[n]["test"]), counts, rule).argmax(1).numpy()
                    true = np.array([m.class_ids.index(f"{n}:{l}") for l in range(len(doms[n]["ids"]))])[doms[n]["y"]]
                    res[n].append(100.0 * (pred == true).mean())
                    res[f"leak_{n}"].append(100.0 * (col[pred] != n).mean())
            lines.append(f"| {ke} / {kg} | {label} | {ms(res['eurosat'])} | {ms(res['galaxy10'])} | "
                         f"{ms(res['leak_eurosat'])} | {ms(res['leak_galaxy10'])} |")
    open(os.path.join(OUT, "incremental.md"), "w").write("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
