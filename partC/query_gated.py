"""Trial of a query-gated statistics weight on Part C data (EuroSAT, Galaxy10).

This is OUR OWN implementation of the idea in member B's progress note ("when the top-2 candidate
scores of a query are close, rely more on the statistical calibration").  B's exact formula and
constants were not available, so the numbers here are not comparable to B's "Query-Gated HyCal".

Scores (cos, md are the per-query CDF-normalised score matrices of the official evaluator):
  Original      s = (1-w_c)*cos + w_c*md,  w_c = sigmoid((N_c-n_min_p)/scale) if N_c >= n_min_p else 0
                with n_min_p=5, scale=5  (official n_min_p=20 would switch md off for K<=10)
  Uniform xB    w_c -> min(1, B*w_c) for every query
  Query-gated   w_c -> min(1, B*w_c) only for queries whose cosine top-1/top-2 margin < tau,
                unchanged otherwise.  The gate uses the query's own scores only, never a label.
K=1 classes keep w_c = 0 (official fallback) in every variant.

Protocol (same as B's): choose (B, tau) on the VALIDATION split, freeze, then evaluate once on TEST.
  * tau grid = 25/50/75% quantiles of the validation cosine margins (pooled over datasets, K, seeds),
    frozen as absolute numbers
  * selection criterion = domain-mean validation accuracy over K in {5, 10} and 3 seeds
Outputs: partC/results/query_gated/{config.json, validation.md, test.md, incremental.md}
"""
import json
import os

import numpy as np
import torch
from sklearn.metrics import f1_score

import common as C
from hycal_cached import HyCalCached
from run_incremental import load_domain

SEEDS, KS, N_MIN_P, SCALE = [0, 1, 42], [5, 10], 5, 5.0
BETAS = [1.5, 2.0, 3.0]
OUT = os.path.join(C.RESULT_DIR, "query_gated")


def base_weights(model):
    n = torch.tensor([model.N[c] for c in model.class_ids], dtype=torch.float32)
    w = torch.sigmoid((n - N_MIN_P) / SCALE)
    return torch.where(n >= N_MIN_P, w, torch.zeros_like(w))


def margins(cos):
    top2 = cos.topk(2, dim=1).values
    return top2[:, 0] - top2[:, 1]


def predict(s, w, beta=1.0, tau=None):
    """tau=None -> uniform boost; beta=1 -> Original."""
    if tau is None:
        boost = torch.full((s["cos"].shape[0],), float(beta))
    else:
        boost = torch.where(margins(s["cos"]) < tau, torch.tensor(float(beta)), torch.tensor(1.0))
    wq = torch.clamp(w.unsqueeze(0) * boost.unsqueeze(1), max=1.0)
    return ((1 - wq) * s["cos"] + wq * s["md"]).argmax(dim=1).numpy()


def ms(v, signed=False):
    v = np.array(v)
    return (f"{v.mean():+.2f}" if signed else f"{v.mean():.2f}") + f" ± {v.std(ddof=1):.2f}"


def main():
    os.makedirs(OUT, exist_ok=True)
    # ------------------------------------------------ scores for val and test, every (dataset, K, seed)
    runs = {}
    for ds in C.DATASETS:
        d = np.load(C.feature_cache(ds))
        ids = [str(c) for c in d["class_ids"]]
        text = {cid: torch.from_numpy(d["text_feats"])[i] for i, cid in enumerate(ids)}
        train = torch.from_numpy(d["train_feats"])
        for k in KS:
            for seed in SEEDS:
                idx = json.load(open(C.index_file(ds, k, seed)))
                model = HyCalCached()
                model.add_session(train[idx["train_indices"]], idx["class_ids"], text)
                runs[(ds, k, seed)] = dict(
                    w=base_weights(model),
                    val=model.score(torch.from_numpy(d["val_feats"])), y_val=d["val_labels"],
                    test=model.score(torch.from_numpy(d["test_feats"])), y_test=d["test_labels"])

    # ------------------------------------------------ candidate configs (tau grid from VALIDATION margins)
    val_margins = torch.cat([margins(r["val"]["cos"]) for r in runs.values()])
    taus = [float(torch.quantile(val_margins, q)) for q in (0.25, 0.5, 0.75)]
    configs = {"Original": (1.0, None)}
    for b in BETAS:
        configs[f"Uniform x{b}"] = (b, None)
        for q, t in zip((25, 50, 75), taus):
            configs[f"Gated x{b}, margin<q{q}"] = (b, t)

    def acc(split, name, ds, k, seed):
        r = runs[(ds, k, seed)]
        b, t = configs[name]
        return 100.0 * (predict(r[split], r["w"], b, t) == r[f"y_{split}"]).mean()

    # ------------------------------------------------ validation selection
    lines = ["Validation accuracy (%), mean over 3 seeds. Selection = mean of the four columns.\n",
             "| config | " + " | ".join(f"{ds} K={k}" for ds in C.DATASETS for k in KS) + " | domain-mean |",
             "|---|" + "---|" * (len(C.DATASETS) * len(KS) + 1)]
    val_score = {}
    for name in configs:
        cells = [np.mean([acc("val", name, ds, k, s) for s in SEEDS]) for ds in C.DATASETS for k in KS]
        val_score[name] = float(np.mean(cells))
        lines.append(f"| {name} | " + " | ".join(f"{c:.2f}" for c in cells) + f" | {val_score[name]:.2f} |")
    best_gated = max((n for n in configs if n.startswith("Gated")), key=lambda n: val_score[n])
    best_uniform = max((n for n in configs if n.startswith("Uniform")), key=lambda n: val_score[n])
    lines.append(f"\nSelected on validation: **{best_gated}**; best uniform control: **{best_uniform}**.")
    lines.append(f"tau grid (absolute cosine-margin values): q25={taus[0]:.4f}, q50={taus[1]:.4f}, q75={taus[2]:.4f}")
    open(os.path.join(OUT, "validation.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    json.dump(dict(n_min_p=N_MIN_P, scale=SCALE, lambda_reg=1e-4, lambda_bg=1e-4, margin="cosine top1-top2 (CDF-normalised)",
                   tau_grid=taus, betas=BETAS, selected=best_gated, selected_beta=configs[best_gated][0],
                   selected_tau=configs[best_gated][1], best_uniform=best_uniform, seeds=SEEDS, shots=KS),
              open(os.path.join(OUT, "config.json"), "w"), indent=1)

    # ------------------------------------------------ test, frozen configs
    report = ["Original", "Uniform x1.5", best_uniform, best_gated]
    report = list(dict.fromkeys(report))
    lines = ["Test results with configs frozen from validation (3 seeds; Δ = paired difference to Original, pp).\n",
             "| dataset | K | config | accuracy | macro-F1 | Δ accuracy | seeds improved |", "|---|---|---|---|---|---|---|"]
    for ds in C.DATASETS:
        for k in KS:
            base = [acc("test", "Original", ds, k, s) for s in SEEDS]
            for name in report:
                a = [acc("test", name, ds, k, s) for s in SEEDS]
                f1 = []
                for s in SEEDS:
                    r = runs[(ds, k, s)]
                    f1.append(100.0 * f1_score(r["y_test"], predict(r["test"], r["w"], *configs[name]), average="macro"))
                diff = np.array(a) - np.array(base)
                lines.append(f"| {ds} | {k} | {name} | {ms(a)} | {ms(f1)} | "
                             + ("—" if name == "Original" else ms(diff, True)) + " | "
                             + ("—" if name == "Original" else f"{int((diff > 0).sum())}/3") + " |")
    open(os.path.join(OUT, "test.md"), "w").write("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))

    # ------------------------------------------------ incremental EuroSAT + Galaxy10 (20 candidates)
    doms = {n: load_domain(n) for n in C.DATASETS}
    lines = ["EuroSAT + Galaxy10, final session, 20 candidate classes, frozen configs (3 seeds).\n",
             "| K_E / K_G | config | EuroSAT acc | Galaxy10 acc | EuroSAT→galaxy class (%) | Galaxy10→eurosat class (%) |",
             "|---|---|---|---|---|---|"]
    for ke, kg in [(5, 5), (10, 10), (10, 1), (1, 10)]:
        res = {name: {"eurosat": [], "galaxy10": [], "leak_eurosat": [], "leak_galaxy10": []} for name in report}
        for seed in SEEDS:
            model = HyCalCached()
            for n, k in [("eurosat", ke), ("galaxy10", kg)]:
                idx = json.load(open(C.index_file(n, k, seed)))
                model.add_session(doms[n]["train"][idx["train_indices"]], idx["class_ids"], doms[n]["text"])
            w = base_weights(model)
            col_dom = np.array([c.split(":")[0] for c in model.class_ids])
            for n in C.DATASETS:
                s = model.score(doms[n]["test"])
                true = np.array([model.class_ids.index(f"{n}:{l}") for l in range(len(doms[n]["ids"]))])[doms[n]["y"]]
                for name in report:
                    pred = predict(s, w, *configs[name])
                    res[name][n].append(100.0 * (pred == true).mean())
                    res[name][f"leak_{n}"].append(100.0 * (col_dom[pred] != n).mean())
        for name in report:
            r = res[name]
            lines.append(f"| {ke} / {kg} | {name} | {ms(r['eurosat'])} | {ms(r['galaxy10'])} | "
                         f"{ms(r['leak_eurosat'])} | {ms(r['leak_galaxy10'])} |")
    open(os.path.join(OUT, "incremental.md"), "w").write("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))


if __name__ == "__main__":
    main()
