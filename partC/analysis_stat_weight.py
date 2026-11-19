"""Checks on Part C data that relate to member B's proposal (stronger / query-gated statistics weight).

B's exact Query-Gated formula is not available here, so nothing below implements it.  These are
formula-free checks of its premises on EuroSAT and Galaxy10, from the saved single-domain scores:

  1. weight sweep   accuracy of (1-w)*cos + w*md for a fixed w applied to every class
  2. margin bins    split queries into quartiles of the cosine top-1/top-2 margin and compare
                    Cosine-only vs Mahalanobis-only inside each quartile
  3. incremental    EuroSAT+Galaxy10, HyCal weighted (n_min_p=5, scale=5) with its weight
                    multiplied by `boost` and clipped to 1  -- OUR reading of "Uniform x1.5"

Output: partC/results/analysis_stat_weight/summary.md
"""
import json
import os

import numpy as np
import torch

import common as C
from hycal_cached import HyCalCached
from run_incremental import load_domain

SEEDS = [0, 1, 42]
WS = [0.0, 0.25, 0.5, 0.75, 1.0]


def ms(v):
    v = np.array(v)
    return f"{v.mean():.2f} ± {v.std(ddof=1):.2f}"


def main():
    out = []
    # ---------------------------------------------------------------- 1 + 2
    for ds in C.DATASETS:
        for k in [5, 10]:
            sweep = {w: [] for w in WS}
            bins = {q: {"cos": [], "md": [], "margin": []} for q in range(4)}
            for seed in SEEDS:
                p = np.load(os.path.join(C.RESULT_DIR, "single_domain", ds, f"preds_K{k}_seed{seed}.npz"))
                cos, md, y = p["cos_scores"], p["md_scores"], p["labels"]
                for w in WS:
                    sweep[w].append(100.0 * (((1 - w) * cos + w * md).argmax(1) == y).mean())
                top2 = np.sort(cos, axis=1)[:, -2:]
                margin = top2[:, 1] - top2[:, 0]
                edges = np.quantile(margin, [0.25, 0.5, 0.75])
                q = np.digitize(margin, edges)
                for b in range(4):
                    m = q == b
                    bins[b]["cos"].append(100.0 * (cos[m].argmax(1) == y[m]).mean())
                    bins[b]["md"].append(100.0 * (md[m].argmax(1) == y[m]).mean())
                    bins[b]["margin"].append(margin[m].mean())
            out.append(f"### {ds}, K={k}\n")
            out.append("Fixed statistics weight w (accuracy %, 3 seeds):\n")
            out.append("| " + " | ".join(f"w={w}" for w in WS) + " |")
            out.append("|" + "---|" * len(WS))
            out.append("| " + " | ".join(ms(sweep[w]) for w in WS) + " |\n")
            out.append("Queries grouped by cosine top-1/top-2 margin (quartiles, Q1 = least certain):\n")
            out.append("| quartile | mean margin | Cosine-only acc | Mahalanobis-only acc | md − cos (pp) |")
            out.append("|---|---|---|---|---|")
            for b in range(4):
                d = np.array(bins[b]["md"]) - np.array(bins[b]["cos"])
                out.append(f"| Q{b + 1} | {np.mean(bins[b]['margin']):.3f} | {ms(bins[b]['cos'])} | "
                           f"{ms(bins[b]['md'])} | {d.mean():+.2f} ± {d.std(ddof=1):.2f} |")
            out.append("")

    # ---------------------------------------------------------------- 3
    doms = {n: load_domain(n) for n in ["eurosat", "galaxy10"]}
    out.append("### EuroSAT + Galaxy10 incremental (final session, 20 candidate classes)\n")
    out.append("HyCal weighted (n_min_p=5, scale=5); `boost` multiplies w_c and clips at 1.\n")
    out.append("| K_E / K_G | boost | EuroSAT acc | Galaxy10 acc | EuroSAT→galaxy class (%) | Galaxy10→eurosat class (%) |")
    out.append("|---|---|---|---|---|---|")
    for ke, kg in [(5, 5), (10, 10), (10, 1), (1, 10), (10, 5), (5, 10)]:
        for boost in [1.0, 1.5]:
            res = {"eurosat": [], "galaxy10": [], "leak_e": [], "leak_g": []}
            for seed in SEEDS:
                model = HyCalCached()
                for n, k in [("eurosat", ke), ("galaxy10", kg)]:
                    idx = json.load(open(C.index_file(n, k, seed)))
                    model.add_session(doms[n]["train"][idx["train_indices"]], idx["class_ids"], doms[n]["text"])
                counts = torch.tensor([model.N[c] for c in model.class_ids], dtype=torch.float32)
                w = torch.clamp(torch.sigmoid((counts - 5) / 5.0) * boost, max=1.0)
                w = torch.where(counts >= 5, w, torch.zeros_like(w))
                col_dom = np.array([c.split(":")[0] for c in model.class_ids])
                for n, key in [("eurosat", "leak_e"), ("galaxy10", "leak_g")]:
                    s = model.score(doms[n]["test"])
                    pred = ((1 - w) * s["cos"] + w * s["md"]).argmax(1).numpy()
                    true = np.array([model.class_ids.index(f"{n}:{l}") for l in range(len(doms[n]["ids"]))])[doms[n]["y"]]
                    res[n].append(100.0 * (pred == true).mean())
                    res[key].append(100.0 * (col_dom[pred] != n).mean())
            out.append(f"| {ke} / {kg} | ×{boost} | {ms(res['eurosat'])} | {ms(res['galaxy10'])} | "
                       f"{ms(res['leak_e'])} | {ms(res['leak_g'])} |")

    d = os.path.join(C.RESULT_DIR, "analysis_stat_weight")
    os.makedirs(d, exist_ok=True)
    open(os.path.join(d, "summary.md"), "w").write("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
