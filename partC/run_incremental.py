"""Cross-domain class-incremental evaluation on cached features.

Domains arrive one per session.  After every session the single global classifier is evaluated
on the test sets of ALL domains seen so far, with ALL learned classes as candidates (the domain
of a query is never given).  Test labels are used only for scoring.

Works for any domain that has  cache/{name}_ViT-B-16.npz  and  indices/{name}_K{K}_seed{s}.json
in the Part C format, so the three-domain merge is:
    python run_incremental.py --domains flowers organamnist eurosat --shots 5 5 5 --all_orders

Reported per (order, shots, method), mean +- std over seeds:
  final per-domain accuracy A(T,d), domain-mean accuracy, overall (pooled) accuracy, class-macro
  average incremental accuracy = mean over sessions of pooled accuracy on all seen classes
  forgetting F(d) = max_{u in [d..T]} A(u,d) - A(T,d)
  leak(d) = % of domain-d test images predicted as a class of another domain (final session)

Outputs under partC/results/incremental/{tag}/.
"""
import argparse
import csv
import itertools
import json
import os

import numpy as np
import torch

import common as C
from hycal_cached import HyCalCached
from run_single_domain import METHODS


def load_domain(name):
    d = np.load(C.feature_cache(name))
    ids = [str(c) for c in d["class_ids"]]
    text = torch.from_numpy(d["text_feats"])
    return dict(name=name, ids=ids, train=torch.from_numpy(d["train_feats"]), train_paths=d["train_paths"],
                test=torch.from_numpy(d["test_feats"]), y=d["test_labels"],
                text={cid: text[i] for i, cid in enumerate(ids)})


def run_one(domains, order, shots, seed, lambda_reg, lambda_bg):
    """Return {method: {'A': {(session, domain): acc}, 'pooled': [acc per session], ...}}."""
    model = HyCalCached(fusion="sum", lambda_reg=lambda_reg, lambda_bg=lambda_bg)
    out = {m: dict(A={}, pooled=[], macro=None, leak={}) for m in METHODS}
    for t, name in enumerate(order):
        dom = domains[name]
        idx = json.load(open(C.index_file(name, shots[name], seed)))
        assert [str(p) for p in dom["train_paths"][idx["train_indices"]]] == idx["paths"]
        model.add_session(dom["train"][idx["train_indices"]], idx["class_ids"], dom["text"])
        col_domain = np.array([c.split(":")[0] for c in model.class_ids])
        seen = order[:t + 1]
        scores = {d: model.score(domains[d]["test"]) for d in seen}
        for m, kw in METHODS.items():
            correct = total = 0
            per_class = []
            for d in seen:
                pred = model.combine(scores[d], **kw).argmax(dim=1).numpy()
                true_col = np.array([model.class_ids.index(f"{d}:{l}") for l in range(len(domains[d]["ids"]))])[domains[d]["y"]]
                hit = pred == true_col
                out[m]["A"][(t, d)] = 100.0 * hit.mean()
                correct += hit.sum()
                total += len(hit)
                if t == len(order) - 1:
                    out[m]["leak"][d] = 100.0 * (col_domain[pred] != d).mean()
                    per_class += [hit[domains[d]["y"] == l].mean() for l in range(len(domains[d]["ids"]))]
            out[m]["pooled"].append(100.0 * correct / total)
            if t == len(order) - 1:
                out[m]["macro"] = 100.0 * float(np.mean(per_class))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domains", nargs="+", default=["eurosat", "galaxy10"])
    ap.add_argument("--shots", type=int, nargs="+", default=None, help="K per domain, same order as --domains")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 42])
    ap.add_argument("--all_orders", action="store_true", help="evaluate every arrival order of the domains")
    ap.add_argument("--lambda_reg", type=float, default=1e-4)
    ap.add_argument("--lambda_bg", type=float, default=1e-4)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()
    shots_list = args.shots or [5] * len(args.domains)
    shots = dict(zip(args.domains, shots_list))
    tag = args.tag or "_".join(f"{d}-K{k}" for d, k in shots.items())
    out_dir = os.path.join(C.RESULT_DIR, "incremental", tag)
    os.makedirs(out_dir, exist_ok=True)

    domains = {n: load_domain(n) for n in args.domains}
    orders = list(itertools.permutations(args.domains)) if args.all_orders else [tuple(args.domains)]

    rows, sess_rows = [], []
    for order in orders:
        for seed in args.seeds:
            res = run_one(domains, list(order), shots, seed, args.lambda_reg, args.lambda_bg)
            T = len(order) - 1
            for m, r in res.items():
                row = dict(order=">".join(order), shots="/".join(str(shots[d]) for d in order), seed=seed, method=m,
                           overall=r["pooled"][-1], class_macro=r["macro"],
                           domain_mean=float(np.mean([r["A"][(T, d)] for d in order])),
                           avg_incremental=float(np.mean(r["pooled"])))
                for d in args.domains:
                    first = order.index(d)
                    row[f"final_{d}"] = r["A"][(T, d)]
                    row[f"forget_{d}"] = max(r["A"][(u, d)] for u in range(first, T + 1)) - r["A"][(T, d)]
                    row[f"leak_{d}"] = r["leak"][d]
                rows.append(row)
                for (t, d), a in r["A"].items():
                    sess_rows.append(dict(order=">".join(order), seed=seed, method=m, session=t + 1,
                                          arrived=order[t], eval_domain=d, acc=a))

    for name, data in [("runs.csv", rows), ("session_matrix.csv", sess_rows)]:
        with open(os.path.join(out_dir, name), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(data[0].keys()))
            w.writeheader()
            w.writerows(data)

    metric_cols = [c for c in rows[0] if c not in ("order", "shots", "seed", "method")]
    lines = ["| order | shots | method | " + " | ".join(metric_cols) + " |", "|" + "---|" * (3 + len(metric_cols))]
    summ = []
    for order in orders:
        o = ">".join(order)
        for m in METHODS:
            r = [x for x in rows if x["order"] == o and x["method"] == m]
            s = dict(order=o, shots=r[0]["shots"], method=m, n_seeds=len(r))
            cells = []
            for c in metric_cols:
                v = np.array([x[c] for x in r])
                s[c + "_mean"], s[c + "_std"] = v.mean(), v.std(ddof=1) if len(v) > 1 else 0.0
                cells.append(f"{s[c + '_mean']:.2f} ± {s[c + '_std']:.2f}")
            summ.append(s)
            lines.append(f"| {o} | {s['shots']} | {m} | " + " | ".join(cells) + " |")
    with open(os.path.join(out_dir, "summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summ[0].keys()))
        w.writeheader()
        w.writerows(summ)
    open(os.path.join(out_dir, "summary.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("saved to", out_dir)


if __name__ == "__main__":
    main()
