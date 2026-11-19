"""Single-domain experiment on cached features (EuroSAT RGB or Galaxy10 DECaLS).

For every (K, seed) it builds HyCal statistics from the frozen Support indices and scores the
full test split of that dataset (all its classes as candidates).  Test labels are used only to
compute metrics.

Methods
  zero_shot       CLIP text prompts only (no Support)
  cosine          HyCal pipeline with sim_metric=cosine   (Cosine-only baseline)
  md              HyCal pipeline with sim_metric=md       (relative Mahalanobis only)
  hycal_official  sim_metric=weighted, n_min_p=20, scale=5 (official scripts/run_balanced.sh)
  hycal_nmin5     sim_metric=weighted, n_min_p=5,  scale=5 (argparse default n_min_p)

Outputs under partC/results/single_domain/{dataset}/.
Usage: python run_single_domain.py --dataset eurosat|galaxy10
"""
import argparse
import csv
import json
import os
import time

import numpy as np
import torch

import common as C
from hycal_cached import HyCalCached, zero_shot_scores

METHODS = {
    "cosine": dict(metric="cosine"),
    "md": dict(metric="md"),
    "hycal_official": dict(metric="weighted", n_min_p=20, scale=5.0),
    "hycal_nmin5": dict(metric="weighted", n_min_p=5, scale=5.0),
}


def metrics(pred, y, n_cls):
    cm = np.zeros((n_cls, n_cls), dtype=np.int64)
    np.add.at(cm, (y, pred), 1)
    per_class = cm.diagonal() / cm.sum(axis=1)
    return 100.0 * (pred == y).mean(), 100.0 * per_class.mean(), 100.0 * per_class, cm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="eurosat", choices=C.DATASETS)
    ap.add_argument("--shots", type=int, nargs="+", default=[1, 5, 10])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 42])
    ap.add_argument("--lambda_reg", type=float, default=1e-4)
    ap.add_argument("--lambda_bg", type=float, default=1e-4)
    args = ap.parse_args()

    out_dir = os.path.join(C.RESULT_DIR, "single_domain", args.dataset)
    os.makedirs(out_dir, exist_ok=True)
    d = np.load(C.feature_cache(args.dataset))
    class_ids = [str(c) for c in d["class_ids"]]
    names = [str(n) for n in d["display_names"]]
    n_cls = len(class_ids)
    train = torch.from_numpy(d["train_feats"])
    test = torch.from_numpy(d["test_feats"])
    y = d["test_labels"]
    text = torch.from_numpy(d["text_feats"])
    text_map = {cid: text[i] for i, cid in enumerate(class_ids)}

    rows, cms = [], {}
    zs_pred = zero_shot_scores(test, text).argmax(dim=1).numpy()
    acc, macro, pc, cm = metrics(zs_pred, y, n_cls)
    for k in args.shots:
        for seed in args.seeds:   # zero-shot does not depend on K/seed; repeated so tables align
            rows.append(dict(K=k, seed=seed, method="zero_shot", acc=acc, macro_acc=macro,
                             calib_s=0.0, infer_s=0.0, **{f"acc_{n}": v for n, v in zip(names, pc)}))
        cms[(k, "zero_shot")] = cm * len(args.seeds)

    for k in args.shots:
        for seed in args.seeds:
            idx = json.load(open(C.index_file(args.dataset, k, seed)))
            assert [str(p) for p in d["train_paths"][idx["train_indices"]]] == idx["paths"]
            t0 = time.time()
            model = HyCalCached(fusion="sum", lambda_reg=args.lambda_reg, lambda_bg=args.lambda_bg)
            model.add_session(train[idx["train_indices"]], idx["class_ids"], text_map)
            t1 = time.time()
            s = model.score(test)
            t2 = time.time()
            assert model.class_ids == class_ids
            preds = {}
            for m, kw in METHODS.items():
                pred = model.combine(s, **kw).argmax(dim=1).numpy()
                preds[m] = pred
                acc, macro, pc, cm = metrics(pred, y, n_cls)
                rows.append(dict(K=k, seed=seed, method=m, acc=acc, macro_acc=macro,
                                 calib_s=t1 - t0, infer_s=t2 - t1,
                                 **{f"acc_{n}": v for n, v in zip(names, pc)}))
                cms[(k, m)] = cms.get((k, m), 0) + cm
            np.savez(os.path.join(out_dir, f"preds_K{k}_seed{seed}.npz"), labels=y,
                     test_paths=d["test_paths"], zero_shot=zs_pred,
                     cos_scores=s["cos"].numpy(), md_scores=s["md"].numpy(), **preds)

    with open(os.path.join(out_dir, "runs.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # summary: mean +- std over seeds
    lines = ["| K | method | overall acc (%) | class-macro acc (%) |", "|---|---|---|---|"]
    summ = []
    for k in args.shots:
        for m in ["zero_shot"] + list(METHODS):
            r = [x for x in rows if x["K"] == k and x["method"] == m]
            a, mc = np.array([x["acc"] for x in r]), np.array([x["macro_acc"] for x in r])
            summ.append(dict(K=k, method=m, n_seeds=len(r), acc_mean=a.mean(), acc_std=a.std(ddof=1),
                             macro_mean=mc.mean(), macro_std=mc.std(ddof=1)))
            lines.append(f"| {k} | {m} | {a.mean():.2f} ± {a.std(ddof=1):.2f} | {mc.mean():.2f} ± {mc.std(ddof=1):.2f} |")
    with open(os.path.join(out_dir, "summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summ[0].keys()))
        w.writeheader()
        w.writerows(summ)
    open(os.path.join(out_dir, "summary.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))

    for (k, m), cm in cms.items():
        with open(os.path.join(out_dir, f"confusion_K{k}_{m}.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["true\\pred"] + names)
            for n, row in zip(names, cm):
                w.writerow([n] + list(row))
    # resource use of calibration + inference on cached features (feature extraction is separate)
    r = [x for x in rows if x["method"] != "zero_shot"]
    res = dict(dataset=args.dataset, n_test=int(len(y)), n_classes=n_cls,
               calib_seconds_mean=float(np.mean([x["calib_s"] for x in r])),
               infer_seconds_mean=float(np.mean([x["infer_s"] for x in r])),
               peak_rss_mb=C.peak_rss_mb(),
               feature_extract_seconds=float(d["extract_seconds"]) if "extract_seconds" in d.files else None,
               feature_extract_peak_rss_mb=float(d["extract_peak_rss_mb"]) if "extract_peak_rss_mb" in d.files else None,
               feature_extract_peak_mps_mb=float(d["extract_peak_mps_mb"]) if "extract_peak_mps_mb" in d.files else None)
    json.dump(res, open(os.path.join(out_dir, "resources.json"), "w"), indent=1)
    print(res)
    print("saved to", out_dir)


if __name__ == "__main__":
    main()
