"""Pick and plot representative error cases from saved single-domain predictions.

Cases are chosen deterministically: for each (true class, wrong class, pattern) slot we take the
test image with the largest score margin of the wrong class over the true class, i.e. the most
confident mistake of that kind.  Patterns:
  fixed   - Cosine-only wrong, Mahalanobis right   (what the distribution-shape evidence repairs)
  broken  - Cosine-only right, Mahalanobis wrong   (what it costs)
  both    - both wrong with the same prediction     (limits of the frozen features)

EuroSAT uses the hand-picked confusion pairs in SLOTS; other datasets take the most frequent
(true, wrong) pairs of each pattern automatically.

Outputs partC/results/single_domain/{dataset}/error_cases_K{K}_seed{seed}.{png,csv}
"""
import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import common as C

# (pattern, true class, predicted-wrong class) using short EuroSAT names
SLOTS = [
    ("fixed", "Forest", "Sea or Lake"),
    ("fixed", "Annual Crop Land", "Permanent Crop Land"),
    ("fixed", "Annual Crop Land", "Herbaceous Vegetation Land"),
    ("broken", "Sea or Lake", "Forest"),
    ("broken", "Highway or Road", "River"),
    ("both", "Annual Crop Land", "Permanent Crop Land"),
    ("both", "Permanent Crop Land", "Annual Crop Land"),
    ("both", "Industrial Buildings", "Residential Buildings"),
    ("both", "Highway or Road", "River"),
    ("both", "Herbaceous Vegetation Land", "Forest"),
]
PATTERN_LABEL = {"fixed": "cosine ✗  MD ✓", "broken": "cosine ✓  MD ✗", "both": "cosine ✗  MD ✗"}


def auto_slots(y, cos, md, names, n_fixed=3, n_broken=2, n_both=5):
    """Most frequent (true, wrong) pairs for each pattern."""
    slots = []
    for pattern, n in [("fixed", n_fixed), ("broken", n_broken), ("both", n_both)]:
        if pattern == "fixed":
            mask, wrong = (cos != y) & (md == y), cos
        elif pattern == "broken":
            mask, wrong = (cos == y) & (md != y), md
        else:
            mask, wrong = (cos != y) & (md == cos), md
        pairs, counts = np.unique(np.stack([y[mask], wrong[mask]], axis=1), axis=0, return_counts=True)
        for i in np.argsort(-counts)[:n]:
            slots.append((pattern, names[pairs[i][0]], names[pairs[i][1]]))
    return slots


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="eurosat", choices=C.DATASETS)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    out_dir = os.path.join(C.RESULT_DIR, "single_domain", args.dataset)
    image_root = C.load_dataset(args.dataset)["image_root"]
    p = np.load(os.path.join(out_dir, f"preds_K{args.k}_seed{args.seed}.npz"))
    d = np.load(C.feature_cache(args.dataset))
    names = [str(n) for n in d["display_names"]]
    y, cos, md = p["labels"], p["cosine"], p["md"]
    cos_s, md_s = p["cos_scores"], p["md_scores"]

    slots = SLOTS if args.dataset == "eurosat" else auto_slots(y, cos, md, names)
    rows, used = [], set()
    for pattern, t_name, w_name in slots:
        t, w = names.index(t_name), names.index(w_name)
        if pattern == "fixed":
            mask = (y == t) & (cos == w) & (md == t)
            margin = cos_s[:, w] - cos_s[:, t]
        elif pattern == "broken":
            mask = (y == t) & (cos == t) & (md == w)
            margin = md_s[:, w] - md_s[:, t]
        else:
            mask = (y == t) & (cos == w) & (md == w)
            margin = md_s[:, w] - md_s[:, t]
        cand = [i for i in np.argsort(-np.where(mask, margin, -np.inf)) if mask[i] and i not in used]
        if not cand:
            print("no case for", pattern, t_name, w_name)
            continue
        i = int(cand[0])
        used.add(i)
        rows.append(dict(pattern=pattern, test_index=i, path=str(p["test_paths"][i]), true=t_name,
                         cosine_pred=names[cos[i]], md_pred=names[md[i]],
                         n_same_kind=int(mask.sum()),
                         cos_score_true=float(cos_s[i, t]), cos_score_wrong=float(cos_s[i, w]),
                         md_score_true=float(md_s[i, t]), md_score_wrong=float(md_s[i, w])))

    base = os.path.join(out_dir, f"error_cases_K{args.k}_seed{args.seed}")
    with open(base + ".csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)

    n = len(rows)
    ncol = 5
    nrow = (n + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.3 * ncol, 4.0 * nrow))
    for ax in np.array(axes).ravel():
        ax.axis("off")
    for ax, r in zip(np.array(axes).ravel(), rows):
        img = Image.open(os.path.join(image_root, r["path"])).convert("RGB")
        ax.imshow(img.resize((224, 224), Image.BICUBIC))
        ax.set_title(f"{PATTERN_LABEL[r['pattern']]}\ntrue: {r['true']}\ncosine: {r['cosine_pred']}\n"
                     f"MD: {r['md_pred']}", fontsize=9, loc="left")
    fig.suptitle(f"{args.dataset} {args.k}-shot (seed {args.seed}): representative errors, "
                 f"Cosine-only vs Mahalanobis-only", fontsize=12)
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.84, hspace=0.42, wspace=0.06)
    fig.savefig(base + ".png", dpi=150)
    for r in rows:
        print(r["pattern"], "|", r["true"], "->", r["cosine_pred"], "/", r["md_pred"], "|", r["path"],
              "| n of this kind:", r["n_same_kind"])
    print("saved", base + ".png")


if __name__ == "__main__":
    main()
