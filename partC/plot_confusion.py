"""Plot a row-normalised confusion matrix saved by run_single_domain.py.

Usage: python plot_confusion.py --dataset eurosat --k 5 --method md
"""
import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import common as C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="eurosat", choices=C.DATASETS)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--method", default="cosine")
    args = ap.parse_args()
    d = os.path.join(C.RESULT_DIR, "single_domain", args.dataset)
    rows = list(csv.reader(open(os.path.join(d, f"confusion_K{args.k}_{args.method}.csv"))))
    names = rows[0][1:]
    cm = np.array([[int(v) for v in r[1:]] for r in rows[1:]])
    pct = 100.0 * cm / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(8.5, 7))
    im = ax.imshow(pct, cmap="Blues", vmin=0, vmax=100)
    ax.set_xticks(range(len(names)))
    ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, rotation=40, ha="right")
    ax.set_yticklabels(names)
    for i in range(len(names)):
        for j in range(len(names)):
            if pct[i, j] >= 0.5:
                ax.text(j, i, f"{pct[i, j]:.0f}", ha="center", va="center", fontsize=8,
                        color="white" if pct[i, j] > 55 else "#222")
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title(f"{args.dataset}, {args.k}-shot, {args.method} (row %, summed over seeds)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="% of true class")
    fig.tight_layout()
    out = os.path.join(d, f"confusion_K{args.k}_{args.method}.png")
    fig.savefig(out, dpi=160)
    print("saved", out)


if __name__ == "__main__":
    main()
