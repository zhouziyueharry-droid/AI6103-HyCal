"""Freeze few-shot Support indices (train split only) for a dataset.

Sampling reproduces the official HyCal code path exactly:
  set_seed(seed) -> random.seed(seed)
  DatasetBase.generate_fewshot_dataset: for each label in order of first appearance in the
  train list, random.sample(items_of_label, K)
so the indices written here are the same images the official runner picks for the same seed
when that dataset is the first/only one.  (In a multi-dataset official run the RNG state
depends on the datasets sampled before it; our merged experiments use these frozen files.)

Output: partC/indices/{dataset}_K{K}_seed{seed}.json
Usage:  python make_support_indices.py --dataset eurosat|galaxy10
"""
import argparse
import json
import os
import random
from collections import OrderedDict

import common as C


def sample_support(train_items, k, seed):
    random.seed(seed)
    by_label = OrderedDict()
    for idx, (_, label) in enumerate(train_items):
        by_label.setdefault(int(label), []).append(idx)
    chosen = []
    for label, idxs in by_label.items():
        chosen.extend(random.sample(idxs, k))
    return chosen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="eurosat", choices=C.DATASETS)
    ap.add_argument("--shots", type=int, nargs="+", default=[1, 5, 10])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 42])
    args = ap.parse_args()

    ds = C.load_dataset(args.dataset)
    train = ds["splits"]["train"]
    os.makedirs(C.INDEX_DIR, exist_ok=True)
    for k in args.shots:
        for seed in args.seeds:
            idxs = sample_support(train, k, seed)
            rec = {
                "dataset": args.dataset, "split": "train", "split_file": os.path.basename(ds["split_file"]),
                "shots_per_class": k, "seed": seed,
                "train_indices": idxs,
                "paths": [train[i][0] for i in idxs],
                "labels": [train[i][1] for i in idxs],
                "class_ids": [f"{args.dataset}:{train[i][1]}" for i in idxs],
            }
            path = C.index_file(args.dataset, k, seed)
            json.dump(rec, open(path, "w"), indent=1)
            print(path, len(idxs))


if __name__ == "__main__":
    main()
