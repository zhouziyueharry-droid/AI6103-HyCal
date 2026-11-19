"""Shared paths, constants and dataset definitions for Part C (remote sensing + astronomy)."""
import csv
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HYCAL_REPO = os.path.join(ROOT, "external", "HyCal-CIL")
DATA_ROOT = os.path.join(ROOT, "data")
PARTC = os.path.join(ROOT, "partC")
CACHE_DIR = os.path.join(PARTC, "cache")
INDEX_DIR = os.path.join(PARTC, "indices")
RESULT_DIR = os.path.join(PARTC, "results")


def peak_rss_mb():
    """Peak process memory in MiB, or None if Windows telemetry is unavailable."""
    if sys.platform == "win32":
        try:
            import psutil
        except ImportError:
            return None
        return psutil.Process().memory_info().peak_wset / 1024**2
    import resource
    scale = 1024**2 if sys.platform == "darwin" else 1024
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / scale

# Settings of the HyCal main paper: OpenAI CLIP ViT-B/16, image+text sum fusion
CLIP_MODEL = "ViT-B/16"
TEMPLATE = "a photo of a {}."          # HyCalModel default template

# ---------------------------------------------------------------- EuroSAT RGB
EUROSAT_DIR = os.path.join(DATA_ROOT, "eurosat")
EUROSAT_IMAGES = os.path.join(EUROSAT_DIR, "2750")
EUROSAT_SPLIT = os.path.join(EUROSAT_DIR, "split_zhou_EuroSAT.json")   # CoOp split
CLASSNAME_SUFFIX = " satellite location"   # official task_datasets/eurosat.py appends this

# ---------------------------------------------------------------- Galaxy10 DECaLS
GALAXY_DIR = os.path.join(DATA_ROOT, "Galaxy10_DECals", "_galaxy_temp")   # official converter output
GALAXY_IMAGES = os.path.join(GALAXY_DIR, "images")
GALAXY_SPLIT = os.path.join(GALAXY_DIR, "galaxy10_split.csv")             # stratified 80/10/10, seed 42
GALAXY_CLASSNAMES = [                      # wording of official task_datasets/galaxy10_decals.py
    "Disturbed Galaxies", "Merging Galaxies", "Round Smooth Galaxies",
    "In-between Round Smooth Galaxies", "Cigar Shaped Smooth Galaxies", "Barred Spiral Galaxies",
    "Unbarred Tight Spiral Galaxies", "Unbarred Loose Spiral Galaxies",
    "Edge-on Galaxies without Bulge", "Edge-on Galaxies with Bulge",
]

DATASETS = ["eurosat", "galaxy10"]


def feature_cache(dataset):
    return os.path.join(CACHE_DIR, f"{dataset}_ViT-B-16.npz")


FEATURE_CACHE = feature_cache("eurosat")   # kept for backwards compatibility


def index_file(dataset, k, seed):
    return os.path.join(INDEX_DIR, f"{dataset}_K{k}_seed{seed}.json")


def load_dataset(name):
    """Return the fixed splits of a dataset.

    dict(image_root, split_file,
         splits={'train'|'val'|'test': [(relative_path, label), ...]}   # order of the split file
         prompt_names=[...]   # per label, text used in the CLIP prompt (official wording)
         display_names=[...]) # per label, short name for tables/plots
    Global class ids are f"{name}:{label}".
    """
    if name == "eurosat":
        split = json.load(open(EUROSAT_SPLIT))
        lab2name = {}
        splits = {}
        for s in ["train", "val", "test"]:
            splits[s] = [(p, int(l)) for p, l, _ in split[s]]
            for _, l, n in split[s]:
                lab2name[int(l)] = n
        display = [lab2name[i] for i in range(len(lab2name))]
        return dict(image_root=EUROSAT_IMAGES, split_file=EUROSAT_SPLIT, splits=splits,
                    prompt_names=[n + CLASSNAME_SUFFIX for n in display], display_names=display)
    if name == "galaxy10":
        splits = {"train": [], "val": [], "test": []}
        with open(GALAXY_SPLIT) as f:
            for row in csv.DictReader(f):
                splits[row["split"].lower()].append((row["filename"], int(row["class_num"])))
        display = [n.replace(" Galaxies", "") for n in GALAXY_CLASSNAMES]
        return dict(image_root=GALAXY_IMAGES, split_file=GALAXY_SPLIT, splits=splits,
                    prompt_names=list(GALAXY_CLASSNAMES), display_names=display)
    raise ValueError(f"unknown dataset {name}; choose from {DATASETS}")
