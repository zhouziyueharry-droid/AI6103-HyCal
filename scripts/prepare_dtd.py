"""Prepare official DTD images and CoOp/Zhou manifests for frozen transfer testing.

Standard library only. No model features, predictions, or metrics are computed.
Run `python scripts/prepare_dtd.py` or append `--verify-only` for a read-only audit.
Supports use each class's first five entries from a seeded ten-image draw.
"""

from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
from contextlib import redirect_stdout
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import platform
import random
import re
import tarfile
import time
from types import SimpleNamespace
import urllib.request


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SEEDS = [0, 1, 42, 1993, *range(2, 18)]
SHOTS = [5]
CLASS_COUNT = 47
IMAGE_COUNT = 5640
EXPECTED_COUNTS = {"train": 2820, "val": 1128, "test": 1692}
EXPECTED_PER_CLASS = {"train": 60, "val": 24, "test": 36}
COOP_COMMIT = "ff61507c790454bce7c5052c3ac39e60772f1f89"
HYCAL_COMMIT = "4558fb7fa6aa389310c9e9dc7db7c2cbebc84a41"
SOURCES = {
    "archive": {
        "relative_path": "dtd-r1.0.1.tar.gz",
        "url": "https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz",
        "sha256": "e42855a52a4950a3b59612834602aa253914755c95b0cff9ead6d07395f8e205",
    },
    "zhou_split": {
        "relative_path": "split_zhou_DescribableTextures.json",
        "url": "https://drive.google.com/uc?export=download&id=1u3_QfB467jqHgNXC00UIzbLZRQCg2S7x",
        "sha256": "f26ebeccd2b58e68d70f07a0dc39fe49bf7b69024bac34396b954dfd87969c38",
    },
    "coop_dataset_instructions": {
        "relative_path": "sources/CoOp_DATASETS.md",
        "url": f"https://raw.githubusercontent.com/KaiyangZhou/CoOp/{COOP_COMMIT}/DATASETS.md",
        "sha256": "3de6caf6c48fbd10c814734b6561e348863aa9e36462cd83266e80c17dbafba3",
    },
    "coop_dtd_loader": {
        "relative_path": "sources/CoOp_dtd.py",
        "url": f"https://raw.githubusercontent.com/KaiyangZhou/CoOp/{COOP_COMMIT}/datasets/dtd.py",
        "sha256": "e38afe03f9c3a858f7335f53659aa9ae7f15c029f2dec6b0c7a9ab48856f808b",
    },
    "hycal_dtd_loader": {
        "relative_path": "sources/HyCal_dtd.py",
        "url": f"https://raw.githubusercontent.com/EJLEE5826/HyCal-CIL/{HYCAL_COMMIT}/task_datasets/dtd.py",
        "sha256": "319b36561967010b4f3a650ca576971e816fd60ef27cba90532250bc8260a0a2",
    },
    "hycal_sampling_code": {
        "relative_path": "sources/HyCal_dataset_utils.py",
        "url": f"https://raw.githubusercontent.com/EJLEE5826/HyCal-CIL/{HYCAL_COMMIT}/task_datasets/utils.py",
        "sha256": "117f6479b57006c23717a5f4aa90aee3f9f8155bc4b4baed56e6033f5be494d9",
    },
    "oxford_readme": {
        "relative_path": "sources/Oxford_DTD_README.txt",
        "url": "https://www.robots.ox.ac.uk/~vgg/data/dtd/download/README.txt",
        "sha256": "7f11a2225f0a140b1c7e96889e3c4cca6cba6cb3148327e84d5c361aa3793413",
    },
}


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def fetch_sources(raw: Path, verify_only: bool) -> dict:
    sources = {}
    for name, source in SOURCES.items():
        path = raw / source["relative_path"]
        if not path.is_file():
            if verify_only:
                raise FileNotFoundError(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(path.suffix + ".part")
            print(f"Downloading {source['url']}", flush=True)
            last_update = time.monotonic()
            with urllib.request.urlopen(source["url"], timeout=90) as response, temporary.open("wb") as output:
                while chunk := response.read(1024 * 1024):
                    output.write(chunk)
                    if time.monotonic() - last_update >= 15:
                        print(f"  {path.name}: {output.tell():,} bytes", flush=True)
                        last_update = time.monotonic()
            if sha256(temporary) != source["sha256"]:
                raise ValueError(f"Downloaded SHA256 mismatch: {path}")
            temporary.replace(path)
        if sha256(path) != source["sha256"]:
            raise ValueError(f"Cached SHA256 mismatch: {path}")
        sources[name] = {**source, "path": str(path.resolve()), "bytes": path.stat().st_size, "sha256_verified": True}
    return sources


def extract_and_verify(raw: Path, verify_only: bool) -> dict:
    """Extract ordinary files under dtd/ only; check every image against tar bytes."""
    images = {}
    root = raw.resolve()
    with tarfile.open(raw / "dtd-r1.0.1.tar.gz", "r:gz") as archive:
        for member in archive:
            original = PurePosixPath(member.name)
            if original.is_absolute() or ".." in original.parts or "\\" in member.name:
                raise ValueError(f"Unsafe archive path: {member.name}")
            if not original.parts or original.parts[0] != "dtd":
                raise ValueError(f"Unexpected archive root: {member.name}")
            relative = PurePosixPath(*original.parts[1:])
            if not re.fullmatch(r"[A-Za-z0-9_./ -]+", str(relative)):
                raise ValueError(f"Unsafe archive filename: {member.name}")
            target = (raw / Path(*relative.parts)).resolve()
            if not target.is_relative_to(root):
                raise ValueError(f"Archive path escapes dataset root: {member.name}")
            if member.isdir():
                if not verify_only:
                    target.mkdir(parents=True, exist_ok=True)
                continue
            if not member.isfile():
                raise ValueError(f"Archive links/special files are forbidden: {member.name}")
            stream = archive.extractfile(member)
            if stream is None:
                raise ValueError(f"Unreadable archive member: {member.name}")
            with stream:
                data = stream.read()
            if len(data) != member.size:
                raise ValueError(f"Archive member length mismatch: {member.name}")
            digest = hashlib.sha256(data).hexdigest()
            intact = target.is_file() and target.stat().st_size == len(data) and sha256(target) == digest
            if not intact:
                if verify_only:
                    raise ValueError(f"Missing/corrupt extracted file: {target}")
                target.parent.mkdir(parents=True, exist_ok=True)
                temporary = target.with_suffix(target.suffix + ".part")
                temporary.write_bytes(data)
                temporary.replace(target)
            if relative.parts and relative.parts[0] == "images":
                # The official release includes one KDE folder metadata file.
                if str(relative) == "images/waffled/.directory":
                    continue
                if len(relative.parts) != 3 or relative.suffix.lower() != ".jpg" or not data.startswith(b"\xff\xd8\xff"):
                    raise ValueError(f"Unexpected image file: {member.name}")
                sample_id = str(relative)
                if sample_id in images:
                    raise ValueError(f"Duplicate archive image: {sample_id}")
                images[sample_id] = {"sha256": digest, "bytes": len(data)}
    on_disk = {path.relative_to(raw).as_posix() for path in (raw / "images").rglob("*.jpg")}
    if len(images) != IMAGE_COUNT or on_disk != set(images):
        raise ValueError("Image archive/directory does not contain exactly 5,640 matching JPEGs")
    category_counts = Counter(PurePosixPath(name).parts[1] for name in images)
    if len(category_counts) != CLASS_COUNT or set(category_counts.values()) != {120}:
        raise ValueError("Original archive does not contain 47 categories with 120 images each")
    print(f"Verified {len(images):,} extracted images against official archive bytes", flush=True)
    return images


def official_support_crosscheck(raw: Path, records: list, support: dict) -> None:
    """Execute only two unchanged methods from the hash-pinned official sampler."""
    source = raw / "sources/HyCal_dataset_utils.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    base = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "DatasetBase")
    method_names = {"generate_fewshot_dataset", "split_dataset_by_label"}
    base.body = [node for node in base.body if isinstance(node, ast.FunctionDef) and node.name in method_names]
    if len(base.body) != 2:
        raise ValueError("Cannot locate the official sampling methods")
    base.bases, base.decorator_list = [], []
    namespace = {"random": random, "defaultdict": defaultdict}
    module = ast.fix_missing_locations(ast.Module(body=[base], type_ignores=[]))
    exec(compile(module, str(source), "exec"), namespace)
    items = [SimpleNamespace(sample_id=row["sample_id"], label=row["label"]) for row in records if row["split"] == "train"]
    prior_state = random.getstate()
    try:
        with redirect_stdout(io.StringIO()):
            for seed in SEEDS:
                random.seed(seed)
                official = namespace["DatasetBase"]().generate_fewshot_dataset(items, num_shots=10)
                prefixes = [item.sample_id for start in range(0, len(official), 10) for item in official[start:start + 5]]
                if prefixes != support["indices"][str(seed)]["5"]:
                    raise ValueError(f"Official sampler mismatch for seed {seed}")
    finally:
        random.setstate(prior_state)


def create_manifests(raw: Path, images: dict) -> tuple[dict, dict]:
    split = json.loads((raw / "split_zhou_DescribableTextures.json").read_text(encoding="utf-8"))
    if set(split) != set(EXPECTED_COUNTS):
        raise ValueError("Unexpected Zhou split names")
    records, class_names, seen = [], {}, set()
    split_ids, counts, train = {}, {}, defaultdict(list)
    for split_name, expected_count in EXPECTED_COUNTS.items():
        if len(split[split_name]) != expected_count:
            raise ValueError(f"Unexpected {split_name} count")
        split_ids[split_name] = set()
        counts[split_name] = Counter()
        for entry in split[split_name]:
            if not isinstance(entry, list) or len(entry) != 3:
                raise ValueError(f"Malformed Zhou entry: {entry!r}")
            filename, label, classname = entry
            if not isinstance(filename, str) or not re.fullmatch(r"[a-z-]+/[A-Za-z0-9_-]+\.jpg", filename):
                raise ValueError(f"Invalid Zhou image path: {filename!r}")
            if type(label) is not int or label not in range(CLASS_COUNT):
                raise ValueError(f"Invalid label: {label!r}")
            if not isinstance(classname, str) or PurePosixPath(filename).parts[0] != classname:
                raise ValueError(f"Zhou classname disagrees with original image category: {entry!r}")
            if label in class_names and class_names[label] != classname:
                raise ValueError(f"Inconsistent class mapping: {label}")
            class_names[label] = classname
            sample_id = f"images/{filename}"
            if sample_id in seen or sample_id not in images:
                raise ValueError(f"Duplicate, overlapping, or missing image: {sample_id}")
            seen.add(sample_id)
            split_ids[split_name].add(sample_id)
            counts[split_name][label] += 1
            records.append({"sample_id": sample_id, "path": str((raw / "images" / filename).resolve()), "split": split_name, "label": label, "classname": classname})
            if split_name == "train":
                train[label].append(sample_id)
        if counts[split_name] != {label: EXPECTED_PER_CLASS[split_name] for label in range(CLASS_COUNT)}:
            raise ValueError(f"Unexpected {split_name} per-class counts")
    if seen != set(images):
        raise ValueError("Zhou split image set does not match official image archive")
    if [class_names[label] for label in range(CLASS_COUNT)] != sorted(class_names.values()):
        raise ValueError("Zhou label ordering does not match official alphabetical class mapping")
    classes = [{"label": label, "classname": class_names[label], "global_id": f"dtd:{label}"} for label in range(CLASS_COUNT)]
    supports = {"seeds": SEEDS, "shots": SHOTS, "indices": {}}
    lookup = {row["sample_id"]: row["label"] for row in records}
    for seed in SEEDS:
        rng = random.Random(seed)
        sampled = [sample_id for items in train.values() for sample_id in rng.sample(items, 10)[:5]]
        if len(sampled) != CLASS_COUNT * 5 or len(set(sampled)) != len(sampled):
            raise ValueError(f"Invalid support count/uniqueness: seed {seed}")
        if Counter(lookup[item] for item in sampled) != {label: 5 for label in range(CLASS_COUNT)}:
            raise ValueError(f"Invalid support per-class counts: seed {seed}")
        if not set(sampled) <= split_ids["train"]:
            raise ValueError(f"Support contains held-out samples: seed {seed}")
        supports["indices"][str(seed)] = {"5": sampled}
    official_support_crosscheck(raw, records, supports)
    checks = {
        "total_images": len(images), "num_classes": len(classes),
        "split_counts": {name: len(ids) for name, ids in split_ids.items()},
        "per_class_counts": {name: {str(label): count[label] for label in range(CLASS_COUNT)} for name, count in counts.items()},
        "split_intersections": {"train_val": len(split_ids["train"] & split_ids["val"]), "train_test": len(split_ids["train"] & split_ids["test"]), "val_test": len(split_ids["val"] & split_ids["test"])},
        "minimum_train_images_per_class": min(map(len, train.values())),
        "train_class_encounter_order": list(train),
        "support_seeds": SEEDS, "support_counts_per_seed": {"5": CLASS_COUNT * 5},
        "all_images_exist_and_match_archive_bytes": True,
        "all_splits_contain_47_classes": True,
        "support_train_only": True, "support_ids_unique_within_each_seed": True,
        "support_exactly_five_per_class": True,
        "official_hycal_ten_shot_per_class_prefixes_match_all_seeds": True,
        "class_names_match_original_archive_category_directories": True,
        "known_non_image_archive_metadata": ["images/waffled/.directory"],
    }
    return {"records.json": records, "classes.json": classes, "support_indices.json": supports}, checks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    data_root = args.data_root.expanduser().resolve()
    raw, manifests = data_root / "raw/dtd", data_root / "manifests/dtd"
    sources = fetch_sources(raw, args.verify_only)
    images = extract_and_verify(raw, args.verify_only)
    outputs, checks = create_manifests(raw, images)
    inventory = raw / "sources/image_inventory.json"
    if args.verify_only:
        for name, expected in outputs.items():
            if json.loads((manifests / name).read_text(encoding="utf-8")) != expected:
                raise ValueError(f"Manifest does not reproduce exactly: {name}")
        provenance = json.loads((manifests / "provenance.json").read_text(encoding="utf-8"))
        if provenance["sources"] != sources or provenance["validation"] != checks:
            raise ValueError("Provenance differs from verified sources/checks")
        for name in outputs:
            if sha256(manifests / name) != provenance["manifest_sha256"][name]:
                raise ValueError(f"Manifest SHA256 mismatch: {name}")
        if sha256(inventory) != provenance["image_inventory_sha256"] or json.loads(inventory.read_text(encoding="utf-8")) != images:
            raise ValueError("Image inventory differs from official archive bytes")
        print("PASS: all DTD sources, images, manifests and 20-seed supports verify", flush=True)
    else:
        for name, value in outputs.items():
            write_json(manifests / name, value)
        write_json(inventory, images)
        write_json(manifests / "provenance.json", {
            "dataset": "Describable Textures Dataset (DTD), release 1.0.1",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "python_version": platform.python_version(),
            "source_discovery": {
                "official_oxford_page": "https://www.robots.ox.ac.uk/~vgg/data/dtd/",
                "official_coop_instructions": f"https://github.com/KaiyangZhou/CoOp/blob/{COOP_COMMIT}/DATASETS.md#dtd",
                "official_coop_split_link": "https://drive.google.com/file/d/1u3_QfB467jqHgNXC00UIzbLZRQCg2S7x/view?usp=sharing",
                "coop_commit": COOP_COMMIT, "hycal_commit": HYCAL_COMMIT,
                "split_mirror_used": False,
                "hash_note": "SHA256 values were computed from downloaded official bytes and pinned here; they are not publisher-supplied checksums.",
            },
            "sources": sources,
            "split_protocol": "Use the fixed CoOp/Zhou 50/20/30 percent split (60/24/36 images per class), not Oxford's original 40/40/40 split.",
            "sampling": {
                "seeds": SEEDS, "shots": SHOTS,
                "rng": "One isolated Python random.Random(seed) instance per seed",
                "class_order": "First encounter order in the original Zhou train array, matching HyCal tracker.items()",
                "within_class_order": "Original Zhou train array order, without sorting or shuffling",
                "algorithm": "Traverse classes once, rng.sample(train_items, 10); concatenate each class draw[:5]",
                "official_equivalence": "Matches the first five samples per class of the official 10-shot sampler; intentionally differs from a separate official 5-shot draw",
                "support_source": "train only",
                "validation_use": "Full Zhou validation split is retained for data organization only; no DTD-specific parameter selection is authorized by this preparation step",
                "cross_seed_overlap": "Allowed; duplicate image IDs are forbidden within each seed",
            },
            "transfer_protocol": {
                "purpose": "Transfer evaluation after model/configuration selection is frozen on Flowers",
                "dtd_hyperparameter_selection_allowed": False,
                "features_or_model_metrics_computed_by_this_script": False,
                "test_label_use": "Labels are read only for dataset integrity, class mapping, and split checks during preparation",
                "encoder": "Use the existing frozen OpenAI CLIP ViT-B/16 protocol downstream; no encoding is performed here",
            },
            "validation": checks,
            "manifest_sha256": {name: sha256(manifests / name) for name in outputs},
            "image_inventory_path": str(inventory.resolve()),
            "image_inventory_sha256": sha256(inventory),
        })
        print(f"Wrote DTD manifests to {manifests}", flush=True)
    print(json.dumps({key: checks[key] for key in ("total_images", "num_classes", "split_counts", "split_intersections", "support_counts_per_seed")}, indent=2))


if __name__ == "__main__":
    main()
