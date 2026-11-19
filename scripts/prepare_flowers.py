"""Download, verify, and prepare the fixed Flowers102 Part A manifests.

Python standard library only. Run from any directory:
    python scripts/prepare_flowers.py
    python scripts/prepare_flowers.py --verify-only

The 1/5-shot supports are per-class prefixes of one 10-shot draw, not
independently sampled official 1/5-shot supports. All source order is retained.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import platform
import random
import re
import tarfile
import time
import urllib.request


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SEEDS = [0, 1, 42, 1993]
SHOTS = [1, 5, 10]
EXPECTED_COUNTS = {"train": 4093, "val": 1633, "test": 2463}
HYCal_COMMIT = "4558fb7fa6aa389310c9e9dc7db7c2cbebc84a41"
COOP_COMMIT = "ff61507c790454bce7c5052c3ac39e60772f1f89"
COOP_DATASETS_URL = (
    f"https://github.com/KaiyangZhou/CoOp/blob/{COOP_COMMIT}/DATASETS.md#flowers102"
)
DRIVE_SPLIT_URL = (
    "https://drive.google.com/file/d/1Pp0sRXzZFZq15zVOzKjKBu4A9i01nozT/view?usp=sharing"
)
SOURCES = {
    "images": {
        "relative_path": "102flowers.tgz",
        "url": "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz",
        "sha256": "2d01ecc807db462958cfe3d92f57a8c252b4abd240eb955770201e45f783b246",
    },
    "zhou_split": {
        "relative_path": "split_zhou_OxfordFlowers.json",
        "url": "https://drive.google.com/uc?export=download&id=1Pp0sRXzZFZq15zVOzKjKBu4A9i01nozT",
        "sha256": "2f50c42fc1a17ca13c271cb4269b1347bdb43d2be672e34f74a4e4cbfbcca794",
    },
    "class_names": {
        "relative_path": "cat_to_name.json",
        "url": "https://drive.google.com/uc?export=download&id=1AkcxCXeK_RCGCEC_GvmWxjcjaNhu-at0",
        "sha256": "0469f1ad7fe0072d98155c49aa58649d998edbbe3326229003aa5f7228525236",
    },
    "original_image_labels": {
        "relative_path": "imagelabels.mat",
        "url": "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/imagelabels.mat",
        "sha256": "4903e94206bac23bf772aadf06451916df56b58fc483a62db32a97b82656651d",
    },
    "coop_dataset_instructions": {
        "relative_path": "sources/CoOp_DATASETS.md",
        "url": f"https://raw.githubusercontent.com/KaiyangZhou/CoOp/{COOP_COMMIT}/DATASETS.md",
        "sha256": "3de6caf6c48fbd10c814734b6561e348863aa9e36462cd83266e80c17dbafba3",
    },
    "hycal_flowers_loader": {
        "relative_path": "sources/HyCal_oxford_flowers.py",
        "url": f"https://raw.githubusercontent.com/EJLEE5826/HyCal-CIL/{HYCal_COMMIT}/task_datasets/oxford_flowers.py",
        "sha256": "d1aa4df3b46dd88d741a5cc81a51dd4cb0515104b5a6f614a7d7ed8bde075c59",
    },
    "hycal_sampling_code": {
        "relative_path": "sources/HyCal_dataset_utils.py",
        "url": f"https://raw.githubusercontent.com/EJLEE5826/HyCal-CIL/{HYCal_COMMIT}/task_datasets/utils.py",
        "sha256": "117f6479b57006c23717a5f4aa90aee3f9f8155bc4b4baed56e6033f5be494d9",
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def fetch_sources(raw: Path, verify_only: bool) -> dict:
    provenance = {}
    for name, source in SOURCES.items():
        path = raw / source["relative_path"]
        if not path.is_file():
            if verify_only:
                raise FileNotFoundError(f"Missing source artifact: {path}")
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(path.suffix + ".part")
            print(f"Downloading {source['url']}", flush=True)
            last_print = time.monotonic()
            with urllib.request.urlopen(source["url"], timeout=90) as response, temporary.open("wb") as output:
                while chunk := response.read(1024 * 1024):
                    output.write(chunk)
                    if time.monotonic() - last_print >= 15:
                        print(f"  {path.name}: {output.tell():,} bytes", flush=True)
                        last_print = time.monotonic()
            if sha256(temporary) != source["sha256"]:
                raise ValueError(f"Downloaded artifact checksum mismatch: {temporary}")
            temporary.replace(path)
        actual_hash = sha256(path)
        if actual_hash != source["sha256"]:
            raise ValueError(f"Cached artifact checksum mismatch: {path}")
        provenance[name] = {
            **source,
            "path": str(path.resolve()),
            "bytes": path.stat().st_size,
            "sha256_verified": True,
        }
    return provenance


def extract_and_verify_images(raw: Path, verify_only: bool) -> dict[str, dict]:
    """Reject links/path escapes and compare every extracted file to the tar bytes."""
    image_info = {}
    root = raw.resolve()
    with tarfile.open(raw / "102flowers.tgz", "r:gz") as archive:
        for member in archive:
            name = PurePosixPath(member.name)
            if name.is_absolute() or ".." in name.parts or "\\" in member.name:
                raise ValueError(f"Unsafe archive member: {member.name}")
            target = (raw / Path(*name.parts)).resolve()
            if not target.is_relative_to(root):
                raise ValueError(f"Archive target escapes dataset directory: {target}")
            if member.isdir() and str(name) == "jpg":
                if not verify_only:
                    target.mkdir(parents=True, exist_ok=True)
                continue
            if not member.isfile() or not re.fullmatch(r"jpg/image_\d{5}\.jpg", str(name)):
                raise ValueError(f"Unexpected archive member/type: {member.name}")
            if str(name) in image_info:
                raise ValueError(f"Duplicate archive image: {name}")
            stream = archive.extractfile(member)
            if stream is None:
                raise ValueError(f"Unreadable archive member: {name}")
            with stream:
                data = stream.read()
            expected_hash = hashlib.sha256(data).hexdigest()
            if len(data) != member.size or not data.startswith(b"\xff\xd8\xff"):
                raise ValueError(f"Invalid JPEG bytes/length: {name}")
            intact = target.is_file() and target.stat().st_size == member.size and sha256(target) == expected_hash
            if not intact:
                if verify_only:
                    raise ValueError(f"Missing or corrupt extracted image: {target}")
                target.parent.mkdir(parents=True, exist_ok=True)
                temporary = target.with_suffix(".jpg.part")
                temporary.write_bytes(data)
                temporary.replace(target)
            image_info[str(name)] = {"sha256": expected_hash, "bytes": member.size}
    expected_ids = {f"jpg/image_{i:05d}.jpg" for i in range(1, 8190)}
    if set(image_info) != expected_ids:
        raise ValueError("Archive does not contain exactly the 8,189 expected image IDs")
    on_disk = {p.relative_to(raw).as_posix() for p in (raw / "jpg").glob("*.jpg")}
    if on_disk != expected_ids:
        raise ValueError("Extracted directory contains missing or extra JPEG images")
    print(f"Verified {len(image_info):,} original image files against archive bytes", flush=True)
    return image_info


def create_manifests(raw: Path, image_info: dict) -> tuple[list, list, dict, dict]:
    split = json.loads((raw / "split_zhou_OxfordFlowers.json").read_text(encoding="utf-8"))
    class_names = json.loads((raw / "cat_to_name.json").read_text(encoding="utf-8"))
    if set(split) != set(EXPECTED_COUNTS):
        raise ValueError("Unexpected source split names")
    records, names, seen = [], {}, set()
    split_ids, counts, train_by_label = {}, {}, defaultdict(list)
    for split_name, expected_count in EXPECTED_COUNTS.items():
        entries = split[split_name]
        if len(entries) != expected_count:
            raise ValueError(f"Unexpected {split_name} count: {len(entries)}")
        split_ids[split_name] = set()
        counts[split_name] = Counter()
        for entry in entries:
            if not isinstance(entry, list) or len(entry) != 3:
                raise ValueError(f"Malformed Zhou entry: {entry!r}")
            filename, label, classname = entry
            if not isinstance(filename, str) or not re.fullmatch(r"image_\d{5}\.jpg", filename):
                raise ValueError(f"Unexpected image name: {filename!r}")
            if type(label) is not int or label not in range(102):
                raise ValueError(f"Invalid class label: {label!r}")
            if not isinstance(classname, str) or class_names[str(label + 1)] != classname:
                raise ValueError(f"Class name differs from official CoOp names: {entry!r}")
            if label in names and names[label] != classname:
                raise ValueError(f"Inconsistent class name for label {label}")
            names[label] = classname
            sample_id = f"jpg/{filename}"
            if sample_id in seen or sample_id not in image_info:
                raise ValueError(f"Duplicate, overlapping, or missing image ID: {sample_id}")
            seen.add(sample_id)
            split_ids[split_name].add(sample_id)
            counts[split_name][label] += 1
            records.append({
                "sample_id": sample_id,
                "path": str((raw / "jpg" / filename).resolve()),
                "split": split_name,
                "label": label,
                "classname": classname,
            })
            if split_name == "train":
                train_by_label[label].append(sample_id)
        if set(counts[split_name]) != set(range(102)):
            raise ValueError(f"{split_name} does not contain all 102 classes")
    if seen != set(image_info):
        raise ValueError("Zhou split image set differs from original image archive")
    min_train = min(map(len, train_by_label.values()))
    if min_train < max(SHOTS):
        raise ValueError(f"Insufficient train images for 10-shot: minimum is {min_train}")
    classes = [
        {"label": label, "classname": names[label], "global_id": f"flowers:{label}"}
        for label in range(102)
    ]
    supports = {"seeds": SEEDS, "shots": SHOTS, "indices": {}}
    sample_labels = {row["sample_id"]: row["label"] for row in records}
    for seed in SEEDS:
        rng = random.Random(seed)
        draws = {label: rng.sample(items, max(SHOTS)) for label, items in train_by_label.items()}
        support_for_seed = {}
        for shot in SHOTS:
            ids = [sample_id for draw in draws.values() for sample_id in draw[:shot]]
            support_for_seed[str(shot)] = ids
            per_class = Counter(sample_labels[sample_id] for sample_id in ids)
            if len(ids) != 102 * shot or len(set(ids)) != len(ids):
                raise ValueError(f"Invalid support size/duplicates: seed {seed}, shot {shot}")
            if per_class != {label: shot for label in range(102)}:
                raise ValueError(f"Invalid per-class support counts: seed {seed}, shot {shot}")
            if not set(ids) <= split_ids["train"]:
                raise ValueError("Support contains a validation or test sample")
        if not set(support_for_seed["1"]) <= set(support_for_seed["5"]) <= set(support_for_seed["10"]):
            raise ValueError("Support nesting check failed")
        supports["indices"][str(seed)] = support_for_seed
    checks = {
        "total_images": len(image_info),
        "num_classes": len(classes),
        "split_counts": {key: len(value) for key, value in split_ids.items()},
        "per_class_counts": {key: {str(label): count[label] for label in range(102)} for key, count in counts.items()},
        "minimum_train_images_per_class": min_train,
        "maximum_train_images_per_class": max(map(len, train_by_label.values())),
        "train_class_encounter_order": list(train_by_label),
        "split_intersections": {"train_val": 0, "train_test": 0, "val_test": 0},
        "all_images_exist_and_match_archive_bytes": True,
        "image_ids_equal_archive_ids": True,
        "all_splits_contain_102_classes": True,
        "support_train_only": True,
        "support_is_unique_and_exactly_k_per_class": True,
        "support_is_nested_per_class": True,
        "support_counts_per_seed": {str(shot): shot * 102 for shot in SHOTS},
        "class_names_match_official_coop_cat_to_name": True,
    }
    return records, classes, supports, checks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--verify-only", action="store_true", help="Read-only full source, image, and manifest verification")
    args = parser.parse_args()
    data_root = args.data_root.expanduser().resolve()
    raw = data_root / "raw" / "oxford_flowers"
    manifest_dir = data_root / "manifests" / "flowers102"
    sources = fetch_sources(raw, args.verify_only)
    image_info = extract_and_verify_images(raw, args.verify_only)
    records, classes, supports, checks = create_manifests(raw, image_info)
    manifests = {"records.json": records, "classes.json": classes, "support_indices.json": supports}
    if args.verify_only:
        for name, expected in manifests.items():
            actual = json.loads((manifest_dir / name).read_text(encoding="utf-8"))
            if actual != expected:
                raise ValueError(f"Manifest does not match deterministic regeneration: {name}")
        provenance = json.loads((manifest_dir / "provenance.json").read_text(encoding="utf-8"))
        if provenance["validation"] != checks or provenance["sources"] != sources:
            raise ValueError("Provenance source/check information does not match current data")
        for name in manifests:
            if provenance["manifest_sha256"][name] != sha256(manifest_dir / name):
                raise ValueError(f"Manifest checksum mismatch: {name}")
        if provenance["image_inventory_sha256"] != sha256(manifest_dir / "image_inventory.json"):
            raise ValueError("Image inventory checksum mismatch")
        if json.loads((manifest_dir / "image_inventory.json").read_text(encoding="utf-8")) != image_info:
            raise ValueError("Image inventory differs from verified archive content")
        print("PASS: all manifests reproduce exactly; source/image SHA256 and split/support checks pass", flush=True)
    else:
        for name, value in manifests.items():
            write_json(manifest_dir / name, value)
        write_json(manifest_dir / "image_inventory.json", image_info)
        write_json(manifest_dir / "provenance.json", {
            "dataset": "Oxford Flowers102",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "python_version": platform.python_version(),
            "source_discovery": {
                "official_oxford_page": "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/",
                "official_coop_instructions": COOP_DATASETS_URL,
                "official_coop_split_link": DRIVE_SPLIT_URL,
                "coop_commit": COOP_COMMIT,
                "hycal_commit": HYCal_COMMIT,
                "split_mirror_used": False,
                "hash_note": "SHA256 values were computed from downloaded official bytes and pinned by this preparation script; they are not publisher-supplied checksums.",
            },
            "sources": sources,
            "sampling": {
                "seeds": SEEDS,
                "shots": SHOTS,
                "rng": "One isolated Python random.Random(seed) instance per seed",
                "class_order": "First encounter order in the original Zhou train array, matching HyCal split_dataset_by_label and tracker.items()",
                "within_class_order": "Original Zhou train array order, without sorting or shuffling",
                "algorithm": "Traverse classes once, rng.sample(train_items, 10); concatenate each class's first K sampled IDs for K in 1,5,10",
                "official_equivalence": "10-shot matches random.seed(seed) followed by the official 10-shot sampler when no earlier RNG draws intervene; nested 1/5-shot intentionally differ from separate official K-shot draws",
                "support_source": "train only",
                "validation_use": "Full 1,633-image Zhou validation split retained; no CoOp few-shot val subsampling",
            },
            "validation": checks,
            "manifest_sha256": {name: sha256(manifest_dir / name) for name in manifests},
            "image_inventory_sha256": sha256(manifest_dir / "image_inventory.json"),
        })
        print(f"Wrote manifests to {manifest_dir}", flush=True)
    print(json.dumps({key: checks[key] for key in ("total_images", "num_classes", "split_counts", "minimum_train_images_per_class", "support_counts_per_seed")}, indent=2))


if __name__ == "__main__":
    main()
