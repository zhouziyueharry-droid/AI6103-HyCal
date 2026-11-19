"""Prepare Oxford FGVC-Aircraft and the official CoOp split; no features/scoring.

The 100 variants and official train/val/test files are used directly. Aircraft
has no Zhou JSON in CoOp. Supports follow the HyCal sample(10)[:5] convention.
Run normally to prepare, or --verify-only to reproduce and audit without writes.
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
import os
from pathlib import Path, PurePosixPath
import platform
import random
import re
import tarfile
import time
from types import SimpleNamespace
import urllib.request

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SEEDS = [0, 1, 42, 1993, *range(2, 18)]
COUNTS = {"train": 3334, "val": 3333, "test": 3333}
COOP_COMMIT = "ff61507c790454bce7c5052c3ac39e60772f1f89"
HYCAL_COMMIT = "4558fb7fa6aa389310c9e9dc7db7c2cbebc84a41"
R_TORCHVISION_COMMIT = "708d14734a1a39e42d9b03b6c87d90197bb1a0d3"
ARCHIVE_SHA256 = "e4e323d410e29f0370c81eabdcbb0e2b813acea1de22891b70b58ff41bfc9834"
CHECKSUM_CORROBORATION = {
    "source": f"https://github.com/mlverse/torchvision/blob/{R_TORCHVISION_COMMIT}/R/dataset-fgvc.R",
    "recorded_archive_md5": "d4acdd33327262359767eeaa97a4f732",
    "role": "Independent library-author source corroborates the current Oxford archive URL and MD5; it is not an Oxford publisher checksum or a data/split source.",
    "python_torchvision_reference": "https://docs.pytorch.org/vision/stable/_modules/torchvision/datasets/fgvc_aircraft.html",
    "python_torchvision_note": "Official Python torchvision source documents 100 variants/10000 images and the same Oxford URL, but does not supply an MD5.",
}
SOURCES = {
    "archive": {
        "path": "fgvc-aircraft-2013b.tar.gz",
        "url": "https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/archives/fgvc-aircraft-2013b.tar.gz",
        "sha256": ARCHIVE_SHA256,
        "publisher_md5_page_value": "32eca553f897e747706144f3bb444df0",
    },
    "annotations_archive": {
        "path": "fgvc-aircraft-2013b-annotations.tar.gz",
        "url": "https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/archives/fgvc-aircraft-2013b-annotations.tar.gz",
        "sha256": "50e4419bd330b68a2e85dbcb32339d31120190efe6944682e87c9b431f5f202a",
        "publisher_md5_page_value": "e3799900175f63abbc48a5bc5b16427e",
    },
    "coop_loader": {
        "path": "sources/CoOp_fgvc_aircraft.py",
        "url": f"https://raw.githubusercontent.com/KaiyangZhou/CoOp/{COOP_COMMIT}/datasets/fgvc_aircraft.py",
        "sha256": "834fcdac5e903be76da6c38d215c2537b3270c39ec7f8cf5765145d1325efcb3",
    },
    "coop_instructions": {
        "path": "sources/CoOp_DATASETS.md",
        "url": f"https://raw.githubusercontent.com/KaiyangZhou/CoOp/{COOP_COMMIT}/DATASETS.md",
        "sha256": "3de6caf6c48fbd10c814734b6561e348863aa9e36462cd83266e80c17dbafba3",
    },
    "hycal_sampler": {
        "path": "sources/HyCal_dataset_utils.py",
        "url": f"https://raw.githubusercontent.com/EJLEE5826/HyCal-CIL/{HYCAL_COMMIT}/task_datasets/utils.py",
        "sha256": "117f6479b57006c23717a5f4aa90aee3f9f8155bc4b4baed56e6033f5be494d9",
    },
    "oxford_archive_md5_page": {
        "path": "sources/Oxford_archive_md5.html",
        "url": "https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/archives/fgvc-aircraft-2013b.html",
        "sha256": "648573f405810bcc8679cf74a2d736f079010be408b4de1df00df3b11505c690",
    },
    "oxford_annotations_md5_page": {
        "path": "sources/Oxford_annotations_md5.html",
        "url": "https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/archives/fgvc-aircraft-2013b-annotations.html",
        "sha256": "e2e2f07a21a4d63f452bd0b2e267d7b4c590d6888822f4b65b95510b4c1cbe74",
    },
    "r_torchvision_checksum_corroboration": {
        "path": "sources/R_torchvision_dataset-fgvc.R",
        "url": f"https://raw.githubusercontent.com/mlverse/torchvision/{R_TORCHVISION_COMMIT}/R/dataset-fgvc.R",
        "sha256": "e8439f85dbc71fd0da692cc580868ed063cfbde1486801757162a7f315f6fcb7",
        "role": "Checksum corroboration only; no images, labels, or split lists are taken from this source.",
    },
}


def hashes(path):
    digests = {name: hashlib.new(name) for name in ("sha256", "md5")}
    with Path(path).open("rb") as stream:
        while chunk := stream.read(4 * 1024 * 1024):
            for digest in digests.values():
                digest.update(chunk)
    return {name: digest.hexdigest() for name, digest in digests.items()}


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sources(raw, verify_only):
    require(re.fullmatch("[0-9a-f]{64}", ARCHIVE_SHA256), "Official archive SHA256 not yet pinned")
    evidence = {}
    deadline = time.monotonic() + 900
    for name, entry in SOURCES.items():
        path = raw / entry["path"]
        if not path.exists():
            if verify_only:
                raise FileNotFoundError(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(path.suffix + ".part")
            print(f"Downloading official source: {entry['url']}", flush=True)
            with urllib.request.urlopen(entry["url"], timeout=45) as response, temporary.open("wb") as output:
                while chunk := response.read(1024 * 1024):
                    if time.monotonic() > deadline:
                        raise TimeoutError("15-minute official download budget exceeded")
                    output.write(chunk)
            require(sha(temporary) == entry["sha256"], f"Downloaded SHA256 mismatch: {path}")
            temporary.replace(path)
        digest = hashes(path)
        require(digest["sha256"] == entry["sha256"], f"Pinned official-byte SHA256 mismatch: {path}")
        evidence[name] = {**entry, "absolute_path": str(path.resolve()), "bytes": path.stat().st_size,
                          "actual_md5": digest["md5"], "pinned_sha256_verified": True}
        if "publisher_md5_page_value" in entry:
            evidence[name]["publisher_md5_matches_current_official_bytes"] = digest["md5"] == entry["publisher_md5_page_value"]
    return evidence


def extract_and_verify(raw, verify_only):
    root = raw.resolve()
    images, annotations = {}, {}
    started = time.monotonic()
    with tarfile.open(raw / "fgvc-aircraft-2013b.tar.gz", "r:gz") as archive:
        for member in archive:
            original = PurePosixPath(member.name)
            require(not original.is_absolute() and ".." not in original.parts and "\\" not in member.name,
                    f"Unsafe archive path: {member.name}")
            require(original.parts and original.parts[0] == "fgvc-aircraft-2013b", f"Unexpected root: {member.name}")
            target = (raw / Path(*original.parts)).resolve()
            require(target.is_relative_to(root), f"Archive path escapes root: {member.name}")
            if member.isdir():
                if not verify_only:
                    target.mkdir(parents=True, exist_ok=True)
                continue
            require(member.isfile(), f"Archive link/special file forbidden: {member.name}")
            with archive.extractfile(member) as stream:
                data = stream.read()
            require(len(data) == member.size, f"Truncated archive member: {member.name}")
            digest = hashlib.sha256(data).hexdigest()
            intact = target.is_file() and target.stat().st_size == len(data) and sha(target) == digest
            if not intact:
                require(not verify_only, f"Missing/corrupt extracted file: {target}")
                target.parent.mkdir(parents=True, exist_ok=True)
                temporary = target.with_suffix(target.suffix + ".part")
                temporary.write_bytes(data)
                temporary.replace(target)
            relative = original.relative_to("fgvc-aircraft-2013b")
            if relative.parts[:2] == ("data", "images"):
                require(len(relative.parts) == 3 and re.fullmatch(r"\d{7}\.jpg", relative.name), f"Unexpected image: {member.name}")
                require(data.startswith(b"\xff\xd8\xff"), f"Invalid JPEG header: {member.name}")
                with Image.open(io.BytesIO(data)) as image:
                    require(image.format == "JPEG", f"Not a JPEG: {member.name}")
                    size = list(image.size)
                    image.verify()
                sample_id = "images/" + relative.name
                require(sample_id not in images, f"Repeated image archive member: {sample_id}")
                images[sample_id] = {"sha256": digest, "bytes": len(data), "width": size[0], "height": size[1]}
                if len(images) % 2000 == 0:
                    print(f"Verified {len(images):,} JPEGs in {time.monotonic()-started:.1f}s", flush=True)
            elif relative.parts and relative.parts[0] == "data":
                annotations[relative.name] = digest
    data_root = raw / "fgvc-aircraft-2013b/data"
    on_disk = {"images/" + p.name for p in (data_root / "images").glob("*.jpg")}
    require(len(images) == 10000 and on_disk == set(images), "Expected exactly 10,000 official JPEGs")
    # The independently downloaded official annotations archive must agree on
    # every class/split list used by CoOp and on the image-ID split lists.
    required = {"variants.txt", *[f"images_variant_{s}.txt" for s in ("train", "val", "test", "trainval")],
                *[f"images_{s}.txt" for s in ("train", "val", "test")]}
    matched = {}
    with tarfile.open(raw / "fgvc-aircraft-2013b-annotations.tar.gz", "r:gz") as archive:
        for member in archive:
            name = PurePosixPath(member.name).name
            if member.isfile() and name in required:
                with archive.extractfile(member) as stream:
                    digest = hashlib.sha256(stream.read()).hexdigest()
                require(name in annotations and digest == annotations[name], f"Official archives disagree: {name}")
                matched[name] = digest
    require(set(matched) == required, "Missing required class/split annotations")
    return images, matched


def official_crosschecks(raw, data_root, records, supports):
    # Execute only the already inspected, pinned CoOp read_data method.
    path = raw / "sources/CoOp_fgvc_aircraft.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "FGVCAircraft")
    cls.body = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "read_data"]
    require(len(cls.body) == 1, "Cannot identify official CoOp read_data")
    cls.bases, cls.decorator_list = [], []
    namespace = {"os": os, "Datum": SimpleNamespace}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[])), str(path), "exec"), namespace)
    reader = namespace["FGVCAircraft"]()
    reader.dataset_dir, reader.image_dir = str(data_root), str(data_root / "images")
    classnames = (data_root / "variants.txt").read_text().splitlines()
    mapping = {name: i for i, name in enumerate(classnames)}
    for split in COUNTS:
        official = reader.read_data(mapping, f"images_variant_{split}.txt")
        ours = [r for r in records if r["split"] == split]
        require([(str(Path(x.impath).resolve()), x.label, x.classname) for x in official]
                == [(x["path"], x["label"], x["classname"]) for x in ours], f"CoOp loader mismatch: {split}")
    # Execute only the pinned official HyCal sampling and grouping methods.
    path = raw / "sources/HyCal_dataset_utils.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "DatasetBase")
    cls.body = [n for n in cls.body if isinstance(n, ast.FunctionDef)
                and n.name in {"generate_fewshot_dataset", "split_dataset_by_label"}]
    require(len(cls.body) == 2, "Cannot identify official HyCal sampling methods")
    cls.bases, cls.decorator_list = [], []
    namespace = {"random": random, "defaultdict": defaultdict}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[])), str(path), "exec"), namespace)
    items = [SimpleNamespace(sample_id=r["sample_id"], label=r["label"]) for r in records if r["split"] == "train"]
    previous = random.getstate()
    try:
        with redirect_stdout(io.StringIO()):
            for seed in SEEDS:
                random.seed(seed)
                drawn = namespace["DatasetBase"]().generate_fewshot_dataset(items, num_shots=10)
                prefix = [x.sample_id for start in range(0, len(drawn), 10) for x in drawn[start:start+5]]
                require(prefix == supports["indices"][str(seed)]["5"], f"HyCal sample10 prefix mismatch: {seed}")
    finally:
        random.setstate(previous)


def build(raw, images):
    data_root = raw / "fgvc-aircraft-2013b/data"
    classnames = (data_root / "variants.txt").read_text(encoding="utf-8").splitlines()
    require(len(classnames) == len(set(classnames)) == 100, "Expected 100 distinct official variants")
    require(all(n == n.strip() and n for n in classnames), "Malformed variant name")
    mapping = {name: i for i, name in enumerate(classnames)}
    classes = [{"label": label, "classname": name, "global_id": f"aircraft:{label}"} for label, name in enumerate(classnames)]
    records, train, split_ids, per_class = [], defaultdict(list), {}, {}
    for split, count in COUNTS.items():
        lines = (data_root / f"images_variant_{split}.txt").read_text(encoding="utf-8").splitlines()
        require(len(lines) == count, f"Wrong split size: {split}")
        ids = []
        for line in lines:
            parts = line.strip().split(" ")
            image_id, classname = parts[0], " ".join(parts[1:])
            require(re.fullmatch(r"\d{7}", image_id) and classname in mapping, f"Invalid official annotation: {line}")
            sid, label = f"images/{image_id}.jpg", mapping[classname]
            require(sid in images, f"Annotation image missing: {sid}")
            records.append({"sample_id": sid, "path": str((data_root / sid).resolve()),
                            "split": split, "label": label, "classname": classname})
            ids.append(sid)
            if split == "train":
                train[label].append(sid)
        split_ids[split] = set(ids)
        require(len(split_ids[split]) == count, f"Repeated sample in {split}")
        listed_ids = (data_root / f"images_{split}.txt").read_text().splitlines()
        require(ids == [f"images/{x}.jpg" for x in listed_ids], f"Variant and image-ID list order differs: {split}")
        per_class[split] = dict(Counter(r["label"] for r in records if r["split"] == split))
        require(set(per_class[split]) == set(range(100)) and set(per_class[split].values()) <= {33, 34},
                f"Unexpected per-class official counts: {split}")
    intersections = {a+"_"+b: len(split_ids[a] & split_ids[b]) for a, b in [("train", "val"), ("train", "test"), ("val", "test")]}
    require(not any(intersections.values()), "Official splits overlap")
    require(set.union(*split_ids.values()) == set(images), "Annotations and image archive disagree")
    require(Counter(r["label"] for r in records) == {label: 100 for label in range(100)}, "Not 100 images per variant")
    trainval = (data_root / "images_variant_trainval.txt").read_text().splitlines()
    trainval_set = {tuple(line.split(" ", 1)) for line in trainval}
    expected_trainval = {(Path(r["sample_id"]).stem, r["classname"]) for r in records if r["split"] in {"train", "val"}}
    require(len(trainval) == 6667 and trainval_set == expected_trainval, "Official trainval is not train union val")
    supports = {"seeds": SEEDS, "shots": [5], "indices": {}}
    lookup = {r["sample_id"]: r for r in records}
    for seed in SEEDS:
        rng = random.Random(seed)
        drawn = [sid for items in train.values() for sid in rng.sample(items, 10)[:5]]
        require(len(drawn) == len(set(drawn)) == 500 and set(drawn) <= split_ids["train"], f"Invalid support: {seed}")
        require(Counter(lookup[sid]["label"] for sid in drawn) == {label: 5 for label in range(100)}, f"Bad per-class support: {seed}")
        supports["indices"][str(seed)] = {"5": drawn}
    official_crosschecks(raw, data_root, records, supports)
    hash_groups = defaultdict(list)
    for sid, entry in images.items():
        hash_groups[entry["sha256"]].append(sid)
    duplicate_groups = [ids for ids in hash_groups.values() if len(ids) > 1]
    cross_split_duplicates = [ids for ids in duplicate_groups if len({lookup[sid]["split"] for sid in ids}) > 1]
    checks = {
        "total_images": len(images), "num_classes": 100, "split_counts": COUNTS,
        "per_class_counts": {split: {str(label): counts[label] for label in range(100)} for split, counts in per_class.items()},
        "split_intersections": intersections, "minimum_train_images_per_class": min(map(len, train.values())),
        "train_class_encounter_order": list(train), "support_seeds": SEEDS, "support_counts_per_seed": {"5": 500},
        "all_images_exist_and_match_archive_bytes": True, "all_jpegs_pillow_verify": True,
        "each_variant_has_100_images": True, "official_trainval_equals_train_union_val": True,
        "all_splits_contain_100_classes": True, "support_train_only": True,
        "support_exactly_five_per_class": True, "official_coop_read_data_exact_match": True,
        "official_hycal_sample10_first5_matches_all_seeds": True,
        "duplicate_image_sha256_groups": duplicate_groups,
        "cross_split_duplicate_image_sha256_groups": cross_split_duplicates,
    }
    return {"records.json": records, "classes.json": classes, "support_indices.json": supports}, checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    data = args.data_root.expanduser().resolve()
    raw, manifests = data / "raw/aircraft", data / "manifests/aircraft"
    evidence = sources(raw, args.verify_only)
    images, annotation_hashes = extract_and_verify(raw, args.verify_only)
    outputs, checks = build(raw, images)
    inventory = raw / "sources/image_inventory.json"
    if args.verify_only:
        previous = load(manifests / "provenance.json")
        require(previous["preparation_script_sha256"] == sha(Path(__file__)), "Preparation script changed")
        for name, value in outputs.items():
            require(load(manifests / name) == value, f"Manifest not reproducible: {name}")
            require(sha(manifests / name) == previous["manifest_sha256"][name], f"Manifest checksum mismatch: {name}")
        require(previous["sources"] == evidence and previous["validation"] == checks, "Provenance source/check mismatch")
        require(previous["matching_official_annotation_sha256"] == annotation_hashes, "Annotation checksum mismatch")
        require(previous["checksum_corroboration"] == CHECKSUM_CORROBORATION, "Checksum corroboration record changed")
        require(load(inventory) == images and sha(inventory) == previous["image_inventory_sha256"], "Image inventory mismatch")
        print("PASS: official Aircraft data, CoOp lists, JPEGs, and all 20 fixed supports reproduce", flush=True)
    else:
        for name, value in outputs.items():
            save(manifests / name, value)
        save(inventory, images)
        save(manifests / "provenance.json", {
            "dataset": "FGVC-Aircraft 2013b, 100 variant classes as used by CoOp",
            "created_at_utc": datetime.now(timezone.utc).isoformat(), "python_version": platform.python_version(),
            "sources": evidence, "matching_official_annotation_sha256": annotation_hashes,
            "checksum_corroboration": CHECKSUM_CORROBORATION,
            "source_discovery": {
                "official_oxford_page": "https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/",
                "coop_commit": COOP_COMMIT, "hycal_commit": HYCAL_COMMIT,
                "mirrors_used": False,
                "hash_note": "SHA256 pins describe current downloaded official bytes, not publisher-supplied SHA256 values. Publisher MD5 comparison is recorded separately and may be false for stale Oxford checksum pages.",
                "legacy_page_note": "Oxford landing page text mentions 102 variants/10200 images; current 2013b archive and CoOp lists contain 100 variants/10000 images. The latter define this protocol.",
            },
            "split_protocol": "Official Oxford images_variant_train/val/test.txt used directly by pinned CoOp FGVCAircraft.read_data; no Zhou JSON exists or is synthesized. variants.txt line order defines labels 0..99; original variant spelling is preserved.",
            "sampling": {
                "seeds": SEEDS, "shots": [5], "rng": "One isolated Python random.Random(seed) per seed",
                "class_order": "First encounter order in official images_variant_train.txt, matching HyCal tracker.items()",
                "within_class_order": "Original official train list order; no sorting or shuffling",
                "algorithm": "rng.sample(class_train_items, 10)[:5], concatenate in class encounter order",
                "official_equivalence": "First five per class from HyCal official 10-shot draw; intentionally not a separate direct 5-shot draw",
                "support_source": "train only", "cross_seed_overlap_allowed": True,
            },
            "transfer_protocol": {
                "purpose": "Independent external blind transfer endpoint using frozen Flowers parameters",
                "decision_timing": "Requested before inspecting any DTD/joint test result; not added to the joint incremental stages",
                "aircraft_hyperparameter_selection_allowed": False,
                "validation_use": "Retained only as official dataset organization; no Aircraft tuning",
                "features_or_scores_computed": False, "gpu_used": False,
                "test_label_use": "Data integrity, variant mapping, split verification only",
                "image_transform": "No crop, resize, augmentation, or encoding performed during preparation",
            },
            "validation": checks,
            "manifest_sha256": {name: sha(manifests / name) for name in outputs},
            "image_inventory_path": str(inventory.resolve()), "image_inventory_sha256": sha(inventory),
            "preparation_script_sha256": sha(Path(__file__)),
        })
        print(f"Wrote Aircraft manifests to {manifests}", flush=True)
    print(json.dumps({key: checks[key] for key in ["total_images", "num_classes", "split_counts", "split_intersections", "minimum_train_images_per_class", "support_counts_per_seed"]}, indent=2))


if __name__ == "__main__":
    main()
