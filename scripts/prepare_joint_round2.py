"""Build CPU-only Flowers+DTD joint manifests and cached features, without scoring.

Numeric labels stay Flowers 0..101, DTD 102..148. Candidate/text row order is
separately sorted by (classname, global_id); it must not be treated as a label.
Run once to create the joint assets; --verify-only independently rebuilds and
compares their data against both frozen source caches without writing files.
"""

from __future__ import annotations

import argparse
import ast
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import time

import torch


ROOT = Path(__file__).resolve().parents[1]
NAME = "joint_flowers_dtd"
DOMAINS = ("flowers102", "dtd")
OFFSETS = {"flowers102": 0, "dtd": 102}
CLASS_COUNTS = {"flowers102": 102, "dtd": 47}
IMAGE_COUNTS = {"flowers102": 8189, "dtd": 5640}
SEEDS = [0, 1, 42, 1993, *range(2, 18)]
COUNTS = {"train": 6913, "val": 2761, "test": 4155}
TEMPLATE = "a photo of a {classname_with_underscores_replaced_by_spaces}."


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def source_payload(domain: str) -> dict:
    manifest_dir = ROOT / "data/manifests" / domain
    cache_dir = ROOT / "artifacts/features" / domain
    names = ["records.json", "classes.json", "support_indices.json", "provenance.json"]
    if domain == "flowers102":
        names += ["replication_support_indices.json", "replication_protocol.json"]
    manifests = {name: read_json(manifest_dir / name) for name in names}
    manifest_hashes = {name: sha256(manifest_dir / name) for name in names}
    for name in ("records.json", "classes.json", "support_indices.json"):
        require(manifest_hashes[name] == manifests["provenance.json"]["manifest_sha256"][name], f"{domain}/{name} has changed")
    if domain == "flowers102":
        protocol = manifests["replication_protocol.json"]
        require(manifest_hashes["replication_support_indices.json"] == protocol["replication_support_indices_sha256"], "Flowers replication support checksum mismatch")
        for name, digest in protocol["frozen_manifest_sha256"].items():
            require(manifest_hashes[name] == digest, f"Flowers frozen {name} checksum mismatch")

    cache = torch.load(cache_dir / "features.pt", map_location="cpu", weights_only=False)
    metadata = read_json(cache_dir / "metadata.json")
    require(cache["metadata"] == metadata, f"{domain} cache metadata differs from sidecar")
    require(metadata["manifest_sha256"] == manifest_hashes["records.json"], f"{domain} cache/records mismatch")
    for name, digest in metadata.get("manifest_hashes", {}).items():
        require(manifest_hashes[name] == digest, f"{domain} cache/{name} mismatch")
    cache_hash = sha256(cache_dir / "features.pt")
    if (cache_dir / "sha256.json").exists():
        for name, digest in read_json(cache_dir / "sha256.json").items():
            require(sha256(cache_dir / name) == digest, f"{domain} source cache checksum mismatch: {name}")

    records, classes = manifests["records.json"], manifests["classes.json"]
    require(len(records) == IMAGE_COUNTS[domain], f"{domain} image count mismatch")
    require(len(classes) == CLASS_COUNTS[domain], f"{domain} class count mismatch")
    require(cache["sample_ids"] == [row["sample_id"] for row in records], f"{domain} image row order mismatch")
    require(cache["labels"].tolist() == [row["label"] for row in records], f"{domain} labels mismatch")
    require(cache["splits"] == [row["split"] for row in records], f"{domain} splits mismatch")
    require({row["label"]: row for row in cache["classes"]} == {row["label"]: row for row in classes}, f"{domain} class mapping mismatch")
    require(len({row["sample_id"] for row in records}) == len(records), f"{domain} duplicate sample IDs")
    require({row["label"] for row in classes} == set(range(CLASS_COUNTS[domain])), f"{domain} label range mismatch")
    for key, expected_shape in (("image_features", (len(records), 512)), ("text_features", (len(classes), 512))):
        values = cache[key]
        require(tuple(values.shape) == expected_shape, f"{domain} {key} shape mismatch")
        require(values.dtype == torch.float32 and values.device.type == "cpu", f"{domain} {key} must be CPU float32")
        require(bool(torch.isfinite(values).all()), f"{domain} {key} contains nonfinite values")

    support_sources = [manifests["support_indices.json"]]
    if domain == "flowers102":
        support_sources.append(manifests["replication_support_indices.json"])
    supports = {}
    rows_by_id = {row["sample_id"]: row for row in records}
    for source in support_sources:
        for seed in source["seeds"]:
            require(seed not in supports, f"{domain} repeated support seed {seed}")
            ids = source["indices"][str(seed)]["5"]
            require(len(ids) == 5 * CLASS_COUNTS[domain] == len(set(ids)), f"{domain} seed {seed} support count/uniqueness mismatch")
            require(all(rows_by_id[sample_id]["split"] == "train" for sample_id in ids), f"{domain} support includes held-out samples")
            require(Counter(rows_by_id[sample_id]["label"] for sample_id in ids) == {label: 5 for label in range(CLASS_COUNTS[domain])}, f"{domain} seed {seed} per-class support mismatch")
            supports[seed] = ids
    require(set(supports) == set(SEEDS), f"{domain} does not contain the 20 required seeds")
    evidence = {
        "feature_cache_path": str((cache_dir / "features.pt").resolve()),
        "feature_cache_sha256": cache_hash,
        "metadata_path": str((cache_dir / "metadata.json").resolve()),
        "metadata_sha256": sha256(cache_dir / "metadata.json"),
        "manifest_directory": str(manifest_dir.resolve()),
        "manifest_sha256": manifest_hashes,
    }
    return {"cache": cache, "records": records, "classes": classes, "supports": supports, "evidence": evidence}


def prompts_from_source(path: Path, classes: list) -> list[str]:
    """Inspect only the trusted, hash-checked text-list expression, without imports."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    matches = [node.value for node in ast.walk(tree) if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "texts" for target in node.targets)]
    require(len(matches) == 1 and isinstance(matches[0], ast.ListComp), f"Cannot identify text template in {path}")
    expression = ast.fix_missing_locations(ast.Expression(body=matches[0]))
    return eval(compile(expression, str(path), "eval"), {"__builtins__": {}, "classes": classes})


def compatible_encoders(sources: dict) -> dict:
    flower = sources["flowers102"]["cache"]["metadata"]
    dtd = sources["dtd"]["cache"]["metadata"]
    for name in ("model", "upstream_commit", "weight_sha256", "dtype", "torch"):
        require(flower[name] == dtd[name], f"Source encoder mismatch: {name}")
    clean = lambda value: re.sub(r"0x[0-9A-Fa-f]+", "<address>", value)
    require(clean(flower["preprocess"]) == clean(dtd["preprocess"]), "Source image preprocessing differs")
    flower_script = ROOT / "scripts/cache_features.py"
    dtd_script = ROOT / "scripts/cache_dataset_round2.py"
    require(sha256(flower_script) == dtd["reference_cache_script_sha256"], "Original cache script changed after DTD encoding")
    require(sha256(dtd_script) == dtd["cache_script_sha256"], "DTD cache script changed after encoding")
    require(dtd["text_template"] == TEMPLATE, "Unexpected DTD text template")
    all_classes = sources["flowers102"]["classes"] + sources["dtd"]["classes"]
    expected_prompts = [f"a photo of a {row['classname'].replace('_', ' ')}." for row in all_classes]
    require(prompts_from_source(flower_script, all_classes) == expected_prompts, "Flowers text template differs")
    require(prompts_from_source(dtd_script, all_classes) == expected_prompts, "DTD text template differs")
    return {
        **{name: flower[name] for name in ("model", "upstream_commit", "weight_sha256", "dtype", "torch")},
        "preprocess": clean(flower["preprocess"]), "text_template": TEMPLATE,
        "same_model_weights_dtype_torch": True,
        "same_preprocess_ignoring_function_memory_addresses": True,
        "same_text_template_verified_from_hash_checked_source_expressions": True,
        "cache_script_sha256": {"flowers102": sha256(flower_script), "dtd": sha256(dtd_script)},
    }


def build_joint(sources: dict) -> tuple[dict, dict, dict]:
    records, unsorted_classes, text_sources = [], [], {}
    for domain in DOMAINS:
        source, offset = sources[domain], OFFSETS[domain]
        class_by_label = {item["label"]: item for item in source["classes"]}
        for row in source["records"]:
            records.append({
                **row, "sample_id": f"{domain}/{row['sample_id']}", "label": offset + row["label"],
                "domain": domain, "source_sample_id": row["sample_id"], "source_label": row["label"],
                "global_id": class_by_label[row["label"]]["global_id"],
            })
        for row_index, item in enumerate(source["cache"]["classes"]):
            global_id = item["global_id"]
            require(global_id not in text_sources, f"Global class ID collision: {global_id}")
            text_sources[global_id] = {"domain": domain, "source_text_row": row_index, "source_label": item["label"]}
            unsorted_classes.append({**item, "label": offset + item["label"], "domain": domain, "source_label": item["label"]})
    classes = sorted(unsorted_classes, key=lambda item: (item["classname"], item["global_id"]))
    supports = {"seeds": SEEDS, "shots": [5], "indices": {}}
    for seed in SEEDS:
        supports["indices"][str(seed)] = {"5": [f"{domain}/{sample_id}" for domain in DOMAINS for sample_id in sources[domain]["supports"][seed]]}
    image_features = torch.cat([sources[domain]["cache"]["image_features"] for domain in DOMAINS], dim=0)
    text_features = torch.stack([sources[text_sources[item["global_id"]]["domain"]]["cache"]["text_features"][text_sources[item["global_id"]]["source_text_row"]] for item in classes])
    payload = {
        "image_features": image_features, "text_features": text_features,
        "sample_ids": [row["sample_id"] for row in records],
        "labels": torch.tensor([row["label"] for row in records]),
        "splits": [row["split"] for row in records], "classes": classes,
        "domains": [row["domain"] for row in records],
    }
    outputs = {"records.json": records, "classes.json": classes, "support_indices.json": supports}
    checks = verify_joint(outputs, payload, sources, text_sources)
    text_row_mapping = [{"joint_text_row": index, "label": item["label"], "global_id": item["global_id"], **text_sources[item["global_id"]]} for index, item in enumerate(classes)]
    return outputs, payload, {"validation": checks, "text_row_mapping": text_row_mapping}


def verify_joint(outputs: dict, payload: dict, sources: dict, text_sources: dict) -> dict:
    records, classes, supports = (outputs[name] for name in ("records.json", "classes.json", "support_indices.json"))
    require(len(records) == 13829 and len(classes) == 149, "Joint image/class counts mismatch")
    require(Counter(row["split"] for row in records) == COUNTS, "Joint split counts mismatch")
    require(len({row["sample_id"] for row in records}) == len(records), "Joint sample ID collision")
    require({row["label"] for row in classes} == set(range(149)), "Joint numeric class labels mismatch")
    require(len({row["global_id"] for row in classes}) == 149, "Joint global class ID collision")
    require(classes == sorted(classes, key=lambda row: (row["classname"], row["global_id"])), "Joint candidate order mismatch")
    split_ids = {name: {row["sample_id"] for row in records if row["split"] == name} for name in COUNTS}
    intersections = {"train_val": len(split_ids["train"] & split_ids["val"]), "train_test": len(split_ids["train"] & split_ids["test"]), "val_test": len(split_ids["val"] & split_ids["test"])}
    require(not any(intersections.values()), "Joint split overlap")
    by_id = {row["sample_id"]: row for row in records}
    for seed in SEEDS:
        ids = supports["indices"][str(seed)]["5"]
        require(len(ids) == len(set(ids)) == 745, f"Joint seed {seed} support count/uniqueness mismatch")
        require(set(ids) <= split_ids["train"], f"Joint seed {seed} support includes held-out samples")
        require(Counter(by_id[item]["label"] for item in ids) == {label: 5 for label in range(149)}, f"Joint seed {seed} support per-class mismatch")
    position = 0
    for domain in DOMAINS:
        source = sources[domain]
        end = position + len(source["records"])
        require(torch.equal(payload["image_features"][position:end], source["cache"]["image_features"]), f"{domain} image features are not exactly preserved")
        for original, joined in zip(source["records"], records[position:end], strict=True):
            require(joined["domain"] == domain and joined["source_sample_id"] == original["sample_id"], f"{domain} source identity mismatch")
            require(joined["sample_id"] == f"{domain}/{original['sample_id']}" and joined["path"] == original["path"], f"{domain} sample path/namespace mismatch")
            require(joined["label"] == OFFSETS[domain] + original["label"] and joined["source_label"] == original["label"], f"{domain} label offset mismatch")
            require(joined["split"] == original["split"] and joined["classname"] == original["classname"], f"{domain} source metadata mismatch")
        position = end
    for index, item in enumerate(classes):
        source = text_sources[item["global_id"]]
        require(item["label"] == OFFSETS[item["domain"]] + item["source_label"], "Class domain/label mapping mismatch")
        expected_prefix = "flowers:" if item["domain"] == "flowers102" else "dtd:"
        require(item["global_id"] == expected_prefix + str(item["source_label"]), "Class global namespace mismatch")
        require(torch.equal(payload["text_features"][index], sources[source["domain"]]["cache"]["text_features"][source["source_text_row"]]), "Joint text row does not exactly match its source")
    return {
        "total_images": len(records), "num_classes": len(classes), "split_counts": dict(COUNTS),
        "split_intersections": intersections, "num_seeds": len(SEEDS), "support_per_seed": 745,
        "domain_image_counts": dict(IMAGE_COUNTS), "domain_class_counts": dict(CLASS_COUNTS),
        "numeric_label_offsets": dict(OFFSETS),
        "all_sample_ids_unique": True, "global_ids_and_numeric_labels_bijective": True,
        "support_train_only_unique_and_five_per_class": True,
        "source_paths_splits_and_class_names_preserved_exactly": True,
        "image_features_exactly_equal_to_source_concatenation": True,
        "text_features_exactly_equal_to_correctly_reordered_source_rows": True,
        "candidate_order": "(classname, global_id)",
        "candidate_text_row_index_is_not_numeric_label": True,
        "gpu_used": False, "classification_scores_computed": False,
    }


def verify_sources_unchanged(sources: dict) -> None:
    for source in sources.values():
        evidence = source["evidence"]
        require(sha256(Path(evidence["feature_cache_path"])) == evidence["feature_cache_sha256"], "Source cache changed during merge")
        require(sha256(Path(evidence["metadata_path"])) == evidence["metadata_sha256"], "Source metadata changed during merge")
        for name, digest in evidence["manifest_sha256"].items():
            require(sha256(Path(evidence["manifest_directory"]) / name) == digest, f"Source manifest changed during merge: {name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(8)
    started = time.perf_counter()
    manifest_dir = ROOT / "data/manifests" / NAME
    cache_dir = ROOT / "artifacts/features" / NAME
    target = cache_dir / "features.pt"
    if not args.verify_only:
        require(not manifest_dir.exists() and not cache_dir.exists(), "Joint outputs already exist; use --verify-only instead of overwriting frozen assets")
    sources = {domain: source_payload(domain) for domain in DOMAINS}
    compatibility = compatible_encoders(sources)
    outputs, payload, details = build_joint(sources)
    source_evidence = {domain: source["evidence"] for domain, source in sources.items()}
    if args.verify_only:
        for name, expected in outputs.items():
            require(read_json(manifest_dir / name) == expected, f"Saved joint {name} differs from sources")
        saved = torch.load(target, map_location="cpu", weights_only=False)
        for name, expected in payload.items():
            actual = saved[name]
            require(torch.equal(actual, expected) if isinstance(expected, torch.Tensor) else actual == expected, f"Saved joint payload differs: {name}")
        metadata = read_json(cache_dir / "metadata.json")
        provenance = read_json(manifest_dir / "provenance.json")
        require(saved["metadata"] == metadata, "Saved metadata sidecar mismatch")
        require(provenance["sources"] == metadata["sources"] == source_evidence, "Saved source hashes differ")
        require(provenance["validation"] == details["validation"] and provenance["encoder_compatibility"] == compatibility, "Saved validation/compatibility differs")
        require(metadata["text_row_mapping"] == details["text_row_mapping"], "Saved text row mapping differs")
        for name, digest in provenance["manifest_sha256"].items():
            require(sha256(manifest_dir / name) == digest, f"Saved joint manifest checksum mismatch: {name}")
        for relative, digest in read_json(cache_dir / "sha256.json").items():
            require(sha256(ROOT / relative) == digest, f"Saved joint output checksum mismatch: {relative}")
    else:
        manifest_dir.mkdir(parents=True)
        cache_dir.mkdir(parents=True)
        for name, value in outputs.items():
            write_json(manifest_dir / name, value)
        manifest_hashes = {name: sha256(manifest_dir / name) for name in outputs}
        stages = [
            {"stage": 1, "arrived_domains": ["flowers102"], "candidate_labels": [item["label"] for item in payload["classes"] if item["domain"] == "flowers102"]},
            {"stage": 2, "arrived_domains": list(DOMAINS), "candidate_labels": [item["label"] for item in payload["classes"]]},
        ]
        created = datetime.now(timezone.utc).isoformat()
        provenance = {
            "dataset": NAME, "created_at_utc": created,
            "sources": source_evidence, "encoder_compatibility": compatibility,
            "validation": details["validation"], "manifest_sha256": manifest_hashes,
            "sampling": "For each of the 20 seeds concatenate the unchanged Flowers 5-shot support with the unchanged DTD 5-shot support, adding only sample-ID domain prefixes",
            "arrival_order": list(DOMAINS), "stages": stages,
            "stage_support_rule": "At each stage use only support from arrived_domains; stage 1 must not consume future DTD support or test distribution",
            "classification_scores_computed": False,
        }
        write_json(manifest_dir / "provenance.json", provenance)
        metadata = {
            **{name: compatibility[name] for name in ("model", "upstream_commit", "weight_sha256", "preprocess", "dtype", "torch", "text_template")},
            "dataset": NAME, "device": "cpu", "created_at_utc": created,
            "manifest_sha256": manifest_hashes["records.json"],
            "manifest_hashes": {**manifest_hashes, "provenance.json": sha256(manifest_dir / "provenance.json")},
            "sources": source_evidence, "encoder_compatibility": compatibility,
            "seconds": time.perf_counter() - started, "cuda_peak_allocated_bytes": 0,
            "operation": "Exact image tensor concatenation and exact text tensor row permutation; no feature extraction or renormalization",
            "candidate_labels": [item["label"] for item in payload["classes"]],
            "text_row_mapping": details["text_row_mapping"],
            "arrival_order": list(DOMAINS), "stages": stages,
            "classification_scores_computed": False,
        }
        payload["metadata"] = metadata
        temporary = target.with_suffix(".tmp")
        torch.save(payload, temporary)
        os.replace(temporary, target)
        write_json(cache_dir / "metadata.json", metadata)
        write_json(cache_dir / "validation.json", {"created_at_utc": created, **details["validation"]})
        files = [target, cache_dir / "metadata.json", cache_dir / "validation.json", *(manifest_dir / name for name in (*outputs, "provenance.json"))]
        write_json(cache_dir / "sha256.json", {path.relative_to(ROOT).as_posix(): sha256(path) for path in files})
    verify_sources_unchanged(sources)
    print("PASS: 149 classes, 13,829 images, 20 x 745 support, disjoint splits and exact source features; source hashes unchanged", flush=True)
    print(json.dumps({"cache": str(target), "cache_sha256": sha256(target), "validation": details["validation"], "seconds": time.perf_counter() - started}, indent=2))


if __name__ == "__main__":
    main()
