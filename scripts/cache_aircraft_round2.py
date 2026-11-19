"""Cache FGVC-Aircraft CLIP features with the existing Flowers encoding protocol.

Example: python scripts/cache_aircraft_round2.py --dataset aircraft --batch-size 16
This script performs encoding only; it does not calculate classification scores.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time

import psutil
import torch
from torch.utils.data import DataLoader

# Reuse the original dataset wrapper and its pinned official CLIP import.
from cache_features import Images, ROOT, clip


WEIGHT_SHA256 = "5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f"
UPSTREAM_COMMIT = "4558fb7fa6aa389310c9e9dc7db7c2cbebc84a41"


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=["aircraft"])
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=0)
    args = parser.parse_args()
    if args.batch_size < 1 or args.workers < 0:
        parser.error("batch size must be positive and workers must be nonnegative")

    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    manifest = ROOT / "data/manifests" / args.dataset / "records.json"
    manifest_dir = manifest.parent
    records = json.loads(manifest.read_text(encoding="utf-8"))
    classes = json.loads((manifest_dir / "classes.json").read_text(encoding="utf-8"))
    classes = sorted(classes, key=lambda value: value["classname"])
    target = ROOT / "artifacts/features" / args.dataset / "features.pt"
    if target.exists():
        raise RuntimeError(f"Cache already exists: {target}; verify provenance before reusing")
    target.parent.mkdir(parents=True, exist_ok=True)

    manifest_hashes = {name: sha256(manifest_dir / name) for name in ("records.json", "classes.json", "support_indices.json", "provenance.json")}
    provenance = json.loads((manifest_dir / "provenance.json").read_text(encoding="utf-8"))
    for name in ("records.json", "classes.json", "support_indices.json"):
        if manifest_hashes[name] != provenance["manifest_sha256"][name]:
            raise ValueError(f"Manifest checksum does not match frozen provenance: {name}")
    if len(records) != 10000 or len(classes) != 100:
        raise ValueError("FGVC-Aircraft must contain exactly 10,000 images and 100 classes")
    if len({row["sample_id"] for row in records}) != len(records):
        raise ValueError("Duplicate sample IDs in image manifest")
    weights = ROOT / "artifacts/models/ViT-B-16.pt"
    if sha256(weights) != WEIGHT_SHA256:
        raise ValueError("CLIP model weight checksum differs from the original experiment")

    model, _, transform = clip.load(str(weights), device=device, jit=False)
    if next(model.parameters()).dtype != torch.float32:
        raise ValueError("Official CLIP must run in float32, as in the Flowers cache")
    model.eval().requires_grad_(False)
    loader = DataLoader(
        Images(records, transform), batch_size=args.batch_size,
        shuffle=False, num_workers=args.workers, pin_memory=device == "cuda",
    )
    if device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    memory_start = psutil.Process().memory_info().rss
    features = []
    with torch.inference_mode():
        # Preserve the original generic template, including its exact punctuation.
        texts = [f"a photo of a {item['classname'].replace('_', ' ')}." for item in classes]
        tokens = clip.tokenize(texts).to(device)
        text_features = model.encode_text(tokens)
        text_features = text_features / (text_features.norm(dim=-1, keepdim=True) + 1e-10)
        for index, batch in enumerate(loader):
            embedding = model.encode_image(batch.to(device, non_blocking=True))
            embedding = embedding / (embedding.norm(dim=-1, keepdim=True) + 1e-10)
            features.append(embedding.cpu())
            count = min((index + 1) * args.batch_size, len(records))
            if index == 0 or count % 256 == 0 or count == len(records):
                elapsed = time.perf_counter() - started
                print(f"encoded {count}/{len(records)} elapsed={elapsed:.1f}s rate={count / elapsed:.1f}/s", flush=True)
    if device == "cuda":
        torch.cuda.synchronize()
    image_features = torch.cat(features)
    text_features = text_features.cpu()
    elapsed = time.perf_counter() - started
    if image_features.shape != (len(records), 512) or text_features.shape != (len(classes), 512):
        raise ValueError("Unexpected CLIP feature dimensions")
    for name, values in (("image", image_features), ("text", text_features)):
        if values.dtype != torch.float32 or not torch.isfinite(values).all():
            raise ValueError(f"Invalid dtype or nonfinite {name} features")
        if not torch.allclose(values.norm(dim=-1), torch.ones(values.shape[0]), atol=2e-6, rtol=0):
            raise ValueError(f"{name} features are not unit normalized")
    if any(sha256(manifest_dir / name) != digest for name, digest in manifest_hashes.items()):
        raise ValueError("FGVC-Aircraft manifests changed during feature encoding")

    # Retain every original cache metadata key; extra provenance identifies FGVC-Aircraft.
    metadata = {
        "model": "ViT-B/16", "upstream_commit": UPSTREAM_COMMIT,
        "manifest_sha256": manifest_hashes["records.json"],
        "weight_sha256": WEIGHT_SHA256,
        "preprocess": str(transform), "dtype": str(image_features.dtype), "device": device,
        "torch": torch.__version__, "batch_size": args.batch_size, "seconds": elapsed,
        "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated() if device == "cuda" else 0,
        "process_rss_end_bytes": psutil.Process().memory_info().rss,
        "process_rss_start_bytes": memory_start,
        "dataset": args.dataset, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "manifest_hashes": manifest_hashes,
        "reference_cache_script_sha256": sha256(ROOT / "scripts/cache_features.py"),
        "cache_script_sha256": sha256(Path(__file__)),
        "text_template": "a photo of a {classname_with_underscores_replaced_by_spaces}.",
        "images_per_second": len(records) / elapsed,
        "classification_scores_computed": False,
    }
    payload = {
        "image_features": image_features, "text_features": text_features,
        "sample_ids": [row["sample_id"] for row in records],
        "labels": torch.tensor([row["label"] for row in records]),
        "splits": [row["split"] for row in records], "classes": classes, "metadata": metadata,
    }
    temporary = target.with_suffix(".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, target)
    metadata_path = target.parent / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    hashes = {"features.pt": sha256(target), "metadata.json": sha256(metadata_path)}
    (target.parent / "sha256.json").write_text(json.dumps(hashes, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"metadata": metadata, "cache_sha256": hashes}, indent=2), flush=True)


if __name__ == "__main__":
    main()
