"""Frozen C shrinkage experiment on Flowers102 or OrganAMNIST-224.

Place the original 102flowers.tgz, organamnist_224.npz and OpenAI ViT-B-16.pt
under artifacts/paired_inputs. Flowers uses A's records and support_indices.
python partC/run_paired_transfer.py flowers102
python partC/run_paired_transfer.py organamnist
"""
import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import statistics
import sys
import tarfile

import numpy as np
from PIL import Image
import torch

import common as C
from hycal_cached import HyCalCached
from make_support_indices import sample_support

INPUT = Path(C.ROOT) / "artifacts/paired_inputs"
RESULT = Path(C.RESULT_DIR) / "paired_transfer"
SEEDS = (0, 1, 42)
SHOTS = (1, 5, 10)
WEIGHTS_SHA = "5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f"


def checked_file(name, algorithm, expected):
    path = INPUT / name
    with path.open("rb") as stream:
        actual = hashlib.file_digest(stream, algorithm).hexdigest()
    if actual != expected:
        raise ValueError(f"{name}: {algorithm} {actual} differs from {expected}")
    return path


def flowers_data():
    folder = Path(C.ROOT) / "data/manifests/flowers102"
    records = json.loads((folder / "records.json").read_text(encoding="utf-8"))
    labels = {r["sample_id"]: r["label"] for r in records}
    train = {r["sample_id"] for r in records if r["split"] == "train"}
    test = [r["sample_id"] for r in records if r["split"] == "test"]
    indices = json.loads((folder / "support_indices.json").read_text(encoding="utf-8"))["indices"]
    draws = {(k, s): indices[str(s)][str(k)] for k in SHOTS for s in SEEDS}
    assert len(train) == 4093 and len(test) == 2463 and len(labels) == 8189
    classes = sorted(json.loads((folder / "classes.json").read_text(encoding="utf-8")),
                     key=lambda c: c["label"])
    prompts = [C.TEMPLATE.format(c["classname"].replace("_", " ")) for c in classes]
    file = checked_file("102flowers.tgz", "sha256",
                        "2d01ecc807db462958cfe3d92f57a8c252b4abd240eb955770201e45f783b246")
    needed = set(test).union(*draws.values())
    images = {}
    with tarfile.open(file, "r:gz") as archive:
        for member in archive:
            if member.name in needed and member.isfile():
                images[member.name] = archive.extractfile(member).read()
    assert images.keys() == needed

    def image_at(key):
        with Image.open(io.BytesIO(images[key])) as im:
            return im.convert("RGB")

    return labels, train, test, draws, prompts, image_at


def organ_data():
    file = checked_file("organamnist_224.npz", "md5", "50747347e05c87dd3aaf92c49f9f3170")
    names = ["bladder", "left femur", "right femur", "heart", "left kidney",
             "right kidney", "liver", "left lung", "right lung", "pancreas", "spleen"]
    prompts = [f"a photo of a CT scan of {name}." for name in names]
    with np.load(file, allow_pickle=False) as d:
        y_train = d["train_labels"].reshape(-1).tolist()
        y_test = d["test_labels"].reshape(-1).tolist()
        assert (len(y_train), len(d["val_labels"]), len(y_test)) == (34561, 6491, 17778)
        train = [f"train/{i}" for i in range(len(y_train))]
        test = [f"test/{i}" for i in range(len(y_test))]
        items = list(zip(train, y_train))
        draws = {(k, s): [train[i] for i in sample_support(items, k, s)]
                 for k in SHOTS for s in SEEDS}
        selected = sorted(set().union(*draws.values()))
        train_images = d["train_images"][[int(x.split("/")[1]) for x in selected]]
        test_images = d["test_images"]
    assert test_images.shape == (17778, 224, 224)
    images = dict(zip(selected, train_images))
    images.update(zip(test, test_images))
    labels = dict(zip(train + test, y_train + y_test))
    return labels, set(train), test, draws, prompts, lambda key: Image.fromarray(images[key]).convert("RGB")


@torch.inference_mode()
def encode(keys, image_at, prompts):
    sys.path.insert(0, C.HYCAL_REPO)
    import clip
    weights = checked_file("ViT-B-16.pt", "sha256", WEIGHTS_SHA)
    model, _, transform = clip.load(str(weights), device="cpu", jit=False)
    model = model.float().eval().cuda()
    feats = []
    for start in range(0, len(keys), 64):
        rows = keys[start:start + 64]
        batch = torch.stack([transform(image_at(key)) for key in rows]).cuda()
        f = model.encode_image(batch).float()
        feats.append((f / f.norm(dim=-1, keepdim=True)).cpu())
        if start % 1024 == 0:
            print(f"Encoded {start + len(rows)}/{len(keys)} images", flush=True)
    text = model.encode_text(clip.tokenize(prompts).cuda()).float()
    return torch.cat(feats), (text / text.norm(dim=-1, keepdim=True)).cpu()


def put_csv(path, rows):
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=["flowers102", "organamnist"])
    ds = parser.parse_args().dataset
    assert torch.cuda.is_available(), "Local CUDA encoder is required"
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    labels, train, test, draws, prompts, image_at = flowers_data() if ds == "flowers102" else organ_data()
    nclass = len(prompts)
    assert train.isdisjoint(test) and len(test) == len(set(test))
    for (k, seed), ids in draws.items():
        assert set(ids) <= train and len(set(ids)) == len(ids) == nclass * k
        assert np.array_equal(np.bincount([labels[i] for i in ids]), np.full(nclass, k))
    out = RESULT / ds
    out.mkdir(parents=True, exist_ok=True)
    keys = sorted(set().union(*draws.values())) + test
    protocol = dict(dataset=ds, prompts=prompts, alpha=[0, 0.75], fusion="sum", lambda_reg=1e-4,
                    lambda_bg=1e-4, metric="weighted", n_min_p=2, scale=1, seeds=SEEDS,
                    shots=SHOTS, clip_sha256=WEIGHTS_SHA, encoder_gpu=torch.cuda.get_device_name(0),
                    torch=str(torch.__version__), encoding="float32; evaluation transform",
                    scoring_device="cpu", supports={f"K{k}_s{s}": ids for (k,s),ids in draws.items()})
    (out / "protocol.json").write_text(json.dumps(protocol, indent=1), encoding="utf-8")
    pos = {key: i for i, key in enumerate(keys)}
    cache = INPUT / f"{ds}_paired_features.npz"
    expected_labels = np.array([labels[key] for key in keys])
    if cache.exists():
        with np.load(cache, allow_pickle=False) as d:
            assert d["weights_sha"].item() == WEIGHTS_SHA and d["prompts"].tolist() == prompts
            assert d["ids"].tolist() == keys and np.array_equal(d["labels"], expected_labels)
            features, text = torch.from_numpy(d["features"]), torch.from_numpy(d["text_features"])
    else:
        features, text = encode(keys, image_at, prompts)
        tmp = cache.with_suffix(".tmp.npz")
        np.savez(tmp, ids=np.array(keys), features=features.numpy(), text_features=text.numpy(),
                 prompts=np.array(prompts), weights_sha=WEIGHTS_SHA, labels=expected_labels)
        os.replace(tmp, cache)
    assert features.shape == (len(keys), 512) and text.shape == (nclass, 512)
    assert torch.isfinite(features).all() and torch.isfinite(text).all()
    queries = features[[pos[key] for key in test]]
    y = np.array([labels[key] for key in test])
    names = [f"{ds}:{i}" for i in range(nclass)]
    text_map = dict(zip(names, text))
    runs = []
    predrows = [{"query_id": key, "label": int(label)} for key, label in zip(test, y)]
    for (k, seed), support in draws.items():
        support_x = features[[pos[key] for key in support]]
        support_ids = [names[labels[key]] for key in support]
        predictions = []
        for alpha in (0, 0.75):
            model = HyCalCached(fusion="sum", lambda_reg=1e-4, lambda_bg=1e-4, shrink_alpha=alpha)
            model.add_session(support_x, support_ids, text_map)
            scores = model.combine(model.score(queries, batch_size=128), metric="weighted", n_min_p=2, scale=1)
            assert torch.isfinite(scores).all()
            class_labels = np.array([int(cid.split(":")[1]) for cid in model.class_ids])
            preds = class_labels[scores.argmax(dim=1).numpy()]
            predictions.append(preds)
            column = f"K{k}_s{seed}_a{alpha}"
            for row, pred in zip(predrows, preds):
                row[column] = int(pred)
        base, pooled = predictions
        if k == 1:
            assert np.array_equal(base, pooled)
        acc0, acc1 = 100 * np.mean(base == y), 100 * np.mean(pooled == y)
        runs.append(dict(dataset=ds, shots=k, seed=seed, n_test=len(y),
                         baseline_pct=acc0, pooled_pct=acc1, delta_pp=acc1-acc0))
        print(f"{ds} {k}-shot seed {seed}: {acc0:.3f} -> {acc1:.3f}", flush=True)
    put_csv(out / "runs.csv", runs)
    put_csv(out / "predictions.csv", predrows)
    summary = []
    for k in SHOTS:
        rows = [r for r in runs if r["shots"] == k]
        delta = [r["delta_pp"] for r in rows]
        summary.append(dict(dataset=ds, shots=k, baseline_pct=statistics.mean(r["baseline_pct"] for r in rows),
                            pooled_pct=statistics.mean(r["pooled_pct"] for r in rows),
                            delta_pp=statistics.mean(delta), delta_sd=statistics.stdev(delta),
                            seeds_improved=sum(v > 0 for v in delta)))
    put_csv(out / "summary.csv", summary)


if __name__ == "__main__":
    main()
