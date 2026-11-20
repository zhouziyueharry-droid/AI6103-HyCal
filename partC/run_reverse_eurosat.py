"""Compare frozen member methods on EuroSAT. Inputs go in artifacts/paired_inputs."""
import argparse
import io
import json
import sys
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from PIL import Image
import torch

import common as C
from hycal_cached import HyCalCached
from run_paired_transfer import INPUT, SEEDS, SHOTS, checked_file, encode, put_csv

ROOT = Path(C.ROOT)
sys.path.insert(0, str(ROOT / "scripts"))
import run_propagation_round2 as propagation
from run_round2 import core


def inputs():
    split_file = checked_file("split_zhou_EuroSAT.json", "sha256",
                              "541503a77d0b26b735d85ec7a3e672804fe0aa466c2e5f07da772cc6c0af8ad2")
    split = json.loads(split_file.read_text())
    train, test = split["train"], split["test"]
    assert [len(split[s]) for s in ("train", "val", "test")] == [13500, 5400, 8100]
    assert len({row[0] for rows in split.values() for row in rows}) == 27000
    draws = {}
    for k in SHOTS:
        for seed in SEEDS:
            index = json.loads(Path(C.index_file("eurosat", k, seed)).read_text())
            assert [train[i][:2] for i in index["train_indices"]] == list(map(list, zip(index["paths"], index["labels"])))
            assert len(set(index["paths"])) == 10 * k
            assert np.array_equal(np.bincount(index["labels"]), np.full(10, k))
            draws[k, seed] = index
    names = {label: name for _, label, name in train}
    prompts = [C.TEMPLATE.format(names[i].replace("_", " ") + C.CLASSNAME_SUFFIX) for i in range(10)]
    support = sorted({path for index in draws.values() for path in index["paths"]})
    keys = support + [row[0] for row in test]
    archive = checked_file("EuroSAT_RGB.zip", "md5", "f46e308c4d50d4bf32fedad2d3d62f3b")
    with ZipFile(archive) as data:
        members = {"/".join(n.split("/")[-2:]): n for n in data.namelist() if n.lower().endswith(".jpg")}
        assert len(members) == 27000 and set(keys) <= members.keys()
        def image_at(key):
            with Image.open(io.BytesIO(data.read(members[key]))) as im:
                return im.convert("RGB")
        features, texts = encode(keys, image_at, prompts)
    assert torch.isfinite(features).all() and torch.isfinite(texts).all()
    return draws, {key: i for i, key in enumerate(support)}, features, texts, test


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--include-gate", action="store_true", help="Run C's reimplementation of B's idea")
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    draws, positions, features, texts, test = inputs()
    query = features[len(positions):]
    labels = np.array([row[1] for row in test])
    predictions = [dict(path=r[0], label=r[1]) for r in test]
    runs = []
    config = json.loads((ROOT / "results/part_a_round2/propagation/selection.json").read_text())
    a = next(r["config"] for r in config["selected"] if r["config"]["family"] == "Propagation")
    if args.include_gate:
        import query_gated as gate
        b = json.loads((Path(C.RESULT_DIR) / "query_gated/config.json").read_text())
    classes = [f"eurosat:{i}" for i in range(10)]
    for (k, seed), index in draws.items():
        x = features[[positions[p] for p in index["paths"]]]
        model = HyCalCached()
        model.add_session(x, index["class_ids"], dict(zip(classes, texts)))
        assert model.class_ids == classes
        scores = model.score(query)
        results = {"HyCal": model.combine(scores, metric="weighted", n_min_p=5, scale=5).argmax(1).numpy()}
        if args.include_gate:
            weights = gate.base_weights(model)
            assert np.array_equal(results["HyCal"], gate.predict(scores, weights))
            results["B_uniform"] = gate.predict(scores, weights, b["selected_beta"])
            results["B_gate_reimplementation"] = gate.predict(scores, weights, b["selected_beta"], b["selected_tau"])
        if k > 1:
            raw = torch.stack([x[np.array(index["labels"]) == i] for i in range(10)]).cuda()
            text_gpu = texts.cuda()
            proto, cov = core.class_statistics(core.normalize_embeddings(raw + text_gpu[:, None]))
            ctx = dict(raw=raw, texts=text_gpu, proto=proto, cov=cov, query=query.cuda(), memo={})
            for tag, cfg in [("A_local", dict(a, alpha=0)), ("A_propagation", a)]:
                s = propagation.scores(ctx, cfg)
                assert torch.isfinite(s).all()
                results[tag] = s.argmax(1).numpy()
        for tag, pred in results.items():
            assert pred.shape == labels.shape
            for row, value in zip(predictions, pred):
                row[f"{tag}_K{k}_s{seed}"] = int(value)
            runs.append(dict(shots=k, seed=seed, method=tag, correct=int((pred == labels).sum()),
                             n=len(labels), top1_pct=100 * (pred == labels).mean()))
        print(f"K={k} seed={seed} done", flush=True)
    out = Path(C.RESULT_DIR) / "reverse_eurosat"
    out.mkdir(exist_ok=True, parents=True)
    put_csv(out / "predictions.csv", predictions)
    put_csv(out / "runs.csv", runs)


if __name__ == "__main__":
    main()
