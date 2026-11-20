"""Run the score function supplied in Member B's reconstruction on C's EuroSAT splits."""
from pathlib import Path

import numpy as np
import torch

import common as C
from hycal_cached import HyCalCached
from member_b_reconstructed.query_gated_hycal import method_scores
from run_paired_transfer import put_csv
from run_reverse_eurosat import inputs


@torch.inference_mode()
def main():
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    draws, positions, features, texts, test = inputs()
    query = features[len(positions):]
    labels = np.array([row[1] for row in test])
    classes = [f"eurosat:{i}" for i in range(10)]
    predictions = [dict(path=row[0], label=row[1]) for row in test]
    runs = []
    for (k, seed), index in draws.items():
        x = features[[positions[path] for path in index["paths"]]]
        model = HyCalCached(lambda_reg=1e-3, lambda_bg=1e-3, device="cuda")
        model.add_session(x, index["class_ids"], dict(zip(classes, texts)))
        assert model.class_ids == classes
        scores = model.score(query, batch_size=64)
        methods = method_scores(scores["cos"].numpy(), scores["md"].numpy(), k)
        for name in ("original", "relaxed", "uniform_150", "low_margin_strong"):
            pred = methods[name].argmax(1)
            if k == 1:
                assert np.array_equal(pred, methods["cosine"].argmax(1))
            for row, value in zip(predictions, pred, strict=True):
                row[f"{name}_K{k}_s{seed}"] = int(value)
            correct = int(np.sum(pred == labels))
            runs.append(dict(shots=k, seed=seed, method=name, correct=correct,
                             n=len(labels), top1_pct=100 * correct / len(labels)))
        print(f"K={k} seed={seed} done", flush=True)
    out = Path(C.RESULT_DIR) / "reverse_eurosat/member_b_reconstructed"
    out.mkdir(parents=True, exist_ok=True)
    put_csv(out / "predictions.csv", predictions)
    put_csv(out / "runs.csv", runs)


if __name__ == "__main__":
    main()
