"""Descriptive Flowers class effects for three frozen methods, CPU only.

Uses exactly the 20 paired support seeds and complete summary.json files.
Top/bottom classes are selected post hoc for explanation, never for tuning.
Error bars are sample SD of paired differences, not confidence intervals.
"""

from __future__ import annotations

from collections import Counter
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/part_a_round2/analysis"
SEEDS = [0, 1, 42, 1993, *range(2, 18)]
METHODS = ("H0", "H2", "Propagation")


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition: bool, message: str):
    if not condition:
        raise ValueError(message)


def main():
    paths = {
        "main_summary": ROOT / "results/part_a_round2/flowers102/test/summary.json",
        "propagation_summary": ROOT / "results/part_a_round2/propagation/flowers102/test/summary.json",
        "main_selection": ROOT / "results/part_a_round2/selection.json",
        "propagation_selection": ROOT / "results/part_a_round2/propagation/selection.json",
        "final_lock": ROOT / "results/part_a_round2/final_execution_lock.json",
        "records": ROOT / "data/manifests/flowers102/records.json",
        "classes": ROOT / "data/manifests/flowers102/classes.json",
    }
    source_hashes = {name: sha(path) for name, path in paths.items()}
    source = {name: read(path) for name, path in paths.items()}
    lock_files = source["final_lock"]["files"]
    for name in ("main_selection", "propagation_selection", "records", "classes"):
        relative = paths[name].relative_to(ROOT).as_posix()
        require(lock_files[relative] == source_hashes[name], f"Frozen input checksum mismatch: {name}")
    classes = {row["label"]: row["classname"] for row in source["classes"]}
    require(set(classes) == set(range(102)), "Expected 102 numeric Flowers labels")
    test_counts = Counter(row["label"] for row in source["records"] if row["split"] == "test")
    require(sum(test_counts.values()) == 2463 and set(test_counts) == set(range(102)), "Unexpected Flowers test partition")
    expected_counts = [test_counts[label] for label in range(102)]
    matrices, configs = {}, {}
    for method in METHODS:
        study = "propagation" if method == "Propagation" else "main"
        choices = [row["config"] for row in source[study + "_selection"]["selected"] if row["config"]["family"] == method]
        require(len(choices) == 1, f"Expected one frozen {method} configuration")
        configs[method] = choices[0]
        rows = [row for row in source[study + "_summary"] if row["config"]["family"] == method]
        require(len(rows) == 20 and {row["seed"] for row in rows} == set(SEEDS), f"{method} does not contain exactly the 20 required seeds")
        by_seed = {row["seed"]: row for row in rows}
        values = []
        for seed in SEEDS:
            row = by_seed[seed]
            require(row["config"] == configs[method], f"{method} configuration differs from frozen development choice")
            require(row["dataset"] == "flowers102" and row["phase"] == "test" and row["shot"] == 5, "Endpoint mismatch")
            require(row["class_counts"] == expected_counts and row["n"] == 2463, "Per-class test counts differ across methods/seeds")
            per_class = np.asarray(row["per_class"], dtype=np.float64)
            require(per_class.shape == (102,) and np.isfinite(per_class).all() and ((per_class >= 0) & (per_class <= 100)).all(), "Invalid class percentages")
            require(abs(per_class.mean() - row["macro"]) < 1e-7, "Per-class accuracy does not reproduce macro")
            require(abs(np.average(per_class, weights=expected_counts) - row["top1"]) < 1e-7, "Per-class accuracy does not reproduce Top1")
            values.append(per_class)
        matrices[method] = np.stack(values)
    differences = matrices["Propagation"] - matrices["H2"]
    means = {method: values.mean(axis=0) for method, values in matrices.items()}
    delta_mean, delta_sd = differences.mean(axis=0), differences.std(axis=0, ddof=1)
    rows = [{
        "label": label, "classname": classes[label], "test_count": test_counts[label], "n_support_seeds": 20,
        "H0_mean_accuracy": float(means["H0"][label]), "H2_mean_accuracy": float(means["H2"][label]),
        "Propagation_mean_accuracy": float(means["Propagation"][label]),
        "Propagation_minus_H2_mean_pp": float(delta_mean[label]),
        "Propagation_minus_H2_sample_sd_pp": float(delta_sd[label]),
        "seeds_improved": int((differences[:, label] > 1e-10).sum()),
        "seeds_tied": int((np.abs(differences[:, label]) <= 1e-10).sum()),
        "seeds_worsened": int((differences[:, label] < -1e-10).sum()),
    } for label in range(102)]
    ascending = sorted(rows, key=lambda row: (row["Propagation_minus_H2_mean_pp"], row["label"]))
    bottom = ascending[:8]
    top = sorted(rows, key=lambda row: (-row["Propagation_minus_H2_mean_pp"], row["label"]))[:8]
    require(not {row["label"] for row in bottom} & {row["label"] for row in top}, "Top/bottom selection overlaps")
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "flowers_class_effects.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    report = {
        "dataset": "flowers102", "phase": "test", "shot": 5, "seeds": SEEDS,
        "methods": list(METHODS), "frozen_configs": configs,
        "units": "Class accuracies are percentages; paired differences and sample SD are percentage points",
        "comparison": "Per seed, subtract H2 accuracy from Propagation accuracy on the same fixed class query images; then summarize the 20 paired differences",
        "selection": "POST HOC DESCRIPTIVE ONLY: top eight by descending mean delta, bottom eight by ascending mean delta; numeric label breaks ties. This display is not used for method/configuration selection.",
        "interpretation": "Sample SD describes support-seed variability on the fixed split. No per-class confidence intervals or significance claims are made. The original four seeds participated in development; all20 is not 20 independent confirmation seeds. No visual causal inference is performed.",
        "rows": rows, "top8_labels": [row["label"] for row in top], "bottom8_labels": [row["label"] for row in bottom],
        "source_sha256": {paths[name].relative_to(ROOT).as_posix(): digest for name, digest in source_hashes.items()},
        "script_sha256": sha(Path(__file__)),
        "paired_seed_deltas_pp_by_label": {str(label): {str(seed): float(differences[index, label]) for index, seed in enumerate(SEEDS)} for label in range(102)},
    }
    (OUT / "flowers_class_effects.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "figure.dpi": 160, "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 7))
    extent = max(abs(row["Propagation_minus_H2_mean_pp"]) + row["Propagation_minus_H2_sample_sd_pp"] for row in top + bottom)
    for ax, selected, title in zip(axes, (top, bottom), ("Top 8 mean changes", "Bottom 8 mean changes")):
        for index, row in enumerate(selected):
            mean, sd = row["Propagation_minus_H2_mean_pp"], row["Propagation_minus_H2_sample_sd_pp"]
            color = "#247A67" if mean >= 0 else "#AD4944"
            ax.errorbar(mean, index, xerr=sd, fmt="o", capsize=4, color=color, markersize=5)
            ax.annotate(f"{mean:+.2f} pp", (mean, index), xytext=(0, -14), textcoords="offset points", ha="center", va="top", fontsize=8, color=color)
        ax.axvline(0, color="#555555", linestyle="--", linewidth=.8)
        ax.set_yticks(range(8), [f"{row['classname']}  (n={row['test_count']})" for row in selected])
        ax.invert_yaxis(); ax.set_ylim(7.65, -.6)
        ax.set_xlim(-extent * 1.08, extent * 1.08)
        ax.set_title(title); ax.set_xlabel("Propagation - H2 class accuracy (percentage points)")
        ax.grid(axis="x", alpha=.2); ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Flowers102 class effects | 20 paired support seeds | mean +/- sample SD", fontsize=13)
    fig.text(.5, .025, "POST HOC DESCRIPTIVE SELECTION: highest and lowest observed class deltas. Not used for tuning; error bars are SD, not CI or significance tests.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .08, 1, .94))
    for suffix in ("png", "pdf"):
        metadata = {"Software": "Matplotlib / analyze_class_effects_round2.py"} if suffix == "png" else {"Creator": "analyze_class_effects_round2.py", "CreationDate": None, "ModDate": None}
        fig.savefig(OUT / f"flowers_class_effects.{suffix}", metadata=metadata, bbox_inches="tight")
    plt.close(fig)
    require(all(sha(paths[name]) == digest for name, digest in source_hashes.items()), "A source changed during analysis")
    print(json.dumps({"status": "complete", "class_count": len(rows), "paired_seed_count": len(SEEDS),
                      "positive_mean_classes": int((delta_mean > 1e-10).sum()), "negative_mean_classes": int((delta_mean < -1e-10).sum()),
                      "zero_mean_classes": int((np.abs(delta_mean) <= 1e-10).sum()),
                      "selected_top8_labels": report["top8_labels"], "selected_bottom8_labels": report["bottom8_labels"], "output": str(OUT)}, indent=2))


if __name__ == "__main__":
    main()
