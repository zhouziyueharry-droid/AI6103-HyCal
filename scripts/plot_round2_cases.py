"""Plot the eight existing seed-0 Propagation-vs-H2 cases without reselection.

Reads only analysis/cases.json and its selected original images. Each domain
must already contain exactly two corrected and two broken cases in the existing
lexicographic order. No images are cropped, enhanced, rewritten, or scored.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "reports/part_a_round2/analysis"
DOMAINS = ("flowers102", "dtd")
DISPLAY = {"flowers102": "Flowers102", "dtd": "DTD"}
CORRECT_COLOR = "#236D50"
WRONG_COLOR = "#AE3C36"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def selected_cases(document: dict) -> list:
    require(document["seed"] == 0, "Case figure requires the preselected seed-0 cases")
    selected = []
    for domain in DOMAINS:
        groups = [group for group in document["comparisons"] if group["dataset"] == domain and group["method"] == "Propagation" and group["reference"] == "H2"]
        require(len(groups) == 1, f"Expected one existing Propagation-vs-H2 comparison for {domain}")
        group = groups[0]
        require(group["status"] == "complete", f"The {domain} cases are incomplete; no substitute cases will be chosen")
        cases = group["cases"]
        require(len(cases) == 4 and Counter(case["case_type"] for case in cases) == {"corrected": 2, "broken": 2}, f"Expected exactly two corrected and two broken {domain} cases")
        row = []
        for category in ("corrected", "broken"):
            existing = [case for case in cases if case["case_type"] == category]
            require([case["sample_id"] for case in existing] == sorted(case["sample_id"] for case in existing), f"Existing {domain}/{category} order is not lexicographic")
            for case in existing:
                require(case["domain"] == domain, "Case domain does not match its comparison")
                correct_method = case["method_pred_label"] == case["true_label"]
                correct_reference = case["reference_pred_label"] == case["true_label"]
                require((correct_method and not correct_reference) if category == "corrected" else (correct_reference and not correct_method), f"Incorrect corrected/broken classification: {case['sample_id']}")
                require(Path(case["path"]).is_absolute() and Path(case["path"]).is_file(), f"Missing absolute original image path: {case['path']}")
                row.append(case)
        selected.append(row)
    require(len({(case["domain"], case["sample_id"]) for row in selected for case in row}) == 8, "Repeated case in figure")
    return selected


def main() -> None:
    source = OUTPUT / "cases.json"
    source_bytes = source.read_bytes()
    source_digest = hashlib.sha256(source_bytes).hexdigest()
    document = json.loads(source_bytes)
    rows = selected_cases(document)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "figure.dpi": 180, "pdf.fonttype": 42})
    figure = plt.figure(figsize=(16.5, 10.6), facecolor="white")
    grid = figure.add_gridspec(2, 4, left=.065, right=.985, bottom=.095, top=.875, wspace=.12, hspace=.24)
    source_images = []
    for row_index, cases in enumerate(rows):
        for column, case in enumerate(cases):
            cell = grid[row_index, column].subgridspec(3, 1, height_ratios=(.28, 2.8, 1.1), hspace=.02)
            heading = figure.add_subplot(cell[0]); heading.axis("off")
            category = case["case_type"]
            heading.text(0, .5, f"{category.capitalize()} {column % 2 + 1}", ha="left", va="center", fontsize=11, fontweight="bold", color=CORRECT_COLOR if category == "corrected" else WRONG_COLOR)
            original = Path(case["path"])
            original_bytes = original.read_bytes()
            with Image.open(original) as image:
                original_size = list(image.size)
                pixels = np.asarray(image.convert("RGB"))
            image_axis = figure.add_subplot(cell[1])
            image_axis.imshow(pixels, aspect="equal", interpolation="nearest")
            image_axis.axis("off")
            caption = figure.add_subplot(cell[2]); caption.axis("off")
            caption.text(0, .95, case["sample_id"], ha="left", va="top", fontfamily="DejaVu Sans Mono", fontsize=7.8, color="#4A4A4A")
            caption.text(0, .69, f"True: {case['true_classname']}", ha="left", va="top", fontsize=10.1, fontweight="bold")
            caption.text(0, .44, f"H2: {case['reference_pred_classname']}", ha="left", va="top", fontsize=10,
                         color=CORRECT_COLOR if case["reference_pred_label"] == case["true_label"] else WRONG_COLOR)
            caption.text(0, .19, f"Propagation: {case['method_pred_classname']}", ha="left", va="top", fontsize=10,
                         color=CORRECT_COLOR if case["method_pred_label"] == case["true_label"] else WRONG_COLOR)
            source_images.append({"row": row_index, "column": column, "domain": case["domain"], "case_type": category,
                                  "sample_id": case["sample_id"], "path": str(original), "original_size_pixels": original_size,
                                  "original_sha256": hashlib.sha256(original_bytes).hexdigest()})
        row_bounds = grid[row_index, 0].get_position(figure)
        figure.text(.025, (row_bounds.y0 + row_bounds.y1) / 2, DISPLAY[DOMAINS[row_index]], rotation=90, ha="center", va="center", fontsize=14, fontweight="bold")
    figure.suptitle("Propagation versus H2 | seed 0", y=.965, fontsize=17, fontweight="bold")
    figure.text(.5, .92, "Two corrected and two broken predictions per domain; exact cases from the fixed lexicographic selection", ha="center", va="center", fontsize=11)
    figure.text(.5, .055, "Green prediction: correct   |   Red prediction: incorrect", ha="center", fontsize=10)
    figure.text(.5, .027, "Original photographs displayed at their native aspect ratios. Case IDs and labels are copied from cases.json; these examples illustrate individual outcomes.", ha="center", fontsize=9, color="#555555")
    require(sha256(source) == source_digest, "cases.json changed during plotting; rerun after analysis finishes")
    for suffix in ("png", "pdf"):
        metadata = {"Software": "Matplotlib / plot_round2_cases.py"} if suffix == "png" else {
            "Title": "Propagation versus H2: fixed seed-0 cases", "Creator": "plot_round2_cases.py",
            "CreationDate": None, "ModDate": None,
        }
        figure.savefig(OUTPUT / f"case_examples.{suffix}", dpi=180, facecolor="white", metadata=metadata)
    plt.close(figure)
    require(all(sha256(Path(item["path"])) == item["original_sha256"] for item in source_images), "An original image changed during plotting")
    provenance = {
        "cases_json_sha256": source_digest, "plot_script_sha256": sha256(Path(__file__)),
        "selection_policy_copied_from_cases": document["selection_rule"],
        "selection_changed": False, "seed": 0, "layout": "2 rows x 4 columns; Flowers102 then DTD; corrected1, corrected2, broken1, broken2",
        "method": "Propagation", "reference": "H2", "matplotlib_version": matplotlib.__version__,
        "original_images_modified": False, "image_display": "RGB decoding at native aspect ratio with nearest-neighbor display resampling; original files unchanged",
        "source_images": source_images,
        "output_sha256": {f"case_examples.{suffix}": sha256(OUTPUT / f"case_examples.{suffix}") for suffix in ("png", "pdf")},
    }
    (OUTPUT / "case_examples_provenance.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "complete", "case_count": len(source_images), "cases_json_sha256": source_digest, "output_sha256": provenance["output_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
