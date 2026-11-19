"""Analyze frozen round2 results without choosing methods from final endpoints.

Run only after the final execution is authorized:
    .venv/Scripts/python.exe scripts/analyze_round2.py
    .venv/Scripts/python.exe scripts/analyze_round2.py --allow-partial

--self-test exercises synthetic statistics only and does not read results.
All writes stay in reports/part_a_round2/analysis. No GPU or scoring is used.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results/part_a_round2"
OUTPUT = ROOT / "reports/part_a_round2/analysis"
DATASETS = ("flowers102", "dtd", "joint_flowers_dtd")
CLASS_COUNTS = {"flowers102": 102, "dtd": 47, "joint_flowers_dtd": 149, "aircraft": 100}
TEST_COUNTS = {"flowers102": 2463, "dtd": 1692, "joint_flowers_dtd": 4155}
PRIMARY = [0, 1, 42, 1993]
SUPPLEMENTARY = list(range(2, 18))
SEED_GROUPS = {"primary4": PRIMARY, "supplementary16": SUPPLEMENTARY, "all20": PRIMARY + SUPPLEMENTARY}
CHOICE_KEYS = ("best_improvement_on_development", "best_geometry_hypothesis_on_development")
MECHANISMS = [
    ("RDA_transport", "RDA_untransported_matched"),
    ("Propagation", "Propagation_raw"),
    ("Propagation", "Propagation_scale"),
    ("Propagation", "Propagation_local"),
]
OTHER_MATCHED = [
    ("RDA_semantic", "RDA_random_ablation"),
    ("RDA_semantic", "RDA_local_ablation"),
    ("RDA_semantic", "RDA_global_matched"),
    ("RDA_semantic", "RDA_rawmean_ablation"),
    ("RDA_semantic", "RDA_logdet_ablation"),
]
TUNED_COMPARISONS = [
    ("RDA_transport", "RDA_global"), ("RDA_transport", "RDA_semantic"),
    ("RDA_semantic", "RDA_global"), ("Propagation", "RDA_transport"),
    ("Propagation", "RDA_global"), ("Propagation", "RDA_semantic"),
    ("Propagation", "Propagation_raw_tuned"), ("Propagation", "Propagation_scale_tuned"),
]
JOINT_METRICS = ["macro", "top1", "equal_domain_macro", "flowers_macro", "dtd_macro",
                 "flowers_top1", "dtd_top1", "cross_domain_error_rate",
                 "flowers_to_dtd_rate", "dtd_to_flowers_rate"]
LOWER_IS_BETTER = {"cross_domain_error_rate", "flowers_to_dtd_rate", "dtd_to_flowers_rate"}
TEN_SHOT_FAMILIES = ["H0", "H2", "GDA", "RDA_global", "RDA_transport", "Propagation", "Propagation_raw_tuned", "Propagation_scale_tuned"]
CI_SCOPE = (
    "Student t 95% intervals describe paired support-seed variation conditional on the fixed query split, "
    "fixed selected configurations, and this experiment. They do not cover dataset-population uncertainty, "
    "hyperparameter-selection uncertainty, or generalization to arbitrary domains. Multiple comparisons are unadjusted."
)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def canonical(value) -> str:
    return json.dumps(value, sort_keys=True)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def finite_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


class Audit:
    def __init__(self):
        self.issues = []
        self.files = {}

    def issue(self, code, detail, **context):
        self.issues.append({"code": code, "detail": str(detail), **context})

    def record(self, path: Path):
        key = path.relative_to(ROOT).as_posix()
        digest = sha256(path)
        if key in self.files:
            require(self.files[key] == digest, f"Input changed during analysis: {key}")
        self.files[key] = digest
        return digest

    def verified_hash(self, relative: str, expected: str):
        path = (ROOT / relative).resolve()
        require(path.is_relative_to(ROOT.resolve()), f"Lock path escapes project: {relative}")
        require(self.record(path) == expected, f"Frozen input checksum mismatch: {relative}")


@dataclass
class Observation:
    dataset: str
    phase: str
    study: str
    family: str
    seed: int
    config: dict
    metrics: dict
    predictions: dict | None
    predictions_valid: bool
    n: int
    tag: str
    shot: int = 5


def protocol(audit: Audit) -> dict:
    """Read only development choices recorded before final endpoint execution."""
    path = RESULTS / "final_execution_lock.json"
    lock = read_json(path)
    audit.record(path)
    require("files" in lock, "Final execution lock has no frozen input list")
    for relative, expected in lock["files"].items():
        audit.verified_hash(relative, expected)
    expected = {}
    tuned = []
    for study, folder in (("main", RESULTS), ("propagation", RESULTS / "propagation")):
        selection_path = folder / "selection.json"
        relative = selection_path.relative_to(ROOT).as_posix()
        require(relative in lock["files"], f"Selection is absent from final lock: {relative}")
        choice = read_json(selection_path)
        audit.record(selection_path)
        for item in choice["selected"]:
            config = item["config"]
            family = config["family"]
            require(family not in expected, f"Frozen family occurs more than once: {family}")
            expected[family] = {"study": study, "config": config}
            if "macro" in item:  # family-best choice frozen on development, never final accuracy
                tuned.append(family)
    choices = {}
    for key in CHOICE_KEYS:
        item = lock[key]
        config = item["config"]
        family = config["family"]
        require(family in expected and expected[family]["config"] == config, f"Locked headline choice disagrees with selected configuration: {key}")
        choices[key] = {"family": family, "config": config, "source": "final_execution_lock.json", "selection_endpoint": "Flowers development"}
    matched_checks = []
    changes = {
        "RDA_untransported_matched": {"transport": False},
        "Propagation_raw": {"mode": "raw"}, "Propagation_scale": {"mode": "scale_only"}, "Propagation_local": {"alpha": 0.},
        "RDA_random_ablation": {"neighbor_mode": "random"}, "RDA_local_ablation": {"alpha": 0.},
        "RDA_global_matched": {"neighbors": 0}, "RDA_rawmean_ablation": {"rawmean_center": True}, "RDA_logdet_ablation": {"logdet": True},
    }
    for method, reference in MECHANISMS + OTHER_MATCHED:
        if method not in expected or reference not in expected:
            continue
        allowed = changes[reference]
        a, b = expected[method]["config"], expected[reference]["config"]
        strip = lambda config: {key: value for key, value in config.items() if key != "family" and key not in allowed}
        require(strip(a) == strip(b) and all(b.get(key) == value for key, value in allowed.items()), f"Named matched control changes unregistered parameters: {method} vs {reference}")
        matched_checks.append({"method": method, "reference": reference, "only_allowed_control_overrides": allowed})
    return {"lock": lock, "expected": expected, "family_order": list(expected), "tuned_families": tuned, "choices": choices, "matched_config_validation": matched_checks}


def query_context(dataset: str, phase: str, audit: Audit) -> dict:
    folder = ROOT / "data/manifests" / dataset
    records_path, classes_path = folder / "records.json", folder / "classes.json"
    records, classes = read_json(records_path), read_json(classes_path)
    audit.record(records_path); audit.record(classes_path)
    by_id = {row["sample_id"]: row for row in records}
    require(len(by_id) == len(records), f"Duplicate manifest IDs: {dataset}")
    require(len(classes) == CLASS_COUNTS[dataset] and {row["label"] for row in classes} == set(range(CLASS_COUNTS[dataset])), f"Unexpected class mapping: {dataset}")
    if phase == "confirmation":
        require(dataset == "flowers102", "Only Flowers has a registered confirmation endpoint")
        partition = ROOT / "data/manifests/flowers102/round2_validation.json"
        ids = set(read_json(partition)["indices"]["confirmation"])
        audit.record(partition)
        require(all(by_id[sample_id]["split"] == "val" for sample_id in ids), "Confirmation IDs must come from Flowers validation")
    else:
        ids = {row["sample_id"] for row in records if row["split"] == "test"}
        require(len(ids) > 0, f"Empty {dataset} test split")
        if dataset in TEST_COUNTS:
            require(len(ids) == TEST_COUNTS[dataset], f"Unexpected {dataset} test size")
    return {"records": by_id, "classes": {row["label"]: row for row in classes}, "ids": ids,
            "counts": Counter(by_id[sample_id]["label"] for sample_id in ids)}


def row_metrics(row: dict, dataset: str) -> dict:
    """per_class/class_counts positions are numeric labels, not text row order."""
    size = CLASS_COUNTS[dataset]
    per_class, counts = row["per_class"], row["class_counts"]
    require(len(per_class) == len(counts) == size, "Incorrect per-class array length")
    require(all(finite_number(value) and 0 <= value <= 100 for value in per_class), "Invalid per-class percentage")
    require(all(type(value) is int and value > 0 for value in counts), "Every endpoint must contain each class; invalid class_counts")
    require(sum(counts) == row["n"], "class_counts sum differs from n")
    macro = statistics.mean(per_class)
    top1 = sum(value * count for value, count in zip(per_class, counts)) / sum(counts)
    require(abs(macro - row["macro"]) < 1e-7 and abs(top1 - row["top1"]) < 1e-7, "Summary metrics disagree with per-class reconstruction")
    result = {"macro": macro, "top1": top1}
    if dataset == "joint_flowers_dtd":
        flower_macro, dtd_macro = statistics.mean(per_class[:102]), statistics.mean(per_class[102:149])
        result.update(
            equal_domain_macro=(flower_macro + dtd_macro) / 2,
            flowers_macro=flower_macro, dtd_macro=dtd_macro,
            flowers_top1=sum(per_class[i] * counts[i] for i in range(102)) / sum(counts[:102]),
            dtd_top1=sum(per_class[i] * counts[i] for i in range(102, 149)) / sum(counts[102:149]),
            cross_domain_error_rate=None, flowers_to_dtd_rate=None, dtd_to_flowers_rate=None,
        )
    return result


def prediction_metrics(path: Path, row: dict, context: dict, dataset: str, retain: bool, audit: Audit):
    size = CLASS_COUNTS[dataset]
    seen, predictions = set(), {} if retain else None
    counts, correct = Counter(), Counter()
    crosses, domain_n = Counter(), Counter()
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        require(set(reader.fieldnames or []) == {"sample_id", "true_label", "pred_label", "correct"}, "Unexpected prediction CSV columns")
        for record in reader:
            sample_id = record["sample_id"]
            true, pred, flag = int(record["true_label"]), int(record["pred_label"]), int(record["correct"])
            require(sample_id not in seen and sample_id in context["ids"], "Duplicate or unexpected query sample ID")
            require(0 <= true < size and 0 <= pred < size, "Prediction label outside numeric-label range")
            require(context["records"][sample_id]["label"] == true, "Prediction CSV true label differs from manifest")
            require(flag == int(true == pred), "Prediction correct flag is inconsistent")
            seen.add(sample_id); counts[true] += 1; correct[true] += flag
            if predictions is not None:
                predictions[sample_id] = {"true_label": true, "pred_label": pred, "correct": bool(flag)}
            if dataset == "joint_flowers_dtd":
                domain = "flowers102" if true < 102 else "dtd"
                domain_n[domain] += 1
                crosses[domain] += int((true < 102) != (pred < 102))
    require(seen == context["ids"] and len(seen) == row["n"], "Prediction CSV query set is incomplete")
    require([counts[i] for i in range(size)] == row["class_counts"], "Prediction CSV class counts disagree with summary")
    require(all(abs(100 * correct[i] / counts[i] - row["per_class"][i]) < 1e-7 for i in range(size)), "Prediction CSV class accuracy disagrees with summary")
    audit.record(path)
    metrics = {}
    if dataset == "joint_flowers_dtd":
        metrics = {
            "cross_domain_error_rate": 100 * sum(crosses.values()) / len(seen),
            "flowers_to_dtd_rate": 100 * crosses["flowers102"] / domain_n["flowers102"],
            "dtd_to_flowers_rate": 100 * crosses["dtd"] / domain_n["dtd"],
        }
    return metrics, predictions


def load_endpoint(dataset: str, phase: str, study: str, frozen: dict, context: dict, audit: Audit,
                  families: list | None = None, shot: int = 5, expected_seeds: list | None = None,
                  not_before_utc: str | None = None) -> list[Observation]:
    folder = (RESULTS if study in ("main", "combined_10shot") else RESULTS / "propagation") / dataset / phase
    summary = folder / "summary.json"
    expected = {family: value["config"] for family, value in frozen["expected"].items() if family in families} if families is not None else {family: value["config"] for family, value in frozen["expected"].items() if value["study"] == study}
    seeds = expected_seeds if expected_seeds is not None else PRIMARY if phase == "confirmation" else SEED_GROUPS["all20"]
    try:
        rows = read_json(summary)
        audit.record(summary)
        run_lock = read_json(folder / "run_lock.json")
        audit.record(folder / "run_lock.json")
        for relative, digest in run_lock["files"].items():
            audit.verified_hash(relative.removeprefix("source:"), digest)
        require(isinstance(rows, list), "summary.json must be an array")
    except FileNotFoundError as error:
        audit.issue("missing_endpoint", error, dataset=dataset, phase=phase, study=study, path=str(summary))
        return []
    except (OSError, ValueError, KeyError, TypeError) as error:
        audit.issue("invalid_endpoint", error, dataset=dataset, phase=phase, study=study, path=str(summary))
        return []
    frequencies = Counter()
    for item in rows:
        if isinstance(item, dict) and isinstance(item.get("config"), dict):
            family, seed = item["config"].get("family"), item.get("seed")
            if isinstance(family, str) and type(seed) is int:
                frequencies[(family, seed)] += 1
    loaded = []
    for row in rows:
        try:
            family, seed, config = row["config"]["family"], row["seed"], row["config"]
            require(frequencies[(family, seed)] == 1, "Duplicate family/seed row; no row is selected from duplicates")
            require(family in expected and config == expected[family], "Row configuration differs from frozen development choice")
            require(type(seed) is int and seed in seeds, "Unexpected endpoint seed")
            require(row["dataset"] == dataset and row["phase"] == phase and row["shot"] == shot, "Row endpoint/shot mismatch")
            require(row["n"] == len(context["ids"]), "Row query count differs from registered endpoint")
            require(row["class_counts"] == [context["counts"][i] for i in range(CLASS_COUNTS[dataset])], "Summary class counts differ from query manifest")
            expected_tag = f"s{seed}_{family}_{hashlib.sha256(canonical(config).encode()).hexdigest()[:10]}"
            require(row["tag"] == expected_tag, "Row tag/config identity mismatch")
            locked = datetime.fromisoformat(frozen["lock"]["locked_before_confirmation_and_test_utc"].replace("Z", "+00:00"))
            computed = datetime.fromisoformat(row["computed_utc"].replace("Z", "+00:00"))
            require(computed >= locked, "Endpoint row was computed before the final execution lock")
            phase_locked = datetime.fromisoformat(run_lock["created_utc"].replace("Z", "+00:00"))
            require(computed >= phase_locked, "Endpoint row was computed before its dataset/phase input lock")
            if not_before_utc is not None:
                registered = datetime.fromisoformat(not_before_utc.replace("Z", "+00:00"))
                require(computed >= registered, "Endpoint row was computed before its additional transfer registration lock")
            metrics = row_metrics(row, dataset)
            predictions, valid = None, False
            try:
                additions, predictions = prediction_metrics(folder / (row["tag"] + "_predictions.csv"), row, context, dataset, seed == 0, audit)
                metrics.update(additions); valid = True
            except OSError as error:
                audit.issue("missing_predictions", error, dataset=dataset, phase=phase, study=study, family=family, seed=seed)
            loaded.append(Observation(dataset, phase, frozen["expected"][family]["study"], family, seed, config, metrics, predictions, valid, row["n"], row["tag"], shot))
        except (OSError, ValueError, KeyError, TypeError, ZeroDivisionError) as error:
            audit.issue("invalid_summary_row", error, dataset=dataset, phase=phase, study=study, tag=row.get("tag") if isinstance(row, dict) else None)
    present = {(row.family, row.seed) for row in loaded}
    missing = [{"family": family, "seeds": [seed for seed in seeds if (family, seed) not in present]} for family in expected]
    missing = [item for item in missing if item["seeds"]]
    if missing:
        audit.issue("missing_expected_rows", "Frozen family/seed coverage is incomplete", dataset=dataset, phase=phase, study=study, missing=missing)
    return loaded


def stats(values: list[float]) -> dict:
    return {"n": len(values), "mean": statistics.mean(values) if values else None,
            "sd": statistics.stdev(values) if len(values) >= 2 else None}


def paired_stats(differences: list[float], higher_is_better: bool = True) -> dict:
    result = stats(differences)
    half = float(t.ppf(.975, result["n"] - 1)) * result["sd"] / math.sqrt(result["n"]) if result["n"] >= 2 else None
    direction = 1 if higher_is_better else -1
    return {"n_pairs": result["n"], "mean_delta_pp": result["mean"], "sd_delta_pp": result["sd"],
            "ci95_low_pp": result["mean"] - half if half is not None else None,
            "ci95_high_pp": result["mean"] + half if half is not None else None,
            "wins": sum(direction * value > 1e-10 for value in differences), "ties": sum(abs(value) <= 1e-10 for value in differences),
            "losses": sum(direction * value < -1e-10 for value in differences)}


def seed_groups(phase: str) -> dict:
    return {"primary4": PRIMARY} if phase in ("confirmation", "test_10shot") else SEED_GROUPS


def primary_metric(dataset: str) -> str:
    return "equal_domain_macro" if dataset == "joint_flowers_dtd" else "macro"


def make_summaries(observations: list, scopes: list, frozen: dict, allow_partial: bool) -> list:
    result = []
    lookup = {(row.dataset, row.phase, row.family, row.seed): row for row in observations}
    for dataset, phase in scopes:
        for group_name, seeds in seed_groups(phase).items():
            for family in frozen["family_order"]:
                found = [lookup[(dataset, phase, family, seed)] for seed in seeds if (dataset, phase, family, seed) in lookup]
                present = [row.seed for row in found]
                missing = [seed for seed in seeds if seed not in present]
                complete = not missing and all(row.predictions_valid for row in found)
                record = {"dataset": dataset, "phase": phase, "shot": 10 if phase == "test_10shot" else 5, "seed_group": group_name, "study": frozen["expected"][family]["study"],
                          "family": family, "config": frozen["expected"][family]["config"], "n_seeds": len(found), "expected_n_seeds": len(seeds),
                          "seeds": present, "missing_seeds": missing, "status": "complete" if complete else "partial",
                          "statistics_available": complete or allow_partial, "primary_metric": primary_metric(dataset)}
                record["development_roles"] = [key for key, choice in frozen.get("choices", {}).items() if choice["family"] == family]
                for metric in JOINT_METRICS if dataset == "joint_flowers_dtd" else ["macro", "top1"]:
                    available = [row.metrics[metric] for row in found if finite_number(row.metrics.get(metric))]
                    estimate = stats(available) if complete or allow_partial else {"n": len(available), "mean": None, "sd": None}
                    for suffix, value in estimate.items():
                        record[f"{metric}_{suffix}"] = value
                result.append(record)
    return result


def comparison_specs(frozen: dict) -> list:
    specs = {}
    for method in frozen["family_order"]:
        for reference in ("H0", "H2", "GDA"):
            if method != reference and reference in frozen["expected"]:
                specs[(method, reference)] = "baseline_reference"
    for kind, pairs in (("matched_mechanism", MECHANISMS + OTHER_MATCHED), ("independently_development_tuned_families", TUNED_COMPARISONS)):
        for method, reference in pairs:
            if method in frozen["expected"] and reference in frozen["expected"]:
                specs[(method, reference)] = kind
    return [(method, reference, kind) for (method, reference), kind in specs.items()]


def paired_record(a: dict, b: dict, seeds: list, metric_a: str, metric_b: str, allow_partial: bool, **fields) -> dict:
    present_a = {seed for seed in seeds if seed in a and a[seed].predictions_valid and finite_number(a[seed].metrics.get(metric_a))}
    present_b = {seed for seed in seeds if seed in b and b[seed].predictions_valid and finite_number(b[seed].metrics.get(metric_b))}
    paired = [seed for seed in seeds if seed in present_a & present_b]
    complete = len(paired) == len(seeds)
    differences = [a[seed].metrics[metric_a] - b[seed].metrics[metric_b] for seed in paired]
    result = paired_stats(differences, higher_is_better=metric_a not in LOWER_IS_BETTER)
    if not complete and not allow_partial:
        for key in ("mean_delta_pp", "sd_delta_pp", "ci95_low_pp", "ci95_high_pp", "wins", "ties", "losses"):
            result[key] = None
    return {**fields, "shot": 10 if fields.get("phase") == "test_10shot" else 5,
            "metric": metric_a, "reference_metric": metric_b,
            "metric_direction": "lower_is_better" if metric_a in LOWER_IS_BETTER else "higher_is_better", "expected_n_pairs": len(seeds), "seeds": paired,
            "missing_method_seeds": [seed for seed in seeds if seed not in present_a],
            "missing_reference_seeds": [seed for seed in seeds if seed not in present_b],
            "status": "complete" if complete else "partial", "statistics_available": complete or allow_partial,
            **result, "seed_deltas_pp": dict(zip(map(str, paired), differences)) if complete or allow_partial else {}}


def make_pairs(observations: list, scopes: list, frozen: dict, allow_partial: bool, include_stage_pairs: bool = True) -> list:
    groups = {}
    for row in observations:
        groups.setdefault((row.dataset, row.phase, row.family), {})[row.seed] = row
    result = []
    for dataset, phase in scopes:
        for group_name, seeds in seed_groups(phase).items():
            for method, reference, kind in comparison_specs(frozen):
                for metric in JOINT_METRICS if dataset == "joint_flowers_dtd" else ["macro", "top1"]:
                    result.append(paired_record(groups.get((dataset, phase, method), {}), groups.get((dataset, phase, reference), {}), seeds, metric, metric, allow_partial,
                                                dataset=dataset, reference_dataset=dataset, phase=phase, seed_group=group_name, method=method, reference=reference, comparison_type=kind))
    if include_stage_pairs and ("flowers102", "test") in scopes and ("joint_flowers_dtd", "test") in scopes:
        for group_name, seeds in SEED_GROUPS.items():
            for family in frozen["family_order"]:
                for metric_a, metric_b in (("flowers_macro", "macro"), ("flowers_top1", "top1")):
                    result.append(paired_record(groups.get(("joint_flowers_dtd", "test", family), {}), groups.get(("flowers102", "test", family), {}), seeds, metric_a, metric_b, allow_partial,
                                                dataset="joint_flowers_dtd", reference_dataset="flowers102", phase="test", seed_group=group_name, method=family, reference=family, comparison_type="candidate_expansion_S2_minus_S1"))
    return result


def verify_stage_query_alignment(contexts: dict, audit: Audit) -> bool:
    """S2-S1 effects must reuse the same Flowers query images and labels."""
    single, joint = contexts.get(("flowers102", "test")), contexts.get(("joint_flowers_dtd", "test"))
    if single is None or joint is None:
        return False
    try:
        joined = [joint["records"][sample_id] for sample_id in joint["ids"] if joint["records"][sample_id]["label"] < 102]
        mapped = {row["source_sample_id"]: row for row in joined}
        require(len(mapped) == len(joined) and set(mapped) == single["ids"], "S1/S2 Flowers query sets differ")
        require(all(row["domain"] == "flowers102" and row["label"] == single["records"][source_id]["label"] for source_id, row in mapped.items()), "S1/S2 Flowers domain/label mapping differs")
        return True
    except (ValueError, KeyError, TypeError) as error:
        audit.issue("invalid_stage_query_alignment", error)
        return False


def optional_tenshot(frozen: dict | None, allow_partial: bool, requested: bool) -> dict:
    """Separate registered appendix; absence never invalidates the main endpoints."""
    result = {
        "status": "not_requested" if not requested else "not_available_optional",
        "dataset": "flowers102", "phase": "test_10shot", "shot": 10,
        "seeds": PRIMARY, "families": TEN_SHOT_FAMILIES,
        "scope": "Optional 8-family x 4-seed supplement using fixed 5-shot-selected settings, without retuning; H0/H2 retain official K-dependent gate/weight behavior. This is separate from the primary 20-seed endpoints.",
        "summaries": [], "paired_comparisons": [], "issues": [], "input_sha256": {},
    }
    if not requested:
        return result
    if frozen is None:
        result["status"] = "blocked_by_missing_or_invalid_final_protocol"
        return result
    if not (RESULTS / "flowers102/test_10shot/summary.json").is_file():
        return result
    audit = Audit()
    try:
        require(set(TEN_SHOT_FAMILIES) <= frozen["expected"].keys(), "10-shot families are not all present in frozen 5-shot selections")
        context = query_context("flowers102", "test_10shot", audit)
        observations = load_endpoint("flowers102", "test_10shot", "combined_10shot", frozen, context, audit,
                                     families=TEN_SHOT_FAMILIES, shot=10, expected_seeds=PRIMARY)
        subset = {**frozen, "family_order": TEN_SHOT_FAMILIES,
                  "expected": {family: frozen["expected"][family] for family in TEN_SHOT_FAMILIES}}
        scopes = [("flowers102", "test_10shot")]
        result["summaries"] = make_summaries(observations, scopes, subset, allow_partial)
        result["paired_comparisons"] = make_pairs(observations, scopes, subset, allow_partial)
        for relative, digest in audit.files.items():
            require(sha256(ROOT / relative) == digest, f"10-shot input changed during analysis: {relative}")
        result["status"] = "complete" if len(observations) == 32 and not audit.issues and all(row["status"] == "complete" for row in result["summaries"]) else "partial_or_invalid_optional"
    except (OSError, ValueError, KeyError, TypeError) as error:
        audit.issue("invalid_optional_tenshot", error)
        result["status"] = "partial_or_invalid_optional"
    result["issues"], result["input_sha256"] = audit.issues, audit.files
    return result


def optional_aircraft(frozen: dict | None, allow_partial: bool, requested: bool) -> dict:
    """A pre-registered additional transfer endpoint, separate from the main three."""
    result = {"status": "not_requested" if not requested else "not_available_optional",
              "dataset": "aircraft", "phase": "test", "shot": 5, "num_classes": 100,
              "seeds": SEED_GROUPS["all20"],
              "scope": "Optional Aircraft100 transfer with unchanged Flowers-selected configurations; no new tuning or method selection. Missing outputs do not invalidate the main three endpoints.",
              "summaries": [], "paired_comparisons": [], "issues": [], "input_sha256": {}}
    if not requested:
        return result
    if frozen is None:
        result["status"] = "blocked_by_missing_or_invalid_final_protocol"
        return result
    paths = [RESULTS / "aircraft/test/summary.json", RESULTS / "propagation/aircraft/test/summary.json"]
    if not any(path.is_file() for path in paths):
        return result
    audit = Audit()
    try:
        registration_path = RESULTS / "aircraft_transfer_execution_lock.json"
        registration = read_json(registration_path)
        audit.record(registration_path)
        required_registration_files = {
            "scripts/run_aircraft_round2.py", "reports/part_a_round2/aircraft_transfer_protocol.md",
            "results/part_a_round2/selection.json", "results/part_a_round2/propagation/selection.json",
        }
        require(required_registration_files <= registration["files"].keys(), "Aircraft registration lock omits runner/protocol/frozen choices")
        for relative, digest in registration["files"].items():
            audit.verified_hash(relative, digest)
        result["transfer_registered_utc"] = registration["created_utc"]
        context = query_context("aircraft", "test", audit)
        result["test_query_count"] = len(context["ids"])
        protocol_path = ROOT / "reports/part_a_round2/aircraft_transfer_protocol.md"
        audit.record(protocol_path)
        observations = []
        for study in ("main", "propagation"):
            observations.extend(load_endpoint("aircraft", "test", study, frozen, context, audit, not_before_utc=registration["created_utc"]))
        scopes = [("aircraft", "test")]
        result["summaries"] = make_summaries(observations, scopes, frozen, allow_partial)
        result["paired_comparisons"] = make_pairs(observations, scopes, frozen, allow_partial)
        for relative, digest in audit.files.items():
            require(sha256(ROOT / relative) == digest, f"Aircraft input changed during analysis: {relative}")
        result["status"] = "complete" if len(observations) == len(frozen["expected"]) * 20 and not audit.issues and all(row["status"] == "complete" for row in result["summaries"]) else "partial_or_invalid_optional"
    except (OSError, ValueError, KeyError, TypeError) as error:
        audit.issue("invalid_optional_aircraft", error)
        result["status"] = "partial_or_invalid_optional"
    result["issues"], result["input_sha256"] = audit.issues, audit.files
    return result


def make_cases(observations: list, contexts: dict, frozen: dict, limit: int = 2) -> dict:
    primary = frozen["choices"][CHOICE_KEYS[0]]["family"]
    geometry = frozen["choices"][CHOICE_KEYS[1]]["family"]
    pairs = [(primary, "H2")]
    reference = {"RDA_transport": "RDA_untransported_matched", "Propagation": "Propagation_raw"}.get(geometry)
    if reference:
        pairs.append((geometry, reference))
    pairs = list(dict.fromkeys((method, ref) for method, ref in pairs if method != ref))
    rows = {(row.dataset, row.family): row for row in observations if row.phase == "test" and row.seed == 0 and row.predictions_valid}
    result = {"seed": 0, "selection_rule": "First sample IDs in lexicographic order within corrected/broken categories; at most two of each per pre-specified comparison; illustrative only, no reselection", "comparisons": []}
    for dataset in DATASETS:
        context = contexts.get((dataset, "test"))
        for method, ref in pairs:
            a, b = rows.get((dataset, method)), rows.get((dataset, ref))
            group = {"dataset": dataset, "method": method, "reference": ref, "status": "missing_seed0_predictions", "cases": []}
            if a is None or b is None or context is None or a.predictions is None or b.predictions is None:
                result["comparisons"].append(group)
                continue
            require(a.predictions.keys() == b.predictions.keys(), "Case comparison query sets differ")
            categories = {"corrected": [], "broken": [], "both_correct": [], "both_wrong": []}
            for sample_id in sorted(a.predictions):
                aa, bb = a.predictions[sample_id], b.predictions[sample_id]
                category = "corrected" if aa["correct"] and not bb["correct"] else "broken" if bb["correct"] and not aa["correct"] else "both_correct" if aa["correct"] else "both_wrong"
                categories[category].append(sample_id)
            group.update(status="complete", counts={key: len(value) for key, value in categories.items()})
            for category in ("corrected", "broken"):
                for sample_id in categories[category][:limit]:
                    record = context["records"][sample_id]
                    aa, bb = a.predictions[sample_id], b.predictions[sample_id]
                    group["cases"].append({
                        "case_type": category, "sample_id": sample_id, "path": record["path"],
                        "domain": record.get("domain", dataset), "source_sample_id": record.get("source_sample_id", sample_id),
                        "true_label": aa["true_label"], "true_classname": context["classes"][aa["true_label"]]["classname"],
                        "method_pred_label": aa["pred_label"], "method_pred_classname": context["classes"][aa["pred_label"]]["classname"],
                        "reference_pred_label": bb["pred_label"], "reference_pred_classname": context["classes"][bb["pred_label"]]["classname"],
                    })
            result["comparisons"].append(group)
    return result


def main_figures(summaries: list, paired: list, frozen: dict) -> list[str]:
    """Fixed method order and development roles; never rank by final accuracy."""
    preferred = ["CLIP", "VisualPrototype", "H0", "H2", "H2_weight", "MD", "GDA", "TipAdapter", "ProKeR",
                 "RDA_global", "RDA_semantic", "RDA_transport", "Propagation", "Propagation_raw_tuned", "Propagation_scale_tuned"]
    methods = [family for family in preferred if family in frozen["tuned_families"]]
    methods += [family for family in frozen["tuned_families"] if family not in methods]
    if not methods:
        return []
    primary = frozen["choices"][CHOICE_KEYS[0]]["family"]
    geometry = frozen["choices"][CHOICE_KEYS[1]]["family"]
    labels = [family + (" *" if family == primary else "") + (" +" if family == geometry else "") for family in methods]
    lookup = {(row["dataset"], row["family"]): row for row in summaries if row["phase"] == "test" and row["seed_group"] == "all20"}
    partial_figure = any((dataset, family) not in lookup or lookup[(dataset, family)]["status"] != "complete" for dataset in DATASETS for family in methods)
    titles = ["Flowers102: class macro", "DTD: class macro", "Joint: equal-domain macro"]
    colors = ["#2B6C93", "#247A67", "#8154A0"]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "figure.dpi": 150})
    fig, axes = plt.subplots(1, 3, figsize=(17, max(7, .43 * len(methods) + 2)), sharey=True)
    for ax, dataset, title, color in zip(axes, DATASETS, titles, colors):
        metric = primary_metric(dataset)
        if not any(lookup.get((dataset, family), {}).get(f"{metric}_mean") is not None for family in methods):
            ax.set_title(title); ax.set_xticks([]); ax.set_xlabel("No completed summary available")
            ax.text(.5, .5, "Awaiting complete endpoint summary\nNo endpoint results displayed", ha="center", va="center", transform=ax.transAxes, color="#777777", fontsize=12, linespacing=1.8)
            ax.spines[["top", "right", "bottom"]].set_visible(False)
            continue
        for index, family in enumerate(methods):
            row = lookup.get((dataset, family))
            if row is None or row.get(f"{metric}_mean") is None:
                ax.text(.98, index, "incomplete", transform=ax.get_yaxis_transform(), ha="right", va="center", color="#888888", fontsize=8)
                continue
            mean, sd = row[f"{metric}_mean"], row[f"{metric}_sd"]
            marker = "*" if family == primary else "D" if family == geometry else "o"
            ax.errorbar(mean, index, xerr=sd if sd is not None else 0, fmt=marker, markersize=9 if marker == "*" else 5,
                        capsize=3, color=color, markerfacecolor=color if row["status"] == "complete" else "white")
            if row["status"] != "complete":
                ax.annotate(f" n={row['n_seeds']}/20", (mean, index), xytext=(5, 5), textcoords="offset points", fontsize=7)
        ax.set_title(title); ax.set_xlabel("Accuracy (%)")
        ax.grid(axis="x", alpha=.2); ax.spines[["top", "right"]].set_visible(False)
        ax.set_yticks(range(len(methods)), labels)
        ax.tick_params(axis="y", length=0)
    axes[0].invert_yaxis()
    fig.suptitle(("PARTIAL ANALYSIS | " if partial_figure else "") + "Frozen development choices | 20 support seeds | mean +/- sample SD", fontsize=13)
    fig.text(.5, .015, "* Primary improvement and + geometry hypothesis were chosen on Flowers development, before confirmation/test. Open markers denote explicitly allowed partial data.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .055, 1, .95))
    artifacts = []
    for suffix in ("png", "pdf"):
        name = f"main_results.{suffix}"; fig.savefig(OUTPUT / name, bbox_inches="tight"); artifacts.append(name)
    plt.close(fig)

    effect_lookup = {(row["dataset"], row["method"], row["reference"]): row for row in paired
                     if row["phase"] == "test" and row["seed_group"] == "all20" and row["comparison_type"] == "matched_mechanism" and row["metric"] == primary_metric(row["dataset"])}
    effect_labels = [f"{method} -\n{reference}" for method, reference in MECHANISMS]
    fig, axes = plt.subplots(1, 3, figsize=(17, 6), sharey=True)
    for ax, dataset, title, color in zip(axes, DATASETS, titles, colors):
        if not any(effect_lookup.get((dataset, method, reference), {}).get("mean_delta_pp") is not None for method, reference in MECHANISMS):
            ax.set_title(title); ax.set_xticks([]); ax.set_xlabel("No completed summary available")
            ax.text(.5, .5, "Awaiting complete endpoint summary\nNo mechanism effects displayed", ha="center", va="center", transform=ax.transAxes, color="#777777", fontsize=12, linespacing=1.8)
            ax.spines[["top", "right", "bottom"]].set_visible(False)
            continue
        ax.axvline(0, color="#555555", linestyle="--", linewidth=.8)
        for index, (method, reference) in enumerate(MECHANISMS):
            row = effect_lookup.get((dataset, method, reference))
            if row is None or row["mean_delta_pp"] is None:
                ax.text(.98, index, "incomplete", transform=ax.get_yaxis_transform(), ha="right", va="center", color="#888888", fontsize=8)
                continue
            mean = row["mean_delta_pp"]
            half = mean - row["ci95_low_pp"] if row["ci95_low_pp"] is not None else 0
            ax.errorbar(mean, index, xerr=half, fmt="o", capsize=4, color=color,
                        markerfacecolor=color if row["status"] == "complete" else "white")
            if row["status"] != "complete":
                ax.annotate(f" n={row['n_pairs']}/20", (mean, index), xytext=(4, 5), textcoords="offset points", fontsize=7)
        ax.set_title(title); ax.set_xlabel("Paired difference (percentage points)")
        ax.set_yticks(range(len(MECHANISMS)), effect_labels); ax.tick_params(axis="y", length=0)
        ax.grid(axis="x", alpha=.2); ax.spines[["top", "right"]].set_visible(False)
    axes[0].invert_yaxis()
    fig.suptitle(("PARTIAL ANALYSIS | " if partial_figure else "") + "Matched mechanism effects | mean paired difference and t 95% interval", fontsize=13)
    fig.text(.5, .02, "Positive favors the named mechanism. Intervals describe support sampling on the fixed query split; multiple comparisons are unadjusted.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .07, 1, .94))
    for suffix in ("png", "pdf"):
        name = f"matched_mechanisms.{suffix}"; fig.savefig(OUTPUT / name, bbox_inches="tight"); artifacts.append(name)
    plt.close(fig)
    return artifacts


def write_json(path: Path, value: object):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_csv(path: Path, rows: list, fallback_fields: list):
    fields = list(dict.fromkeys(key for row in rows for key in row)) if rows else fallback_fields
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False, sort_keys=True) if isinstance(value, (list, dict)) else value for key, value in row.items()})
    temporary.replace(path)


def self_test():
    """Use deliberately synthetic percentages; never open result paths."""
    per_class, counts = [80.] * 102 + [20.] * 47, [1] * 102 + [4] * 47
    row = {"per_class": per_class, "class_counts": counts, "n": sum(counts), "macro": statistics.mean(per_class),
           "top1": sum(a * b for a, b in zip(per_class, counts)) / sum(counts)}
    metric = row_metrics(row, "joint_flowers_dtd")
    require(metric["equal_domain_macro"] == 50., "Synthetic equal-domain weighting failed")
    require(metric["flowers_macro"] == metric["flowers_top1"] == 80. and metric["dtd_macro"] == metric["dtd_top1"] == 20., "Synthetic domain metric failed")
    require(abs(metric["macro"] - 50.) > 10 and abs(metric["top1"] - 50.) > 5, "Synthetic fixture must distinguish all aggregation rules")
    confidence = paired_stats([-1., 1.])
    require(abs(confidence["ci95_high_pp"] - 12.706204736432095) < 1e-8 and confidence["mean_delta_pp"] == 0, "Known paired t interval failed")
    require(paired_stats([1.])["ci95_low_pp"] is None and stats([1.])["sd"] is None, "n<2 must not fabricate SD/CI")
    error_rate = paired_stats([-2., -1.], higher_is_better=False)
    require(error_rate["wins"] == 2 and error_rate["losses"] == 0 and error_rate["mean_delta_pp"] == -1.5, "Lower error rates must count as improvements without reversing reported differences")
    require(stats([])["mean"] is None, "Empty estimates must be null")
    def observation(seed, value):
        return Observation("flowers102", "test", "main", "dummy", seed, {"family": "dummy"}, {"macro": value}, None, True, 1, "dummy")
    a, b = {1: observation(1, 30.), 0: observation(0, 10.)}, {0: observation(0, 8.), 1: observation(1, 25.)}
    paired = paired_record(a, b, [0, 1], "macro", "macro", False)
    require(paired["seed_deltas_pp"] == {"0": 2., "1": 5.} and paired["mean_delta_pp"] == 3.5, "Seed-wise pairing failed")
    incomplete = paired_record(a, {0: b[0]}, [0, 1], "macro", "macro", False)
    require(incomplete["status"] == "partial" and incomplete["mean_delta_pp"] is None and incomplete["missing_reference_seeds"] == [1], "Partial pair must not masquerade as complete")
    partial = paired_record(a, {0: b[0]}, [0, 1], "macro", "macro", True)
    require(partial["mean_delta_pp"] == 2. and partial["ci95_low_pp"] is None, "Explicit partial behavior failed")
    require(list(seed_groups("confirmation")) == ["primary4"], "Confirmation must not imply 20 registered seeds")
    require(list(seed_groups("test_10shot")) == ["primary4"] and len(TEN_SHOT_FAMILIES) == 8, "Optional 10-shot scope must stay separate from main20")
    require("aircraft" not in DATASETS and CLASS_COUNTS["aircraft"] == 100, "Optional Aircraft must not become a mandatory main endpoint")
    require(set(PRIMARY).isdisjoint(SUPPLEMENTARY) and len(SEED_GROUPS["all20"]) == 20, "Seed group definitions failed")
    print("PASS: synthetic joint weighting, percentage-point units, paired t95 CI, error-rate direction, seed alignment, partial-data guards, confirmation, optional10shot and Aircraft scope; no results read")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("all", "test", "confirmation"), default="all")
    parser.add_argument("--allow-partial", action="store_true", help="Explicitly include incomplete observed-seed estimates; they remain marked partial")
    parser.add_argument("--self-test", action="store_true", help="Run only in-memory synthetic checks; no experiment files are opened")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    audit = Audit()
    scopes = ([] if args.phase == "test" else [("flowers102", "confirmation")])
    if args.phase != "confirmation":
        scopes += [(dataset, "test") for dataset in DATASETS]
    observations, contexts, frozen = [], {}, None
    try:
        frozen = protocol(audit)
        for dataset, phase in scopes:
            try:
                context = query_context(dataset, phase, audit)
                contexts[(dataset, phase)] = context
            except (OSError, ValueError, KeyError, TypeError) as error:
                audit.issue("invalid_query_context", error, dataset=dataset, phase=phase)
                continue
            for study in ("main", "propagation"):
                observations.extend(load_endpoint(dataset, phase, study, frozen, context, audit))
    except (OSError, ValueError, KeyError, TypeError) as error:
        audit.issue("missing_or_invalid_final_protocol", error)
        frozen = None  # no endpoint selection fallback is allowed
    summaries = make_summaries(observations, scopes, frozen, args.allow_partial) if frozen else []
    stage_aligned = verify_stage_query_alignment(contexts, audit)
    paired = make_pairs(observations, scopes, frozen, args.allow_partial, include_stage_pairs=stage_aligned) if frozen else []
    cases = make_cases(observations, contexts, frozen) if frozen else {"status": "blocked_by_missing_or_invalid_final_protocol", "comparisons": []}
    tenshot = optional_tenshot(frozen, args.allow_partial, requested=args.phase != "confirmation")
    aircraft = optional_aircraft(frozen, args.allow_partial, requested=args.phase != "confirmation")
    for relative, expected in list(audit.files.items()):
        if not (ROOT / relative).is_file() or sha256(ROOT / relative) != expected:
            audit.issue("input_changed_during_analysis", relative)
    complete = frozen is not None and not audit.issues and all(row["status"] == "complete" for row in summaries)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    write_csv(OUTPUT / "summary.csv", summaries, ["dataset", "phase", "seed_group", "family", "status", "n_seeds", "expected_n_seeds", "macro_mean", "macro_sd", "top1_mean", "top1_sd"])
    write_csv(OUTPUT / "paired.csv", paired, ["dataset", "phase", "seed_group", "method", "reference", "metric", "status", "n_pairs", "mean_delta_pp", "ci95_low_pp", "ci95_high_pp"])
    write_csv(OUTPUT / "summary_10shot.csv", tenshot["summaries"], ["dataset", "phase", "shot", "seed_group", "family", "status", "n_seeds", "macro_mean", "macro_sd", "top1_mean", "top1_sd"])
    write_csv(OUTPUT / "paired_10shot.csv", tenshot["paired_comparisons"], ["dataset", "phase", "shot", "method", "reference", "metric", "status", "n_pairs", "mean_delta_pp", "ci95_low_pp", "ci95_high_pp"])
    write_csv(OUTPUT / "summary_aircraft.csv", aircraft["summaries"], ["dataset", "phase", "shot", "seed_group", "family", "status", "n_seeds", "macro_mean", "macro_sd", "top1_mean", "top1_sd"])
    write_csv(OUTPUT / "paired_aircraft.csv", aircraft["paired_comparisons"], ["dataset", "phase", "shot", "seed_group", "method", "reference", "metric", "status", "n_pairs", "mean_delta_pp", "ci95_low_pp", "ci95_high_pp"])
    write_json(OUTPUT / "cases.json", cases)
    figures = main_figures(summaries, paired, frozen) if frozen and args.phase != "confirmation" else []
    invalid = any(issue["code"].startswith("invalid") or issue["code"] in ("missing_or_invalid_final_protocol", "input_changed_during_analysis") for issue in audit.issues)
    analysis = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "status": "complete" if complete else "invalid" if invalid else "partial",
        "requested_phase": args.phase, "allow_partial_statistics": args.allow_partial,
        "scope": [{"dataset": dataset, "phase": phase} for dataset, phase in scopes],
        "development_choices": frozen["choices"] if frozen else None,
        "matched_config_validation": frozen["matched_config_validation"] if frozen else [],
        "method_order": frozen["family_order"] if frozen else [],
        "configuration_selection": "Each family uses its frozen development-selected configuration or pre-specified matched control; no confirmation/test sorting or winner selection",
        "metric_definitions": {
            "units": "Accuracy and error rates are percentages (0..100); all differences and CI bounds are percentage points",
            "joint_equal_domain_macro": "0.5 * mean(per_class[0:102]) + 0.5 * mean(per_class[102:149]); primary joint metric",
            "joint_overall_macro": "mean(per_class[0:149]); reported as macro, distinct from equal-domain macro",
            "domain_top1": "Class-count-weighted accuracy within each true domain, checked against prediction CSV",
            "cross_domain_error_rate": "100 * cross-domain predictions / all joint queries; this is sample-weighted",
            "flowers_to_dtd_rate": "100 * true Flowers queries predicted as DTD / all true Flowers queries",
            "dtd_to_flowers_rate": "100 * true DTD queries predicted as Flowers / all true DTD queries",
            "class_array_order": "numeric labels, not candidate text row order",
        },
        "seed_groups": SEED_GROUPS,
        "seed_group_interpretation": "Primary4 were used in development selection; supplementary16 are additional support draws with frozen configurations. All20 combines these groups and is not 20 independent confirmation seeds. All groups reuse the same fixed query images.",
        "ci_scope": CI_SCOPE,
        "endpoint_interpretation": "Flowers round2 is exploratory because round1 test and validation were already observed. DTD evaluates transfer of unchanged Flowers-selected parameters; DTD final results never select the headline method.",
        "partial_policy": "Missing/invalid endpoints and duplicate/config-mismatched rows are reported. Without --allow-partial, incomplete group means/SDs/CIs are null. Explicit partial pairs list every missing seed; they are never labeled complete20.",
        "timing_policy": "seconds_including_cache_reuse is not used to rank runtime because memo reuse and configuration order affect it",
        "summaries": summaries, "paired_comparisons": paired, "optional_10shot": tenshot, "optional_aircraft": aircraft,
        "stage_flowers_query_alignment_verified": stage_aligned, "issues": audit.issues,
        "input_sha256": audit.files, "script_sha256": sha256(Path(__file__)),
        "artifacts": ["summary.csv", "paired.csv", "summary_10shot.csv", "paired_10shot.csv", "summary_aircraft.csv", "paired_aircraft.csv", "cases.json", *figures],
        "classification_scores_computed_by_analysis": False,
    }
    write_json(OUTPUT / "analysis.json", analysis)
    print(json.dumps({"status": analysis["status"], "observations": len(observations), "summary_rows": len(summaries), "paired_rows": len(paired), "issue_count": len(audit.issues), "output": str(OUTPUT)}, indent=2))
    return 0 if complete else 2


if __name__ == "__main__":
    sys.exit(main())
