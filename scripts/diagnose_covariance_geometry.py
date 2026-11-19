"""CPU, support-only covariance geometry diagnostic; no query scoring."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/part_a_round2/covariance_geometry.json"
DATASETS = ("flowers102", "dtd", "joint_flowers_dtd")
SEED, SHOT = 0, 5


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def normalize(x):
    return x / (x.norm(dim=-1, keepdim=True) + 1e-10)


def summary(x):
    return {"min": x.min().item(), "mean": x.mean().item(), "max": x.max().item()}


def spectrum(matrix):
    eig = torch.linalg.eigvalsh(matrix)
    trace = matrix.trace().item()
    squared_frobenius = matrix.square().sum().item()
    cutoff = eig[-1].item() * 1e-6
    return {
        "trace": trace,
        "participation_ratio_effective_rank": trace * trace / squared_frobenius,
        "numerical_rank_relative_1e-6": int((eig > cutoff).sum()),
        "max_eigenvalue": eig[-1].item(),
        "min_eigenvalue": eig[0].item(),
        "rank_threshold": cutoff,
        "squared_frobenius_norm": squared_frobenius,
    }


def diagnose(dataset):
    folder = ROOT / "data/manifests" / dataset
    cache_path = ROOT / "artifacts/features" / dataset / "features.pt"
    records = read_json(folder / "records.json")
    manifest_classes = read_json(folder / "classes.json")
    support_manifest = read_json(folder / "support_indices.json")
    support_ids = support_manifest["indices"][str(SEED)][str(SHOT)]
    # The cache is a combined serialization. Only selected train rows enter
    # any feature calculation; no validation/test rows are analyzed or scored.
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    classes = cache["classes"]
    assert {x["label"]: x for x in classes} == {x["label"]: x for x in manifest_classes}
    assert len({x["label"] for x in classes}) == len(classes)
    assert cache["sample_ids"] == [x["sample_id"] for x in records]
    assert cache["metadata"]["manifest_sha256"] == sha(folder / "records.json")
    assert len(support_ids) == SHOT * len(classes) == len(set(support_ids))
    lookup = {r["sample_id"]: i for i, r in enumerate(records)}
    by_label = {c["label"]: [] for c in classes}
    for sid in support_ids:
        i = lookup[sid]
        record = records[i]
        assert record["split"] == "train"
        assert int(cache["labels"][i]) == record["label"]
        assert cache["splits"][i] == "train"
        by_label[record["label"]].append(i)
    index = torch.tensor([by_label[c["label"]] for c in classes], dtype=torch.long)
    assert tuple(index.shape) == (len(classes), SHOT)
    # Text row order follows cache.classes, never a numeric-label assumption.
    raw = cache["image_features"][index].to(dtype=torch.float64, device="cpu")
    text = cache["text_features"].to(dtype=torch.float64, device="cpu")
    assert raw.shape[-1] == text.shape[-1] and text.shape[0] == len(classes)
    assert bool(torch.isfinite(raw).all() and torch.isfinite(text).all())
    if "candidate_labels" in cache:
        assert torch.as_tensor(cache["candidate_labels"]).tolist() == [c["label"] for c in classes]

    c, k, d = raw.shape
    mean = raw.mean(1)
    visual_residual = raw - mean[:, None, :]
    flat_visual = visual_residual.reshape(c * k, d)
    raw_pooled = flat_visual.T @ flat_visual / (c * (k - 1))
    fused = normalize(raw + text[:, None, :])
    fused_mean = fused.mean(1)
    prototype = normalize(fused_mean)
    ordinary_residual = fused - fused_mean[:, None, :]
    official_residual = fused - prototype[:, None, :]
    ordinary_flat = ordinary_residual.reshape(c * k, d)
    official_flat = official_residual.reshape(c * k, d)
    fused_ordinary_pooled = ordinary_flat.T @ ordinary_flat / (c * (k - 1))
    fused_official_pooled = official_flat.T @ official_flat / (c * (k - 1))
    offset = fused_mean - prototype
    mean_offset_matrix = (k / (k - 1)) * (offset.T @ offset) / c
    identity_error = (fused_official_pooled - fused_ordinary_pooled - mean_offset_matrix).abs().max().item()
    assert identity_error < 1e-12
    trace_c = ordinary_residual.square().sum((1, 2)) / (k - 1)
    trace_b = offset.square().sum(1) * k / (k - 1)

    center = mean + text
    radius = center.norm(dim=-1)
    assert bool((radius > 1e-8).all()), "Degenerate anchor; do not silently clamp this diagnostic"
    u = center / radius[:, None]
    projected_residual = visual_residual - (visual_residual * u[:, None, :]).sum(-1, keepdim=True) * u[:, None, :]
    # Exact ideal-normalization Jacobian at raw mean. The intercept uses the
    # implementation's epsilon normalization, as do the actual fused targets.
    linearized = normalize(center)[:, None, :] + projected_residual / radius[:, None, None]
    error_sq = (linearized - fused).square().sum((1, 2))
    centered_sq = ordinary_residual.square().sum((1, 2))
    relative = torch.sqrt(error_sq / centered_sq)

    result = {
        "dataset": dataset, "seed": SEED, "shot": SHOT,
        "class_count": c, "support_count": c * k, "feature_dimension": d,
        "candidate_labels_in_text_row_order": [x["label"] for x in classes],
        "support_ids_manifest_order": support_ids,
        "raw_pooled_ordinary": spectrum(raw_pooled),
        "fused_pooled_hycal_prototype_centered": spectrum(fused_official_pooled),
        "fused_pooled_ordinary": spectrum(fused_ordinary_pooled),
        "jacobian_radius": summary(radius),
        "linearization": {
            "total_squared_error": error_sq.sum().item(),
            "total_true_fused_ordinary_centered_squared_deviation": centered_sq.sum().item(),
            "relative_rms_error": torch.sqrt(error_sq.sum() / centered_sq.sum()).item(),
            "per_class_relative_rms_summary": summary(relative),
            "absolute_rms_l2_error_per_support": torch.sqrt(error_sq.sum() / (c * k)).item(),
            "true_fused_rms_l2_deviation_per_support": torch.sqrt(centered_sq.sum() / (c * k)).item(),
        },
        "empirical_centering_offset": {
            "mean_trace_ordinary_fused_covariance_C": trace_c.mean().item(),
            "mean_trace_offset_B": trace_b.mean().item(),
            "trace_B_over_trace_C_pooled": (trace_b.sum() / trace_c.sum()).item(),
            "per_class_trace_B_over_trace_C_summary": summary(trace_b / trace_c),
            "hycal_equals_ordinary_plus_offset_max_abs_error": identity_error,
        },
        "selected_input_norms": {
            "visual": summary(raw.norm(dim=-1)), "text": summary(text.norm(dim=-1)),
        },
        "checks": {"support_train_only": True, "five_per_class": True,
                   "text_label_mapping_verified": True, "finite_support_features": True,
                   "no_jacobian_radius_clamp_needed": True},
        "per_class": [
            {"label": cls["label"], "classname": cls["classname"], "global_id": cls["global_id"],
             "domain": cls.get("domain", dataset),
             "support_ids": [records[i]["sample_id"] for i in index[j].tolist()],
             "jacobian_radius": radius[j].item(), "relative_rms_error": relative[j].item(),
             "trace_ordinary_fused_C": trace_c[j].item(), "trace_offset_B": trace_b[j].item(),
             "trace_B_over_trace_C": (trace_b[j] / trace_c[j]).item()}
            for j, cls in enumerate(classes)
        ],
        "input_sha256": {
            str(path.relative_to(ROOT)): sha(path)
            for path in [cache_path, folder / "records.json", folder / "classes.json", folder / "support_indices.json"]
        },
    }
    selected_by_global = {cls["global_id"]: (raw[j], text[j]) for j, cls in enumerate(classes)}
    return result, {"raw": raw_pooled, "fused": fused_official_pooled}, selected_by_global


@torch.no_grad()
def main():
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    results, matrices, selected = {}, {}, {}
    for dataset in DATASETS:
        results[dataset], matrices[dataset], selected[dataset] = diagnose(dataset)
    c1, c2 = results["flowers102"]["class_count"], results["dtd"]["class_count"]
    joint_checks = {}
    for name in ("raw", "fused"):
        expected = (c1 * matrices["flowers102"][name] + c2 * matrices["dtd"][name]) / (c1 + c2)
        error = (expected - matrices["joint_flowers_dtd"][name]).abs().max().item()
        assert error < 1e-12
        joint_checks[name + "_class_count_weighted_pool_max_abs_error"] = error
    for source in ("flowers102", "dtd"):
        for global_id, (raw, text) in selected[source].items():
            joint_raw, joint_text = selected["joint_flowers_dtd"][global_id]
            assert torch.equal(raw, joint_raw) and torch.equal(text, joint_text)
    joint_checks["joint_support_and_text_features_equal_source_selection"] = True
    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Fixed seed0 5-shot TRAIN support only; no query, validation, confirmation, or test scoring.",
        "device": "cpu", "diagnostic_dtype": "float64", "cached_feature_dtype": "float32",
        "torch_version": torch.__version__, "cpu_threads": 2,
        "formulas": {
            "raw_pooled": "sum_ci (v_ci-m_c)(v_ci-m_c)^T / [C*(K-1)]",
            "hycal_fused_pooled": "sum_ci (g_c(v_ci)-p_c)(g_c(v_ci)-p_c)^T / [C*(K-1)]",
            "participation_ratio": "trace(S)^2 / sum_ab S_ab^2 = trace(S)^2 / trace(S^2) for symmetric S",
            "numerical_rank": "count(eigenvalue > max_eigenvalue * 1e-6), with float64 eigvalsh",
            "linearization": "g_c(m_c) + [I-u_c*u_c^T]*(v_ci-m_c)/r_c",
            "relative_rms": "sqrt(sum_ci ||linearization-g_c(v_ci)||^2 / sum_ci ||g_c(v_ci)-mean_i g_c(v_ci)||^2)",
            "offset": "B_c=K/(K-1)*(fused_mean_c-p_c)*(fused_mean_c-p_c)^T",
        },
        "limitations": [
            "Describes the fixed empirical support only, not recovery of true population covariance.",
            "In-sample linearization error is descriptive, not a held-out accuracy or covariance-estimation guarantee.",
            "Full cache serialization was loaded on CPU; feature calculations selected train support rows only.",
            "Uses float64 diagnostics on cached float32 features, not an additional official numerical-equivalence check.",
            "Ideal normalization Jacobian; actual fused/intercept/prototype normalization retains norm+1e-10.",
        ],
        "joint_checks": joint_checks, "datasets": results,
        "diagnostic_script_sha256": sha(Path(__file__)),
    }
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({name: {"raw": value["raw_pooled_ordinary"],
                             "fused": value["fused_pooled_hycal_prototype_centered"],
                             "r": value["jacobian_radius"],
                             "relative_rms": value["linearization"]["relative_rms_error"],
                             "offset_ratio": value["empirical_centering_offset"]["trace_B_over_trace_C_pooled"]}
                      for name, value in results.items()}, indent=2))


if __name__ == "__main__":
    main()
