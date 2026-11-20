"""Member B: score-level reconstructions from the ChatGPT experiment log.

All arrays are [N, C] and already normalized using the upstream HyCal
Gaussian-CDF-per-query implementation. Do not use ground-truth labels to gate.
This file is a documented reconstruction, not an export from CCDS-TC2.
"""
from __future__ import annotations
import numpy as np


def top_two_margin(cosine_scores: np.ndarray) -> np.ndarray:
    """Normalized margin used by the selected Low-Margin Strong rule."""
    cosine = np.asarray(cosine_scores, dtype=np.float64)
    if cosine.ndim != 2 or cosine.shape[1] < 2:
        raise ValueError("cosine_scores must be [N, C], C >= 2")
    if not np.isfinite(cosine).all():
        raise ValueError("nonfinite cosine scores")
    ordered = np.sort(cosine, axis=1)
    margin = (ordered[:, -1] - ordered[:, -2]) / (
        ordered[:, -1] - ordered[:, 0] + 1e-8
    )
    return np.clip(margin, 0.0, 1.0)


def base_weight(shots: int, n_min_p: int = 10, scale: float = 5.0) -> float:
    """Upstream sigmoid weight, assuming every class has `shots` supports."""
    return float(1.0 / (1.0 + np.exp(-(shots - n_min_p) / scale)))


def method_scores(
    cosine_scores: np.ndarray,
    rmd_scores: np.ndarray,
    shots: int,
    reliability_by_class: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """All score-level methods used in the conversation.

    Original: enable RMD at >=10 shot; Relaxed: >=3 shot;
    Reliability: Relaxed per-class weighting by support-only reliability.
    Query-gate family: cosine + gate(query)*(relaxed - cosine).
    """
    cosine = np.asarray(cosine_scores, dtype=np.float64)
    rmd = np.asarray(rmd_scores, dtype=np.float64)
    if cosine.shape != rmd.shape or cosine.ndim != 2:
        raise ValueError("cosine and RMD must have the same [N,C] shape")
    if not (np.isfinite(cosine).all() and np.isfinite(rmd).all()):
        raise ValueError("nonfinite score array")
    if shots < 1:
        raise ValueError("shots must be positive")

    weight = base_weight(shots)
    base = cosine.copy()
    original = (1.0 - weight) * cosine + weight * rmd if shots >= 10 else base.copy()
    relaxed = (1.0 - weight) * cosine + weight * rmd if shots >= 3 else base.copy()
    delta = relaxed - cosine
    margin = top_two_margin(cosine)

    result = {
        "cosine": base,
        "original": original,
        "relaxed": relaxed,
        "uniform_075": cosine + 0.75 * delta,
        "uniform_125": cosine + 1.25 * delta,
        "uniform_150": cosine + 1.50 * delta,
        "low_margin": cosine + (1.50 - 0.75 * margin[:, None]) * delta,
        "high_margin": cosine + (0.75 + 0.75 * margin[:, None]) * delta,
        "low_margin_strong": cosine + (1.75 - margin[:, None]) * delta,
        "high_margin_strong": cosine + (0.75 + margin[:, None]) * delta,
    }
    if reliability_by_class is not None:
        reliability = np.asarray(reliability_by_class, dtype=np.float64)
        if reliability.shape != (cosine.shape[1],):
            raise ValueError("reliability must have one value per class")
        if np.any(reliability < 0) or np.any(reliability > 1):
            raise ValueError("reliability must lie in [0,1]")
        result["reliability"] = cosine + reliability[None, :] * delta
        result["uniform_matched"] = cosine + reliability.mean() * delta
    return result


def predict(scores: np.ndarray) -> np.ndarray:
    return np.asarray(scores).argmax(axis=1)
