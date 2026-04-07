"""Verification metrics: FAR, FRR, EER, accuracy at threshold."""

from __future__ import annotations

import numpy as np


def compute_far_frr_curve(
    scores_genuine: np.ndarray,
    scores_impostor: np.ndarray,
    thresholds: np.ndarray,
    higher_is_genuine: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    For each threshold t: accept if score >= t (when higher_is_genuine).
    FRR = P(genuine rejected). FAR = P(impostor accepted).
    """
    g = np.asarray(scores_genuine, dtype=np.float64)
    i = np.asarray(scores_impostor, dtype=np.float64)
    far_list = []
    frr_list = []
    for t in thresholds:
        if higher_is_genuine:
            frr = float((g < t).mean()) if len(g) else 0.0
            far = float((i >= t).mean()) if len(i) else 0.0
        else:
            frr = float((g > t).mean()) if len(g) else 0.0
            far = float((i <= t).mean()) if len(i) else 0.0
        far_list.append(far)
        frr_list.append(frr)
    return np.array(far_list), np.array(frr_list)


def eer_and_accuracy(
    scores_genuine: np.ndarray,
    scores_impostor: np.ndarray,
    n_thresholds: int = 2000,
    higher_is_genuine: bool = True,
) -> dict[str, float]:
    all_scores = np.concatenate(
        [np.asarray(scores_genuine), np.asarray(scores_impostor)]
    )
    lo, hi = float(all_scores.min()), float(all_scores.max())
    pad = max(1e-6, (hi - lo) * 0.05)
    thresholds = np.linspace(lo - pad, hi + pad, n_thresholds)
    far, frr = compute_far_frr_curve(
        scores_genuine, scores_impostor, thresholds, higher_is_genuine
    )
    diff = np.abs(far - frr)
    j = int(np.argmin(diff))
    eer = float((far[j] + frr[j]) / 2)
    t_star = float(thresholds[j])
    # Accuracy at t_star: correct accept genuine + correct reject impostor
    g = np.asarray(scores_genuine)
    i = np.asarray(scores_impostor)
    if higher_is_genuine:
        correct_g = (g >= t_star).sum()
        correct_i = (i < t_star).sum()
    else:
        correct_g = (g <= t_star).sum()
        correct_i = (i > t_star).sum()
    n = len(g) + len(i)
    acc = float((correct_g + correct_i) / n) if n else 0.0
    return {
        "eer": eer,
        "far_at_eer": float(far[j]),
        "frr_at_eer": float(frr[j]),
        "threshold": t_star,
        "accuracy_at_eer_threshold": acc,
        "thresholds": thresholds,
        "far_curve": far,
        "frr_curve": frr,
    }
