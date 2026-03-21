"""Helpers for reporting metrics with ALL as the positive class."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score


DEFAULT_THRESHOLD_SWEEP = np.arange(0.35, 0.76, 0.05)


def to_all_positive_targets(targets: np.ndarray) -> np.ndarray:
    """Convert project labels (ALL=0, HEM=1) to sklearn-style ALL-positive labels."""
    targets = np.asarray(targets, dtype=np.int64)
    return (targets == 0).astype(np.int64)


def compute_all_positive_metrics(
    targets: np.ndarray,
    all_scores: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Compute binary metrics with ALL treated as the positive class."""
    all_positive_targets = to_all_positive_targets(targets)
    all_scores = np.asarray(all_scores, dtype=np.float32)
    preds_all_positive = (all_scores > threshold).astype(np.int64)

    tn, fp, fn, tp = confusion_matrix(
        all_positive_targets, preds_all_positive, labels=[0, 1]
    ).ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    acc = (preds_all_positive == all_positive_targets).mean()
    f1 = f1_score(all_positive_targets, preds_all_positive, zero_division=0)

    try:
        auc = roc_auc_score(all_positive_targets, all_scores)
    except ValueError:
        auc = float("nan")

    return {
        "threshold": float(threshold),
        "acc": float(acc),
        "auc": float(auc),
        "sens": float(sensitivity),
        "spec": float(specificity),
        "f1": float(f1),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def sweep_all_positive_thresholds(
    targets: np.ndarray,
    all_scores: np.ndarray,
    thresholds: np.ndarray | None = None,
) -> dict[str, float]:
    """Find the Youden-J-optimal threshold for ALL-positive scoring."""
    thresholds = DEFAULT_THRESHOLD_SWEEP if thresholds is None else np.asarray(thresholds)

    best_metrics: dict[str, float] | None = None
    best_j = -float("inf")

    for threshold in thresholds:
        metrics = compute_all_positive_metrics(targets, all_scores, float(threshold))
        j_score = metrics["sens"] + metrics["spec"] - 1.0
        if j_score > best_j:
            best_j = j_score
            best_metrics = metrics

    if best_metrics is None:
        raise ValueError("No thresholds provided for metric sweep.")

    best_metrics = best_metrics.copy()
    best_metrics["j"] = float(best_j)
    return best_metrics
