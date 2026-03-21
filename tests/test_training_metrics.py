import numpy as np

from src.utils.training_metrics import (
    compute_all_positive_metrics,
    sweep_all_positive_thresholds,
    to_all_positive_targets,
)


def test_to_all_positive_targets_flips_project_encoding():
    targets = np.array([0, 1, 0, 1], dtype=np.int64)
    assert np.array_equal(to_all_positive_targets(targets), np.array([1, 0, 1, 0]))


def test_compute_all_positive_metrics_uses_all_score():
    targets = np.array([0, 0, 1, 1], dtype=np.int64)
    all_scores = np.array([0.95, 0.80, 0.20, 0.05], dtype=np.float32)

    metrics = compute_all_positive_metrics(targets, all_scores, threshold=0.5)

    assert metrics["acc"] == 1.0
    assert metrics["auc"] == 1.0
    assert metrics["sens"] == 1.0
    assert metrics["spec"] == 1.0
    assert metrics["f1"] == 1.0
    assert (metrics["tn"], metrics["fp"], metrics["fn"], metrics["tp"]) == (2, 0, 0, 2)


def test_threshold_sweep_returns_best_all_positive_operating_point():
    targets = np.array([0, 0, 1, 1], dtype=np.int64)
    all_scores = np.array([0.70, 0.60, 0.55, 0.20], dtype=np.float32)

    metrics = sweep_all_positive_thresholds(
        targets,
        all_scores,
        thresholds=np.array([0.50, 0.55, 0.60]),
    )

    assert metrics["threshold"] == 0.55
    assert metrics["j"] == 1.0
    assert metrics["sens"] == 1.0
    assert metrics["spec"] == 1.0
