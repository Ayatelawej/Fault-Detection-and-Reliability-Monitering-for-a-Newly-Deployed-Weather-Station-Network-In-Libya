from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score


def binary_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray | None = None,
    threshold: float | None = None,
    y_pred: np.ndarray | None = None,
) -> dict[str, float]:
    if y_pred is None:
        if y_score is None or threshold is None:
            raise ValueError("score and threshold required")
        y_pred = y_score >= threshold
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    if y_score is None or len(np.unique(y_true)) < 2:
        pr_auc = np.nan
    else:
        pr_auc = float(average_precision_score(y_true, y_score))
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "pr_auc": pr_auc,
    }


def choose_threshold_max_f1(y_true: np.ndarray, y_score: np.ndarray) -> float:
    scores = np.unique(np.asarray(y_score, dtype=float))
    if len(scores) == 0:
        return 0.5
    best_threshold = float(scores[0])
    best_f1 = -1.0
    for threshold in sorted(scores, reverse=True):
        metrics = binary_metrics(y_true, y_score, float(threshold))
        score = metrics["f1"]
        if score > best_f1 or (score == best_f1 and threshold > best_threshold):
            best_f1 = score
            best_threshold = float(threshold)
    return best_threshold
