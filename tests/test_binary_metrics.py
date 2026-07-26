import numpy as np

from src.model.binary_metrics import binary_metrics, choose_threshold_max_f1


def test_binary_metrics_reports_expected_scores():
    metrics = binary_metrics(
        np.asarray([1, 1, 0, 0]),
        np.asarray([0.9, 0.1, 0.8, 0.7]),
        threshold=0.75,
    )

    assert metrics["precision"] == 0.5
    assert metrics["recall"] == 0.5
    assert metrics["f1"] == 0.5
    assert metrics["pr_auc"] == 0.75


def test_choose_threshold_max_f1_prefers_higher_tied_threshold():
    threshold = choose_threshold_max_f1(
        np.asarray([0, 0]),
        np.asarray([0.2, 0.8]),
    )

    assert threshold == 0.8


def test_binary_metrics_handles_single_class_and_explicit_predictions():
    metrics = binary_metrics(
        np.asarray([0, 0]),
        y_pred=np.asarray([0, 1]),
    )

    assert metrics["precision"] == 0.0
    assert metrics["recall"] == 0.0
    assert metrics["f1"] == 0.0
    assert np.isnan(metrics["pr_auc"])
