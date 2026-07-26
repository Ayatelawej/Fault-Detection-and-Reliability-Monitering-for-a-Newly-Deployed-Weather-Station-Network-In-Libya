from __future__ import annotations

import joblib
import numpy as np
import pytest

from src.model.hourly_rgfn_tuning_logistic import (
    HourlyLogisticTuningConfig,
    flatten_hourly_logistic_features,
    train_hourly_logistic_variant,
)


def _examples(count: int = 40) -> dict[str, np.ndarray]:
    labels = np.asarray([index % 2 for index in range(count)], dtype=np.int64)
    x_cont = np.zeros((count, 7, 2), dtype=np.float32)
    for index, label in enumerate(labels):
        x_cont[index, :, 0] = float(index) / 100.0
        x_cont[index, :, 1] = float(label) * 3.0
    return {
        "X_cont": x_cont,
        "mask": np.ones((count, 7, 1), dtype=np.float32),
        "time_since_last": np.zeros((count, 7, 1), dtype=np.float32),
        "static": labels.reshape(-1, 1).astype(np.float32),
        "rule_evidence": np.column_stack((labels, 1 - labels)).astype(np.float32),
        "y_binary": labels,
        "continuous_feature_names": np.asarray(["temperature", "pressure"], dtype=object),
        "static_feature_names": np.asarray(["station_altitude"], dtype=object),
        "rule_evidence_feature_names": np.asarray(["rule_a", "rule_b"], dtype=object),
    }


def _splits() -> dict[str, np.ndarray]:
    return {
        "train": np.arange(0, 18, dtype=np.int64),
        "validation": np.arange(18, 36, dtype=np.int64),
        "test": np.arange(36, 40, dtype=np.int64),
    }


def test_flattened_features_use_all_seven_hours_and_reject_other_lengths() -> None:
    examples = _examples()
    values, names = flatten_hourly_logistic_features(examples)

    assert values.shape == (40, 7 * (2 + 2) + 2 + 1)
    assert len(names) == values.shape[1]
    assert names[0] == "temperature_tminus_6h"
    assert names[24] == "temperature_t0"

    invalid = dict(examples)
    invalid["X_cont"] = examples["X_cont"][:, :-1, :]
    invalid["mask"] = examples["mask"][:, :-1, :]
    invalid["time_since_last"] = examples["time_since_last"][:, :-1, :]
    with pytest.raises(ValueError, match="seven-hour"):
        flatten_hourly_logistic_features(invalid)


def test_logistic_selection_uses_validation_before_one_test_evaluation(tmp_path, monkeypatch) -> None:
    examples = _examples()
    examples["X_cont"][36:, :, 0] = 1_000_000.0
    observed_sizes: list[int] = []
    from sklearn.linear_model import LogisticRegression

    original = LogisticRegression.predict_proba

    def wrapped(self, values):
        observed_sizes.append(int(values.shape[0]))
        return original(self, values)

    monkeypatch.setattr(LogisticRegression, "predict_proba", wrapped)
    result = train_hourly_logistic_variant(
        examples,
        _splits(),
        "random",
        model_dir=tmp_path,
        weights=(1.0, 2.0),
        thresholds=(0.30, 0.50, 0.70),
        config=HourlyLogisticTuningConfig(max_iter=100),
    )

    assert observed_sizes.count(18) == 2
    assert observed_sizes.count(4) == 1
    assert result["test_evaluation_count"] == 1
    assert result["selection_source"] == "validation"
    assert result["best_balanced"]["fault_class_weight"] in {1.0, 2.0}
    assert result["best_balanced"]["threshold"] in {0.30, 0.50, 0.70}
    assert result["test_summary"]["runs"] == 1
    assert result["model_path"] is not None

    bundle = joblib.load(result["model_path"])
    assert bundle["window_hours"] == 7
    assert bundle["feature_dimension"] == result["feature_dimension"]
    assert float(bundle["scaler"].median[0]) < 1.0
