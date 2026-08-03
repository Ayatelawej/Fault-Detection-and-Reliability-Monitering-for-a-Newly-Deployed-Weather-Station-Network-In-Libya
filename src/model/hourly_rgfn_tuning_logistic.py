from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import warnings

import joblib
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression

from src.model.hourly_baseline import HourlyLogisticScaler, binary_metrics
from src.model.hourly_calibration import CALIBRATION_THRESHOLDS, select_operating_point, target_check


DEFAULT_LOGISTIC_WEIGHTS = (1.0, 2.0, 4.0, 6.0, 8.0)
DEFAULT_LOGISTIC_THRESHOLDS = CALIBRATION_THRESHOLDS
_METRIC_NAMES = ("precision", "recall", "f1", "accuracy", "tp", "fp", "fn", "tn")


@dataclass(frozen=True)
class HourlyLogisticTuningConfig:
    seed: int = 0
    max_iter: int = 2000
    c_value: float = 1.0
    tolerance: float = 1e-4


def _names(examples: dict[str, np.ndarray], key: str, width: int, prefix: str) -> list[str]:
    values = examples.get(key)
    if values is None:
        return [f"{prefix}_{index}" for index in range(width)]
    names = [str(value) for value in np.asarray(values, dtype=object).tolist()]
    if len(names) != width:
        raise ValueError(f"hourly logistic {key} width does not match its tensor")
    return names


def flatten_hourly_logistic_features(examples: dict[str, np.ndarray]) -> tuple[np.ndarray, list[str]]:
    required = ("X_cont", "mask", "time_since_last", "static", "rule_evidence", "y_binary")
    missing = sorted(set(required).difference(examples))
    if missing:
        raise KeyError(f"hourly logistic examples lack fields: {missing}")
    continuous = np.asarray(examples["X_cont"], dtype=np.float32)
    mask = np.asarray(examples["mask"], dtype=np.float32)
    elapsed = np.asarray(examples["time_since_last"], dtype=np.float32)
    static = np.asarray(examples["static"], dtype=np.float32)
    rules = np.asarray(examples["rule_evidence"], dtype=np.float32)
    labels = np.asarray(examples["y_binary"], dtype=int)
    if continuous.ndim != 3 or continuous.shape[1] != 7:
        raise ValueError("hourly logistic requires a seven-hour continuous tensor")
    if mask.shape != (continuous.shape[0], continuous.shape[1], 1):
        raise ValueError("hourly logistic mask shape does not match continuous data")
    if elapsed.shape != (continuous.shape[0], continuous.shape[1], 1):
        raise ValueError("hourly logistic elapsed shape does not match continuous data")
    if static.ndim != 2 or rules.ndim != 2:
        raise ValueError("hourly logistic static and rule arrays must be two-dimensional")
    if static.shape[0] != continuous.shape[0] or rules.shape[0] != continuous.shape[0] or len(labels) != continuous.shape[0]:
        raise ValueError("hourly logistic sample dimensions do not agree")
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("hourly logistic requires binary eligible labels")
    temporal = np.concatenate((continuous, mask, elapsed), axis=2).reshape(continuous.shape[0], -1)
    values = np.concatenate((temporal, rules, static), axis=1).astype(np.float32)
    continuous_names = _names(examples, "continuous_feature_names", continuous.shape[2], "continuous")
    static_names = _names(examples, "static_feature_names", static.shape[1], "static")
    rule_names = _names(examples, "rule_evidence_feature_names", rules.shape[1], "rule")
    names: list[str] = []
    for hour_index in range(continuous.shape[1]):
        hours_ago = continuous.shape[1] - hour_index - 1
        suffix = "t0" if hours_ago == 0 else f"tminus_{hours_ago}h"
        names.extend([f"{name}_{suffix}" for name in continuous_names])
        names.append(f"row_present_{suffix}")
        names.append(f"time_since_last_{suffix}")
    names.extend(rule_names)
    names.extend(static_names)
    if values.shape[1] != len(names):
        raise RuntimeError("hourly logistic feature names do not match flattened values")
    return np.ascontiguousarray(values, dtype=np.float32), names


def _validate_splits(splits: dict[str, np.ndarray], labels: np.ndarray) -> dict[str, np.ndarray]:
    required = ("train", "validation", "test")
    if set(splits) != set(required):
        raise ValueError("hourly logistic requires train, validation, and test partitions")
    result = {name: np.asarray(splits[name], dtype=np.int64) for name in required}
    combined = np.concatenate([result[name] for name in required])
    if len(combined) != len(labels) or len(np.unique(combined)) != len(labels):
        raise ValueError("hourly logistic partitions must be disjoint and complete")
    if set(combined.tolist()) != set(range(len(labels))):
        raise ValueError("hourly logistic partitions do not cover every example")
    for name in ("train", "validation"):
        partition_labels = labels[result[name]]
        if not np.equal(partition_labels, 0).any() or not np.equal(partition_labels, 1).any():
            raise ValueError(f"hourly logistic {name} partition lacks a binary class")
    if not len(result["test"]):
        raise ValueError("hourly logistic test partition is empty")
    return result


def _fit_candidate(
    values: np.ndarray,
    labels: np.ndarray,
    splits: dict[str, np.ndarray],
    fault_class_weight: float,
    config: HourlyLogisticTuningConfig,
) -> tuple[LogisticRegression, HourlyLogisticScaler, np.ndarray]:
    scaler = HourlyLogisticScaler.fit(values, splits["train"])
    train_values = scaler.transform(values[splits["train"]])
    validation_values = scaler.transform(values[splits["validation"]])
    model = LogisticRegression(
        class_weight={0: 1.0, 1: float(fault_class_weight)},
        max_iter=int(config.max_iter),
        C=float(config.c_value),
        tol=float(config.tolerance),
        solver="lbfgs",
        random_state=int(config.seed),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        model.fit(train_values, labels[splits["train"]])
    probabilities = model.predict_proba(validation_values)[:, 1].astype(np.float32)
    return model, scaler, probabilities


def _summary(metric: dict[str, object]) -> dict[str, object]:
    result: dict[str, object] = {"runs": 1}
    for name in _METRIC_NAMES:
        result[name] = metric[name]
        result[f"{name}_std"] = float("nan")
    return result


def save_hourly_logistic_model(
    path: Path,
    model: LogisticRegression,
    scaler: HourlyLogisticScaler,
    feature_names: list[str],
    config: HourlyLogisticTuningConfig,
    selection: dict[str, object],
) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "estimator": model,
            "scaler": scaler,
            "scaler_payload": scaler.payload(),
            "feature_names": list(feature_names),
            "window_hours": 7,
            "feature_dimension": int(len(feature_names)),
            "configuration": asdict(config),
            "selection": selection,
            "parameter_count": int(model.coef_.size + model.intercept_.size),
        },
        destination,
    )
    return destination


def train_hourly_logistic_variant(
    examples: dict[str, np.ndarray],
    splits: dict[str, np.ndarray],
    split_name: str,
    model_dir: Path | None = None,
    weights: tuple[float, ...] = DEFAULT_LOGISTIC_WEIGHTS,
    thresholds: tuple[float, ...] = DEFAULT_LOGISTIC_THRESHOLDS,
    config: HourlyLogisticTuningConfig | None = None,
) -> dict[str, object]:
    if not weights or not thresholds:
        raise ValueError("hourly logistic selection requires weights and thresholds")
    weight_values = tuple(float(value) for value in weights)
    threshold_values = tuple(float(value) for value in thresholds)
    if any(value <= 0.0 for value in weight_values) or any(value <= 0.0 or value >= 1.0 for value in threshold_values):
        raise ValueError("hourly logistic weights and thresholds must be in valid ranges")
    tuning = HourlyLogisticTuningConfig() if config is None else config
    if tuning.max_iter < 1 or tuning.c_value <= 0.0 or tuning.tolerance <= 0.0:
        raise ValueError("hourly logistic configuration values must be positive")
    values, feature_names = flatten_hourly_logistic_features(examples)
    labels = np.asarray(examples["y_binary"], dtype=int)
    partitions = _validate_splits(splits, labels)
    candidates: dict[float, tuple[LogisticRegression, HourlyLogisticScaler]] = {}
    validation_rows: list[dict[str, object]] = []
    validation_labels = labels[partitions["validation"]]
    for weight in weight_values:
        model, scaler, probabilities = _fit_candidate(values, labels, partitions, weight, tuning)
        candidates[weight] = (model, scaler)
        for threshold in threshold_values:
            metric = binary_metrics(validation_labels, probabilities, threshold)
            validation_rows.append(
                {
                    "seed": int(tuning.seed),
                    "fault_class_weight": weight,
                    "threshold": threshold,
                    "validation": metric,
                    "validation_minimum_metric": float(min(metric[name] for name in ("precision", "recall", "f1"))),
                }
            )
    best_balanced = select_operating_point(validation_rows, "balanced")
    max_f1 = select_operating_point(validation_rows, "f1")
    selected_weight = float(best_balanced["fault_class_weight"])
    selected_threshold = float(best_balanced["threshold"])
    selected_model, selected_scaler = candidates[selected_weight]
    selected_validation_metric = next(
        row["validation"]
        for row in validation_rows
        if float(row["fault_class_weight"]) == selected_weight and float(row["threshold"]) == selected_threshold
    )
    test_values = selected_scaler.transform(values[partitions["test"]])
    test_probabilities = selected_model.predict_proba(test_values)[:, 1].astype(np.float32)
    test_metric = binary_metrics(labels[partitions["test"]], test_probabilities, selected_threshold)
    selection = {
        "split": str(split_name),
        "fault_class_weight": selected_weight,
        "threshold": selected_threshold,
        "validation": selected_validation_metric,
        "test": test_metric,
    }
    model_path = None
    if model_dir is not None:
        destination = Path(model_dir) / f"hourly_logistic_{split_name}.joblib"
        model_path = str(
            save_hourly_logistic_model(
                destination,
                selected_model,
                selected_scaler,
                feature_names,
                tuning,
                selection,
            )
        )
    parameter_count = int(selected_model.coef_.size + selected_model.intercept_.size)
    return {
        "model": "logistic_regression",
        "split": str(split_name),
        "window_hours": 7,
        "feature_dimension": int(values.shape[1]),
        "parameter_count": parameter_count,
        "weights": list(weight_values),
        "thresholds": list(threshold_values),
        "seed_count": 1,
        "configuration": asdict(tuning),
        "validation_seed_rows": validation_rows,
        "validation_grid": validation_rows,
        "max_f1": max_f1,
        "best_balanced": best_balanced,
        "selected_validation_seed_rows": [
            {"seed": int(tuning.seed), **selected_validation_metric}
        ],
        "selected_validation_summary": _summary(selected_validation_metric),
        "test_seed_rows": [{"seed": int(tuning.seed), **test_metric}],
        "test_summary": _summary(test_metric),
        "final_test_target_check": target_check(test_metric),
        "test_evaluation_count": 1,
        "test_evaluation_count_per_seed": 1,
        "selection_source": "validation",
        "model_path": model_path,
    }
