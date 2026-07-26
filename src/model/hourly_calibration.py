from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from src.model.hourly_baseline import (
    HourlyBaselineConfig,
    binary_metrics,
    make_classifier,
)


CALIBRATION_WEIGHTS = (1.0, 2.0, 4.0, 6.0, 8.0, 10.0, 13.0)
CALIBRATION_THRESHOLDS = tuple(float(value) for value in np.round(np.arange(0.30, 0.8001, 0.05), 2))
METRIC_NAMES = ("precision", "recall", "f1")


def minimum_metric(metric: dict[str, object]) -> float:
    return float(min(float(metric[name]) for name in METRIC_NAMES))


def _selection_key(row: dict[str, object], criterion: str) -> tuple[float, ...]:
    metric = row["validation"]
    if criterion == "f1":
        first = float(metric["f1"])
        second = minimum_metric(metric)
    elif criterion == "balanced":
        first = minimum_metric(metric)
        second = float(metric["f1"])
    else:
        raise KeyError(criterion)
    return (
        first,
        second,
        float(metric["precision"]),
        float(metric["recall"]),
        -float(row["fault_class_weight"]),
        -float(row["threshold"]),
    )


def select_operating_point(
    rows: list[dict[str, object]],
    criterion: str,
) -> dict[str, object]:
    if not rows:
        raise ValueError("calibration selection requires validation rows")
    return max(rows, key=lambda row: _selection_key(row, criterion)).copy()


def target_check(metric: dict[str, object], target: float = 0.90) -> dict[str, object]:
    values = {name: float(metric[name]) for name in METRIC_NAMES}
    gaps = {name: float(max(0.0, target - value)) for name, value in values.items()}
    binding = min(values, key=values.get)
    return {
        "target": float(target),
        "achieved": bool(all(value >= target for value in values.values())),
        "shortfall": gaps,
        "binding_constraint": binding,
    }


def _validate_calibration_splits(
    splits: dict[str, np.ndarray],
    labels: np.ndarray,
) -> None:
    required = ("train", "validation", "test")
    if set(splits) != set(required):
        raise ValueError("calibration requires exactly train, validation, and test partitions")
    pieces = [np.asarray(splits[name], dtype=np.int64) for name in required]
    combined = np.concatenate(pieces)
    if len(combined) != len(labels) or len(np.unique(combined)) != len(labels) or set(combined.tolist()) != set(range(len(labels))):
        raise ValueError("calibration split membership is not disjoint and complete")
    for name in ("train", "validation"):
        values = labels[np.asarray(splits[name], dtype=np.int64)]
        if not len(values) or not np.equal(values, 0).any() or not np.equal(values, 1).any():
            raise ValueError(f"calibration {name} partition does not contain both binary classes")
    if not len(splits["test"]):
        raise ValueError("calibration test partition is empty")


def calibrate_split(
    values: np.ndarray,
    labels: np.ndarray,
    splits: dict[str, np.ndarray],
    base_config: HourlyBaselineConfig,
    weights: tuple[float, ...] = CALIBRATION_WEIGHTS,
    thresholds: tuple[float, ...] = CALIBRATION_THRESHOLDS,
    model_factory=make_classifier,
) -> dict[str, object]:
    labels = np.asarray(labels, dtype=int)
    _validate_calibration_splits(splits, labels)
    rows: list[dict[str, object]] = []
    models: dict[float, object] = {}
    for weight in weights:
        weight_value = float(weight)
        config = replace(base_config, fault_class_weight=weight_value)
        model = model_factory(config)
        model.fit(values[splits["train"]], labels[splits["train"]])
        probabilities = model.predict_proba(values[splits["validation"]])[:, 1]
        models[weight_value] = model
        for threshold in thresholds:
            threshold_value = float(threshold)
            metric = binary_metrics(labels[splits["validation"]], probabilities, threshold_value)
            rows.append(
                {
                    "fault_class_weight": weight_value,
                    "threshold": threshold_value,
                    "validation": metric,
                    "validation_minimum_metric": minimum_metric(metric),
                }
            )
    max_f1 = select_operating_point(rows, "f1")
    best_balanced = select_operating_point(rows, "balanced")
    selected_weight = float(best_balanced["fault_class_weight"])
    selected_model = models[selected_weight]
    test_probabilities = selected_model.predict_proba(values[splits["test"]])[:, 1]
    final_test = binary_metrics(labels[splits["test"]], test_probabilities, float(best_balanced["threshold"]))
    return {
        "validation_grid": rows,
        "max_f1": max_f1,
        "best_balanced": best_balanced,
        "final_test": final_test,
        "final_test_target_check": target_check(final_test),
        "selection_source": "validation",
        "test_evaluation_count": 1,
        "weight_count": int(len(weights)),
        "threshold_count": int(len(thresholds)),
    }


def validation_grid_frame(result: dict[str, object]) -> pd.DataFrame:
    rows = []
    for row in result["validation_grid"]:
        metric = row["validation"]
        rows.append(
            {
                "fault_class_weight": row["fault_class_weight"],
                "threshold": row["threshold"],
                "precision": metric["precision"],
                "recall": metric["recall"],
                "f1": metric["f1"],
                "minimum_metric": row["validation_minimum_metric"],
            }
        )
    return pd.DataFrame(rows).sort_values(["fault_class_weight", "threshold"]).reset_index(drop=True)


def f1_surface(result: dict[str, object]) -> pd.DataFrame:
    frame = validation_grid_frame(result)
    return frame.pivot(
        index="fault_class_weight",
        columns="threshold",
        values="f1",
    ).round(6)


def operating_point_frame(row: dict[str, object]) -> pd.DataFrame:
    metric = row["validation"]
    return pd.DataFrame(
        [
            {
                "fault_class_weight": row["fault_class_weight"],
                "threshold": row["threshold"],
                "precision": metric["precision"],
                "recall": metric["recall"],
                "f1": metric["f1"],
                "minimum_metric": row["validation_minimum_metric"],
            }
        ]
    )


def calibration_report(results: dict[str, dict[str, object]]) -> str:
    parts = [
        "SHORT-WINDOW BINARY CALIBRATION",
        "",
        "SELECTION RULE",
        "All operating points are selected from validation metrics only.",
        "The balanced point maximizes the minimum of precision, recall, and F1, with deterministic tie resolution.",
        "",
    ]
    for split_name, result in results.items():
        final_test = result["final_test"]
        target = result["final_test_target_check"]
        parts.extend(
            [
                f"SPLIT {split_name}",
                "VALIDATION F1 SURFACE",
                f1_surface(result).to_string(),
                "",
                "MAXIMUM VALIDATION F1 POINT",
                operating_point_frame(result["max_f1"]).to_string(index=False),
                "",
                "MOST BALANCED VALIDATION POINT",
                operating_point_frame(result["best_balanced"]).to_string(index=False),
                "",
                "FINAL TEST METRICS",
                pd.DataFrame(
                    [
                        {
                            "precision": final_test["precision"],
                            "recall": final_test["recall"],
                            "f1": final_test["f1"],
                            "accuracy": final_test["accuracy"],
                        }
                    ]
                ).to_string(index=False),
                "FINAL TEST CONFUSION MATRIX",
                pd.DataFrame(
                    [{"tp": final_test["tp"], "fp": final_test["fp"], "fn": final_test["fn"], "tn": final_test["tn"]}]
                ).to_string(index=False),
                f"target_90_90_90_achieved={target['achieved']}",
                f"precision_shortfall={target['shortfall']['precision']:.6f}",
                f"recall_shortfall={target['shortfall']['recall']:.6f}",
                f"f1_shortfall={target['shortfall']['f1']:.6f}",
                f"binding_constraint={target['binding_constraint']}",
                f"test_evaluation_count={result['test_evaluation_count']}",
                "",
            ]
        )
    return "\n".join(parts)
