from __future__ import annotations

from dataclasses import dataclass
from copy import deepcopy
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
import torch
from torch import nn
from torch.nn import functional as torch_functional
from torch.utils.data import DataLoader, TensorDataset

from src.availability.risk_dataset import (
    FEATURE_COLUMNS,
    INCIDENT_HAZARD_CONTEXT_FEATURE_COLUMNS,
    INCIDENT_HAZARD_EVIDENCE_FEATURE_COLUMNS,
    INCIDENT_HAZARD_SEQUENCE_WINDOW_HOURS,
    INCIDENT_HAZARD_TEMPORAL_FEATURE_COLUMNS,
)
from src.model.binary_metrics import binary_metrics, choose_threshold_max_f1

MODEL_NAMES = ("majority", "flicker", "logistic_regression", "gradient_boosted_trees")
LEARNED_MODEL_NAMES = ("logistic_regression", "gradient_boosted_trees")
FORECAST_MODEL_NAME = "hist_gradient_boosting"
FORECAST_WEIGHT_MULTIPLIERS = (0.5, 0.75, 1.0, 1.25, 1.5)
FORECAST_THRESHOLDS = tuple(
    round(float(value), 2) for value in np.arange(0.05, 1.0, 0.05)
)
FORECAST_THRESHOLD_SELECTION_RULES = (
    "maximin",
    "max_f1",
    "max_recall_precision_floor",
)
DISCRETE_HAZARD_METHODS = ("logistic_hazard", "boosted_hazard")
DISCRETE_HAZARD_THRESHOLDS = tuple(
    dict.fromkeys(
        [
            *(round(float(value), 3) for value in np.arange(0.001, 0.051, 0.001)),
            *(round(float(value), 2) for value in np.arange(0.06, 1.0, 0.01)),
        ]
    )
)
INCIDENT_HAZARD_MODEL_NAME = "causal_tcn_evidence_gate"
INCIDENT_HAZARD_WEIGHT_MULTIPLIERS = (0.5, 1.0, 2.0)
INCIDENT_HAZARD_THRESHOLDS = tuple(
    round(float(value), 2) for value in np.arange(0.01, 1.0, 0.01)
)


@dataclass
class RiskPrediction:
    model: str
    threshold: float
    prob: np.ndarray
    pred: np.ndarray


@dataclass(frozen=True)
class ForecastModelSelection:
    model: HistGradientBoostingClassifier
    positive_class_weight: float
    weight_multiplier: float
    threshold: float
    validation_probability: np.ndarray
    validation_metrics: dict[str, float]
    selection_trace: pd.DataFrame


@dataclass(frozen=True)
class DiscreteHazardModelFit:
    method: str
    model: object
    positive_class_weight: float
    feature_columns: tuple[str, ...]


@dataclass(frozen=True)
class HazardProbabilityCalibrator:
    model: LogisticRegression | None
    method: str


class ForecastTrainingTimeboxReached(RuntimeError):
    pass


def _feature_matrix(frame: pd.DataFrame, feature_columns: list[str] | None = None) -> np.ndarray:
    columns = FEATURE_COLUMNS if feature_columns is None else feature_columns
    return frame.loc[:, columns].astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=float)


def majority_predict(train_y: np.ndarray, eval_frame: pd.DataFrame) -> RiskPrediction:
    value = int(np.mean(train_y.astype(int)) >= 0.5) if len(train_y) else 0
    prob = np.full(len(eval_frame), float(value), dtype=float)
    pred = np.full(len(eval_frame), value, dtype=int)
    return RiskPrediction("majority", 0.5, prob, pred)


def flicker_predict(eval_frame: pd.DataFrame) -> RiskPrediction:
    pred = eval_frame["trailing_missing_frac_24h"].astype(float).gt(0.0).astype(int).to_numpy()
    return RiskPrediction("flicker", 0.5, pred.astype(float), pred)


def make_logistic_regression(seed: int = 2026) -> Pipeline:
    return Pipeline(
        [
            ("scale", RobustScaler()),
            ("model", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed)),
        ]
    )


def make_gradient_boosted_trees(seed: int = 2026):
    try:
        from lightgbm import LGBMClassifier
    except Exception:
        return GradientBoostingClassifier(random_state=seed)
    return LGBMClassifier(
        random_state=seed,
        n_estimators=200,
        learning_rate=0.05,
        num_leaves=15,
        subsample=0.9,
        colsample_bytree=0.9,
        class_weight="balanced",
        verbosity=-1,
    )


def fit_learned_model(name: str, train: pd.DataFrame, seed: int = 2026):
    if name == "logistic_regression":
        model = make_logistic_regression(seed)
    elif name == "gradient_boosted_trees":
        model = make_gradient_boosted_trees(seed)
    else:
        raise KeyError(name)
    model.fit(_feature_matrix(train), train["y"].to_numpy(dtype=int))
    return model


def predict_proba(model, frame: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(_feature_matrix(frame))[:, 1].astype(float)
    values = model.decision_function(_feature_matrix(frame))
    return (1.0 / (1.0 + np.exp(-values))).astype(float)


def learned_prediction(name: str, train: pd.DataFrame, validation: pd.DataFrame, test: pd.DataFrame, seed: int = 2026) -> RiskPrediction:
    model = fit_learned_model(name, train, seed)
    validation_prob = predict_proba(model, validation)
    threshold = choose_threshold_max_f1(validation["y"].to_numpy(dtype=int), validation_prob)
    test_prob = predict_proba(model, test)
    pred = test_prob >= threshold
    return RiskPrediction(name, float(threshold), test_prob, pred.astype(int))


def metric_row(y_true: np.ndarray, prediction: RiskPrediction) -> dict[str, object]:
    metrics = binary_metrics(y_true.astype(int), prediction.prob.astype(float), prediction.threshold)
    return {
        "model": prediction.model,
        "threshold": float(prediction.threshold),
        **metrics,
    }


def forecast_feature_matrix(
    frame: pd.DataFrame,
    feature_columns: tuple[str, ...] | list[str],
) -> np.ndarray:
    return (
        frame.loc[:, list(feature_columns)]
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .to_numpy(dtype=np.float32)
    )


def forecast_classification_metrics(
    y_true: np.ndarray,
    probability: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    truth = np.asarray(y_true, dtype=int)
    score = np.asarray(probability, dtype=float)
    prediction = score >= float(threshold)
    true_positive = int(((truth == 1) & prediction).sum())
    false_positive = int(((truth == 0) & prediction).sum())
    false_negative = int(((truth == 1) & ~prediction).sum())
    true_negative = int(((truth == 0) & ~prediction).sum())
    precision = (
        float(true_positive / (true_positive + false_positive))
        if true_positive + false_positive
        else 0.0
    )
    recall = (
        float(true_positive / (true_positive + false_negative))
        if true_positive + false_negative
        else 0.0
    )
    f1 = (
        float(2.0 * precision * recall / (precision + recall))
        if precision + recall
        else 0.0
    )
    accuracy = float((true_positive + true_negative) / len(truth)) if len(truth) else np.nan
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "true_negative": float(true_negative),
        "false_positive": float(false_positive),
        "false_negative": float(false_negative),
        "true_positive": float(true_positive),
        "maximin_prf": float(min(precision, recall, f1)),
    }


def forecast_training_positive_class_weight(train_y: np.ndarray) -> float:
    values = np.asarray(train_y, dtype=int)
    positives = int(values.sum())
    negatives = int(len(values) - positives)
    if positives == 0:
        raise ValueError("forecast training partition contains no positive labels")
    if negatives == 0:
        raise ValueError("forecast training partition contains no negative labels")
    return float(negatives / positives)


def make_forecast_hist_gradient_boosting(seed: int = 2026) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=200,
        max_leaf_nodes=15,
        min_samples_leaf=30,
        l2_regularization=1.0,
        early_stopping=False,
        random_state=seed,
    )


def _selection_key(row: pd.Series) -> tuple[float, float, float, float, float, float]:
    return (
        float(row["validation_maximin_prf"]),
        float(row["validation_f1"]),
        float(row["validation_precision"]),
        float(row["validation_recall"]),
        -float(row["weight_multiplier"]),
        float(row["threshold"]),
    )


def select_validation_maximin(selection_trace: pd.DataFrame) -> pd.Series:
    required = {
        "weight_multiplier",
        "threshold",
        "validation_precision",
        "validation_recall",
        "validation_f1",
        "validation_maximin_prf",
    }
    missing = sorted(required.difference(selection_trace.columns))
    if missing:
        raise KeyError(missing)
    if selection_trace.empty:
        raise ValueError("validation selection trace is empty")
    best_index = max(selection_trace.index, key=lambda index: _selection_key(selection_trace.loc[index]))
    return selection_trace.loc[best_index].copy()


def select_validation_threshold_rule(
    selection_trace: pd.DataFrame,
    rule: str,
    *,
    precision_floor: float = 0.55,
) -> pd.Series | None:
    required = {
        "threshold",
        "validation_precision",
        "validation_recall",
        "validation_f1",
        "validation_accuracy",
        "validation_maximin_prf",
    }
    missing = sorted(required.difference(selection_trace.columns))
    if missing:
        raise KeyError(missing)
    if selection_trace.empty:
        raise ValueError("validation threshold trace is empty")
    if rule not in FORECAST_THRESHOLD_SELECTION_RULES:
        raise ValueError(f"unknown forecast threshold selection rule: {rule}")
    if not 0.0 <= float(precision_floor) <= 1.0:
        raise ValueError("precision_floor must be between zero and one")

    candidates = selection_trace.copy()
    if rule == "max_recall_precision_floor":
        candidates = candidates.loc[
            candidates["validation_precision"].astype(float).ge(float(precision_floor))
        ].copy()
        if candidates.empty:
            return None

    def key(row: pd.Series) -> tuple[float, float, float, float, float]:
        if rule == "maximin":
            return (
                float(row["validation_maximin_prf"]),
                float(row["validation_f1"]),
                float(row["validation_precision"]),
                float(row["validation_recall"]),
                float(row["threshold"]),
            )
        if rule == "max_f1":
            return (
                float(row["validation_f1"]),
                float(row["validation_precision"]),
                float(row["validation_recall"]),
                float(row["validation_accuracy"]),
                float(row["threshold"]),
            )
        return (
            float(row["validation_recall"]),
            float(row["validation_f1"]),
            float(row["validation_precision"]),
            float(row["validation_accuracy"]),
            float(row["threshold"]),
        )

    best_index = max(candidates.index, key=lambda index: key(candidates.loc[index]))
    return candidates.loc[best_index].copy()


def fit_forecast_hist_gradient_boosting(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    feature_columns: tuple[str, ...] | list[str],
    *,
    seed: int = 2026,
    weight_multipliers: tuple[float, ...] = FORECAST_WEIGHT_MULTIPLIERS,
    thresholds: tuple[float, ...] = FORECAST_THRESHOLDS,
    deadline_monotonic: float | None = None,
) -> ForecastModelSelection:
    if not weight_multipliers or not thresholds:
        raise ValueError("forecast class-weight and threshold grids must be non-empty")
    columns = tuple(feature_columns)
    train_y = train["y"].to_numpy(dtype=int)
    validation_y = validation["y"].to_numpy(dtype=int)
    base_weight = forecast_training_positive_class_weight(train_y)
    train_matrix = forecast_feature_matrix(train, columns)
    validation_matrix = forecast_feature_matrix(validation, columns)
    models: dict[float, HistGradientBoostingClassifier] = {}
    probabilities: dict[float, np.ndarray] = {}
    rows: list[dict[str, object]] = []
    for multiplier in weight_multipliers:
        if deadline_monotonic is not None and time.monotonic() >= float(
            deadline_monotonic
        ):
            raise ForecastTrainingTimeboxReached(
                "forecast training timebox reached before the next class-weight fit"
            )
        multiplier = float(multiplier)
        if multiplier <= 0.0:
            raise ValueError("forecast class-weight multipliers must be positive")
        positive_weight = base_weight * multiplier
        sample_weight = np.where(train_y == 1, positive_weight, 1.0)
        model = make_forecast_hist_gradient_boosting(seed)
        model.fit(train_matrix, train_y, sample_weight=sample_weight)
        validation_probability = model.predict_proba(validation_matrix)[:, 1].astype(float)
        models[multiplier] = model
        probabilities[multiplier] = validation_probability
        for threshold in thresholds:
            threshold = float(threshold)
            metrics = forecast_classification_metrics(
                validation_y,
                validation_probability,
                threshold,
            )
            rows.append(
                {
                    "weight_multiplier": multiplier,
                    "training_base_positive_class_weight": base_weight,
                    "positive_class_weight": positive_weight,
                    "threshold": threshold,
                    **{
                        f"validation_{name}": value
                        for name, value in metrics.items()
                    },
                }
            )
    trace = pd.DataFrame(rows)
    selected = select_validation_maximin(trace)
    selected_multiplier = float(selected["weight_multiplier"])
    selected_threshold = float(selected["threshold"])
    selected_probability = probabilities[selected_multiplier]
    validation_metrics = forecast_classification_metrics(
        validation_y,
        selected_probability,
        selected_threshold,
    )
    trace["selected"] = (
        trace["weight_multiplier"].eq(selected_multiplier)
        & trace["threshold"].eq(selected_threshold)
    )
    return ForecastModelSelection(
        model=models[selected_multiplier],
        positive_class_weight=float(selected["positive_class_weight"]),
        weight_multiplier=selected_multiplier,
        threshold=selected_threshold,
        validation_probability=selected_probability,
        validation_metrics=validation_metrics,
        selection_trace=trace,
    )


def discrete_hazard_feature_matrix(
    frame: pd.DataFrame,
    feature_columns: tuple[str, ...] | list[str],
) -> np.ndarray:
    values = forecast_feature_matrix(frame, feature_columns)
    return np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)


def make_discrete_hazard_logistic(seed: int = 2026) -> Pipeline:
    return Pipeline(
        [
            ("scale", RobustScaler()),
            (
                "model",
                LogisticRegression(
                    C=0.05,
                    penalty="l2",
                    solver="lbfgs",
                    max_iter=1000,
                    random_state=seed,
                ),
            ),
        ]
    )


def make_discrete_hazard_boosted(seed: int = 2026) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=120,
        max_depth=3,
        max_leaf_nodes=7,
        min_samples_leaf=100,
        l2_regularization=25.0,
        early_stopping=False,
        random_state=seed,
    )


def fit_discrete_hazard_model(
    method: str,
    train: pd.DataFrame,
    feature_columns: tuple[str, ...] | list[str],
    *,
    seed: int = 2026,
    deadline_monotonic: float | None = None,
) -> DiscreteHazardModelFit:
    method = str(method)
    if method not in DISCRETE_HAZARD_METHODS:
        raise ValueError(f"unknown discrete hazard method: {method}")
    if deadline_monotonic is not None and time.monotonic() >= float(deadline_monotonic):
        raise ForecastTrainingTimeboxReached(
            "forecast training timebox reached before discrete hazard fitting"
        )
    columns = tuple(feature_columns)
    train_y = train["y"].to_numpy(dtype=int)
    positive_weight = forecast_training_positive_class_weight(train_y)
    sample_weight = np.where(train_y == 1, positive_weight, 1.0)
    matrix = discrete_hazard_feature_matrix(train, columns)
    if method == "logistic_hazard":
        model = make_discrete_hazard_logistic(seed)
        model.fit(matrix, train_y, model__sample_weight=sample_weight)
    else:
        model = make_discrete_hazard_boosted(seed)
        model.fit(matrix, train_y, sample_weight=sample_weight)
    return DiscreteHazardModelFit(
        method=method,
        model=model,
        positive_class_weight=float(positive_weight),
        feature_columns=columns,
    )


def discrete_hazard_probability(
    fit: DiscreteHazardModelFit,
    frame: pd.DataFrame,
) -> np.ndarray:
    values = fit.model.predict_proba(
        discrete_hazard_feature_matrix(frame, fit.feature_columns)
    )[:, 1]
    return np.asarray(values, dtype=float)


def fit_hazard_probability_calibrator(
    validation_y: np.ndarray,
    raw_validation_probability: np.ndarray,
) -> HazardProbabilityCalibrator:
    truth = np.asarray(validation_y, dtype=int)
    probability = np.clip(np.asarray(raw_validation_probability, dtype=float), 1e-6, 1.0 - 1e-6)
    if len(truth) != len(probability):
        raise ValueError("hazard calibration labels and probabilities differ in length")
    if len(np.unique(truth)) < 2:
        return HazardProbabilityCalibrator(None, "identity_insufficient_validation_classes")
    logits = np.log(probability / (1.0 - probability)).reshape(-1, 1)
    model = LogisticRegression(C=1.0, penalty="l2", solver="lbfgs", max_iter=1000)
    model.fit(logits, truth)
    return HazardProbabilityCalibrator(model, "validation_platt")


def calibrate_hazard_probability(
    calibrator: HazardProbabilityCalibrator,
    raw_probability: np.ndarray,
) -> np.ndarray:
    probability = np.clip(np.asarray(raw_probability, dtype=float), 1e-6, 1.0 - 1e-6)
    if calibrator.model is None:
        return probability
    logits = np.log(probability / (1.0 - probability)).reshape(-1, 1)
    return calibrator.model.predict_proba(logits)[:, 1].astype(float)


def cumulate_stationary_hazard(
    hourly_hazard: np.ndarray,
    horizon: int,
) -> np.ndarray:
    horizon = int(horizon)
    if horizon <= 0:
        raise ValueError("hazard cumulation horizon must be positive")
    probability = np.clip(np.asarray(hourly_hazard, dtype=float), 0.0, 1.0)
    return 1.0 - np.power(1.0 - probability, horizon)


def select_discrete_hazard_threshold(
    validation_y: np.ndarray,
    horizon_probability: np.ndarray,
    *,
    thresholds: tuple[float, ...] = DISCRETE_HAZARD_THRESHOLDS,
) -> tuple[float, dict[str, float], pd.DataFrame]:
    if not thresholds:
        raise ValueError("discrete hazard threshold grid must be non-empty")
    truth = np.asarray(validation_y, dtype=int)
    probability = np.asarray(horizon_probability, dtype=float)
    if len(truth) != len(probability):
        raise ValueError("hazard threshold labels and probabilities differ in length")
    rows: list[dict[str, object]] = []
    for threshold in thresholds:
        threshold = float(threshold)
        metrics = forecast_classification_metrics(truth, probability, threshold)
        rows.append(
            {
                "threshold": threshold,
                **{f"validation_{name}": value for name, value in metrics.items()},
            }
        )
    trace = pd.DataFrame(rows)
    selected = select_validation_maximin(trace.assign(weight_multiplier=1.0))
    threshold = float(selected["threshold"])
    metrics = forecast_classification_metrics(truth, probability, threshold)
    trace["selected"] = trace["threshold"].eq(threshold)
    return threshold, metrics, trace


def forecast_model_probability(
    model: HistGradientBoostingClassifier,
    frame: pd.DataFrame,
    feature_columns: tuple[str, ...] | list[str],
) -> np.ndarray:
    return model.predict_proba(forecast_feature_matrix(frame, feature_columns))[:, 1].astype(
        float
    )


def forecast_base_rate_prediction(
    train_y: np.ndarray,
    frame: pd.DataFrame,
) -> RiskPrediction:
    return majority_predict(train_y, frame)


def forecast_persistence_prediction(
    frame: pd.DataFrame,
    horizon: int,
) -> RiskPrediction:
    column = f"persistence_event_count_{int(horizon)}h"
    if column not in frame.columns:
        raise KeyError(column)
    prediction = pd.to_numeric(frame[column], errors="coerce").fillna(0.0).gt(0.0)
    values = prediction.to_numpy(dtype=int)
    return RiskPrediction("persistence", 0.5, values.astype(float), values)


def forecast_recurrence_prediction(
    frame: pd.DataFrame,
    *,
    event_source: str | None = None,
    window_hours: int = 24 * 7,
    recurrence_column: str | None = None,
) -> RiskPrediction:
    window_hours = int(window_hours)
    if window_hours <= 0:
        raise ValueError("recurrence window must be positive")
    if recurrence_column is None and event_source is None:
        raise ValueError("recurrence prediction requires an event source or feature column")
    column = (
        str(recurrence_column)
        if recurrence_column is not None
        else f"history_{str(event_source)}_any_event_trailing_{window_hours}h"
    )
    if column not in frame.columns:
        raise KeyError(column)
    prediction = pd.to_numeric(frame[column], errors="coerce").fillna(0.0).gt(0.0)
    values = prediction.to_numpy(dtype=int)
    return RiskPrediction(
        f"recurrence_{window_hours}h",
        0.5,
        values.astype(float),
        values,
    )


def validation_permutation_importance(
    model: HistGradientBoostingClassifier,
    validation: pd.DataFrame,
    feature_columns: tuple[str, ...] | list[str],
    threshold: float,
    *,
    repeats: int = 3,
    seed: int = 2026,
) -> pd.DataFrame:
    columns = tuple(feature_columns)
    matrix = forecast_feature_matrix(validation, columns)
    target = validation["y"].to_numpy(dtype=int)

    def f1_at_selected_threshold(estimator, values, truth) -> float:
        probability = estimator.predict_proba(values)[:, 1]
        return forecast_classification_metrics(
            np.asarray(truth, dtype=int), probability, threshold
        )["f1"]

    importance = permutation_importance(
        model,
        matrix,
        target,
        scoring=f1_at_selected_threshold,
        n_repeats=int(repeats),
        random_state=seed,
        n_jobs=1,
    )
    return pd.DataFrame(
        {
            "feature": columns,
            "importance_mean": importance.importances_mean.astype(float),
            "importance_std": importance.importances_std.astype(float),
            "n_repeats": int(repeats),
            "importance_split": "validation",
            "importance_metric": "f1_at_validation_selected_threshold",
        }
    ).sort_values("importance_mean", ascending=False, kind="mergesort").reset_index(
        drop=True
    )


@dataclass(frozen=True)
class IncidentHazardRgfnConfig:
    window_hours: int = INCIDENT_HAZARD_SEQUENCE_WINDOW_HOURS
    temporal_width: int = len(INCIDENT_HAZARD_TEMPORAL_FEATURE_COLUMNS)
    evidence_width: int = len(INCIDENT_HAZARD_EVIDENCE_FEATURE_COLUMNS)
    context_width: int = len(INCIDENT_HAZARD_CONTEXT_FEATURE_COLUMNS)
    station_embedding_dim: int = 8
    temporal_hidden_size: int = 32
    evidence_hidden_size: int = 32
    evidence_embedding_size: int = 16
    context_embedding_size: int = 16
    dilations: tuple[int, ...] = (1, 2, 4, 8)
    dropout: float = 0.20
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 256
    max_epochs: int = 40
    patience: int = 6
    seed: int = 2026


@dataclass(frozen=True)
class IncidentHazardNormalizer:
    temporal_center: np.ndarray
    temporal_scale: np.ndarray
    evidence_center: np.ndarray
    evidence_scale: np.ndarray
    context_center: np.ndarray
    context_scale: np.ndarray
    station_to_code: dict[str, int]


@dataclass(frozen=True)
class IncidentHazardTensorBundle:
    temporal_values: np.ndarray
    temporal_mask: np.ndarray
    evidence_values: np.ndarray
    evidence_mask: np.ndarray
    context_values: np.ndarray
    context_mask: np.ndarray
    station_codes: np.ndarray
    y: np.ndarray
    keys: pd.DataFrame


@dataclass(frozen=True)
class IncidentHazardModelSelection:
    model: "CausalIncidentHazardRGFN"
    normalizer: IncidentHazardNormalizer
    config: IncidentHazardRgfnConfig
    positive_class_weight: float
    weight_multiplier: float
    threshold: float
    validation_probability: np.ndarray
    validation_metrics: dict[str, float]
    selection_trace: pd.DataFrame
    training_history: pd.DataFrame


def incident_hazard_input_schema() -> dict[str, tuple[str, ...]]:
    return {
        "temporal": tuple(INCIDENT_HAZARD_TEMPORAL_FEATURE_COLUMNS),
        "evidence": tuple(INCIDENT_HAZARD_EVIDENCE_FEATURE_COLUMNS),
        "context": tuple(INCIDENT_HAZARD_CONTEXT_FEATURE_COLUMNS),
    }


def _require_incident_feature_columns(frame: pd.DataFrame) -> None:
    required = {
        "station_id",
        "hour_utc",
        "y",
        *INCIDENT_HAZARD_TEMPORAL_FEATURE_COLUMNS,
        *INCIDENT_HAZARD_EVIDENCE_FEATURE_COLUMNS,
        *INCIDENT_HAZARD_CONTEXT_FEATURE_COLUMNS,
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise KeyError(f"incident-hazard input is missing columns: {missing}")


def attach_incident_hazard_features(
    partition: pd.DataFrame,
    feature_frame: pd.DataFrame,
) -> pd.DataFrame:
    required_partition = {"station_id", "hour_utc", "y"}
    missing_partition = sorted(required_partition.difference(partition.columns))
    if missing_partition:
        raise KeyError(f"incident-hazard partition is missing columns: {missing_partition}")
    schema_columns = tuple(
        dict.fromkeys(
            [
                *INCIDENT_HAZARD_TEMPORAL_FEATURE_COLUMNS,
                *INCIDENT_HAZARD_EVIDENCE_FEATURE_COLUMNS,
                *INCIDENT_HAZARD_CONTEXT_FEATURE_COLUMNS,
            ]
        )
    )
    required_features = {"station_id", "hour_utc", *schema_columns}
    missing_features = sorted(required_features.difference(feature_frame.columns))
    if missing_features:
        raise KeyError(f"incident-hazard features are missing columns: {missing_features}")
    source = feature_frame.loc[:, ["station_id", "hour_utc", *schema_columns]].copy()
    source["station_id"] = source["station_id"].astype(str)
    source["hour_utc"] = pd.to_datetime(source["hour_utc"], utc=True, errors="coerce")
    if source.duplicated(["station_id", "hour_utc"]).any():
        raise ValueError("incident-hazard feature rows must have unique station-hour keys")
    result = partition.copy(deep=True)
    result["station_id"] = result["station_id"].astype(str)
    result["hour_utc"] = pd.to_datetime(result["hour_utc"], utc=True, errors="coerce")
    result = result.merge(
        source,
        on=["station_id", "hour_utc"],
        how="left",
        validate="one_to_one",
    )
    if result.loc[:, list(schema_columns)].isna().all(axis=1).any():
        raise RuntimeError("an incident-hazard partition row lacks its feature source")
    _require_incident_feature_columns(result)
    return result.sort_values(["hour_utc", "station_id"], kind="mergesort").reset_index(
        drop=True
    )


def _robust_location_scale(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    numeric = np.asarray(values, dtype=np.float64)
    if numeric.ndim != 2:
        raise ValueError("normalization values must be two-dimensional")
    finite = np.where(np.isfinite(numeric), numeric, np.nan)
    with np.errstate(all="ignore"):
        center = np.nanmedian(finite, axis=0)
        lower = np.nanpercentile(finite, 25.0, axis=0)
        upper = np.nanpercentile(finite, 75.0, axis=0)
    center = np.where(np.isfinite(center), center, 0.0)
    scale = upper - lower
    with np.errstate(all="ignore"):
        fallback = np.nanstd(finite, axis=0)
    scale = np.where(np.isfinite(scale) & (scale > 1e-6), scale, fallback)
    scale = np.where(np.isfinite(scale) & (scale > 1e-6), scale, 1.0)
    return center.astype(np.float32), scale.astype(np.float32)


def _normalize_with_mask(
    values: np.ndarray,
    center: np.ndarray,
    scale: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    numeric = np.asarray(values, dtype=np.float32)
    mask = np.isfinite(numeric)
    scaled = (numeric - center) / scale
    scaled = np.where(mask, scaled, 0.0)
    scaled = np.clip(scaled, -12.0, 12.0).astype(np.float32)
    return scaled, mask.astype(np.float32)


def _continuous_incident_feature_frame(feature_frame: pd.DataFrame) -> pd.DataFrame:
    source = feature_frame.copy(deep=True)
    source["station_id"] = source["station_id"].astype(str)
    source["hour_utc"] = pd.to_datetime(source["hour_utc"], utc=True, errors="coerce")
    if source["hour_utc"].isna().any():
        raise ValueError("incident-hazard feature timestamps must be valid")
    if source.duplicated(["station_id", "hour_utc"]).any():
        raise ValueError("incident-hazard feature rows must have unique station-hour keys")
    frames: list[pd.DataFrame] = []
    for station_id, station in source.groupby("station_id", sort=False):
        station = station.sort_values("hour_utc", kind="mergesort")
        full_index = pd.date_range(
            station["hour_utc"].min(),
            station["hour_utc"].max(),
            freq="h",
            tz="UTC",
        )
        reindexed = station.set_index("hour_utc").reindex(full_index)
        reindexed.index.name = "hour_utc"
        reindexed["station_id"] = str(station_id)
        frames.append(reindexed.reset_index())
    if not frames:
        return source.iloc[0:0].copy()
    return pd.concat(frames, ignore_index=True).sort_values(
        ["station_id", "hour_utc"], kind="mergesort"
    ).reset_index(drop=True)


def fit_incident_hazard_normalizer(
    feature_frame: pd.DataFrame,
    train_partition: pd.DataFrame,
) -> IncidentHazardNormalizer:
    _require_incident_feature_columns(train_partition)
    source = _continuous_incident_feature_frame(feature_frame)
    train_end = pd.to_datetime(train_partition["hour_utc"], utc=True).max()
    source_train = source.loc[source["hour_utc"].le(train_end)].copy()
    if source_train.empty:
        raise ValueError("incident-hazard training history is empty")
    temporal_center, temporal_scale = _robust_location_scale(
        source_train.loc[:, INCIDENT_HAZARD_TEMPORAL_FEATURE_COLUMNS].apply(
            pd.to_numeric, errors="coerce"
        ).to_numpy(dtype=np.float32)
    )
    evidence_center, evidence_scale = _robust_location_scale(
        train_partition.loc[:, INCIDENT_HAZARD_EVIDENCE_FEATURE_COLUMNS].apply(
            pd.to_numeric, errors="coerce"
        ).to_numpy(dtype=np.float32)
    )
    context_center, context_scale = _robust_location_scale(
        train_partition.loc[:, INCIDENT_HAZARD_CONTEXT_FEATURE_COLUMNS].apply(
            pd.to_numeric, errors="coerce"
        ).to_numpy(dtype=np.float32)
    )
    stations = sorted(train_partition["station_id"].astype(str).unique())
    station_to_code = {station_id: index + 1 for index, station_id in enumerate(stations)}
    return IncidentHazardNormalizer(
        temporal_center=temporal_center,
        temporal_scale=temporal_scale,
        evidence_center=evidence_center,
        evidence_scale=evidence_scale,
        context_center=context_center,
        context_scale=context_scale,
        station_to_code=station_to_code,
    )


def build_incident_hazard_tensor_bundle(
    feature_frame: pd.DataFrame,
    partition: pd.DataFrame,
    normalizer: IncidentHazardNormalizer,
    *,
    window_hours: int = INCIDENT_HAZARD_SEQUENCE_WINDOW_HOURS,
) -> IncidentHazardTensorBundle:
    if int(window_hours) < 1:
        raise ValueError("incident-hazard window_hours must be positive")
    _require_incident_feature_columns(partition)
    source = _continuous_incident_feature_frame(feature_frame)
    source["_position"] = source.groupby("station_id", sort=False).cumcount()
    lookup = source.loc[:, ["station_id", "hour_utc", "_position"]]
    examples = partition.copy(deep=True)
    examples["station_id"] = examples["station_id"].astype(str)
    examples["hour_utc"] = pd.to_datetime(examples["hour_utc"], utc=True, errors="coerce")
    examples["_example_position"] = np.arange(len(examples), dtype=int)
    examples = examples.merge(
        lookup,
        on=["station_id", "hour_utc"],
        how="left",
        validate="one_to_one",
    )
    if examples["_position"].isna().any():
        raise RuntimeError("an incident-hazard partition row cannot be placed on its clock grid")
    temporal_width = len(INCIDENT_HAZARD_TEMPORAL_FEATURE_COLUMNS)
    n_examples = len(examples)
    temporal_values = np.zeros((n_examples, int(window_hours), temporal_width), dtype=np.float32)
    temporal_mask = np.zeros_like(temporal_values)
    for station_id, station_examples in examples.groupby("station_id", sort=False):
        station_source = source.loc[source["station_id"].eq(str(station_id))].sort_values(
            "hour_utc", kind="mergesort"
        )
        raw = station_source.loc[:, INCIDENT_HAZARD_TEMPORAL_FEATURE_COLUMNS].apply(
            pd.to_numeric, errors="coerce"
        ).to_numpy(dtype=np.float32)
        normalized, observed = _normalize_with_mask(
            raw,
            normalizer.temporal_center,
            normalizer.temporal_scale,
        )
        example_rows = station_examples["_example_position"].to_numpy(dtype=int)
        endpoint_positions = station_examples["_position"].to_numpy(dtype=int)
        offsets = np.arange(int(window_hours) - 1, -1, -1, dtype=int)
        positions = endpoint_positions[:, None] - offsets[None, :]
        valid = positions >= 0
        clipped = np.clip(positions, 0, len(station_source) - 1)
        values = normalized[clipped]
        masks = observed[clipped]
        values[~valid] = 0.0
        masks[~valid] = 0.0
        temporal_values[example_rows] = values
        temporal_mask[example_rows] = masks
    evidence_values, evidence_mask = _normalize_with_mask(
        examples.loc[:, INCIDENT_HAZARD_EVIDENCE_FEATURE_COLUMNS].apply(
            pd.to_numeric, errors="coerce"
        ).to_numpy(dtype=np.float32),
        normalizer.evidence_center,
        normalizer.evidence_scale,
    )
    context_values, context_mask = _normalize_with_mask(
        examples.loc[:, INCIDENT_HAZARD_CONTEXT_FEATURE_COLUMNS].apply(
            pd.to_numeric, errors="coerce"
        ).to_numpy(dtype=np.float32),
        normalizer.context_center,
        normalizer.context_scale,
    )
    station_codes = examples["station_id"].map(normalizer.station_to_code).fillna(0).to_numpy(
        dtype=np.int64
    )
    keys = examples.loc[:, ["station_id", "hour_utc"]].copy().reset_index(drop=True)
    return IncidentHazardTensorBundle(
        temporal_values=temporal_values,
        temporal_mask=temporal_mask,
        evidence_values=evidence_values,
        evidence_mask=evidence_mask,
        context_values=context_values,
        context_mask=context_mask,
        station_codes=station_codes,
        y=examples["y"].to_numpy(dtype=np.float32),
        keys=keys,
    )


class _CausalDilatedResidualLayer(nn.Module):
    def __init__(self, width: int, dilation: int, dropout: float) -> None:
        super().__init__()
        if int(dilation) < 1:
            raise ValueError("TCN dilation must be positive")
        self.left_padding = 2 * int(dilation)
        self.first = nn.Conv1d(width, width, kernel_size=3, dilation=int(dilation))
        self.second = nn.Conv1d(width, width, kernel_size=3, dilation=int(dilation))
        self.dropout = nn.Dropout(float(dropout))

    def _causal(self, layer: nn.Conv1d, values: torch.Tensor) -> torch.Tensor:
        return layer(torch_functional.pad(values, (self.left_padding, 0)))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        hidden = self._causal(self.first, values)
        hidden = self.dropout(torch_functional.gelu(hidden))
        hidden = self._causal(self.second, hidden)
        hidden = self.dropout(torch_functional.gelu(hidden))
        return torch_functional.gelu(values + hidden)


class CausalIncidentHazardRGFN(nn.Module):
    def __init__(
        self,
        config: IncidentHazardRgfnConfig | None = None,
        *,
        n_stations: int,
    ) -> None:
        super().__init__()
        self.config = config or IncidentHazardRgfnConfig()
        if n_stations < 1:
            raise ValueError("incident-hazard model needs at least the unknown station code")
        if self.config.window_hours < 1:
            raise ValueError("incident-hazard window must be positive")
        if self.config.temporal_width < 1 or self.config.evidence_width < 1 or self.config.context_width < 1:
            raise ValueError("incident-hazard feature widths must be positive")
        if not self.config.dilations:
            raise ValueError("incident-hazard TCN needs at least one dilation")
        self.n_stations = int(n_stations)
        temporal_input_width = self.config.temporal_width * 2
        self.temporal_input = nn.Conv1d(
            temporal_input_width,
            self.config.temporal_hidden_size,
            kernel_size=1,
        )
        self.temporal_layers = nn.ModuleList(
            _CausalDilatedResidualLayer(
                self.config.temporal_hidden_size,
                int(dilation),
                self.config.dropout,
            )
            for dilation in self.config.dilations
        )
        self.temporal_head = nn.Linear(self.config.temporal_hidden_size, 1)
        self.station_embedding = nn.Embedding(
            self.n_stations,
            self.config.station_embedding_dim,
        )
        self.evidence_net = nn.Sequential(
            nn.Linear(self.config.evidence_width * 2, self.config.evidence_hidden_size),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.evidence_hidden_size, self.config.evidence_embedding_size),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
        )
        self.context_net = nn.Sequential(
            nn.Linear(self.config.context_width * 2, self.config.context_embedding_size),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
        )
        evidence_width = (
            self.config.evidence_embedding_size
            + self.config.context_embedding_size
            + self.config.station_embedding_dim
        )
        fusion_width = self.config.temporal_hidden_size + evidence_width
        self.evidence_head = nn.Linear(evidence_width, 1)
        self.station_head = nn.Linear(self.config.station_embedding_dim, 1)
        self.gate = nn.Sequential(
            nn.Linear(fusion_width, self.config.temporal_hidden_size),
            nn.GELU(),
            nn.Linear(self.config.temporal_hidden_size, 1),
        )
        self.fusion_residual = nn.Linear(fusion_width, 1)

    def parameter_count(self) -> int:
        return int(sum(parameter.numel() for parameter in self.parameters()))

    def _validate_inputs(
        self,
        temporal_values: torch.Tensor,
        temporal_mask: torch.Tensor,
        evidence_values: torch.Tensor,
        evidence_mask: torch.Tensor,
        context_values: torch.Tensor,
        context_mask: torch.Tensor,
        station_codes: torch.Tensor,
    ) -> None:
        if temporal_values.ndim != 3:
            raise ValueError("incident-hazard temporal values must have shape [batch, hours, features]")
        if tuple(temporal_values.shape[1:]) != (
            self.config.window_hours,
            self.config.temporal_width,
        ):
            raise ValueError("incident-hazard temporal values do not match model configuration")
        if tuple(temporal_mask.shape) != tuple(temporal_values.shape):
            raise ValueError("incident-hazard temporal mask does not match temporal values")
        batch = temporal_values.shape[0]
        if tuple(evidence_values.shape) != (batch, self.config.evidence_width):
            raise ValueError("incident-hazard evidence values do not match model configuration")
        if tuple(evidence_mask.shape) != tuple(evidence_values.shape):
            raise ValueError("incident-hazard evidence mask does not match evidence values")
        if tuple(context_values.shape) != (batch, self.config.context_width):
            raise ValueError("incident-hazard context values do not match model configuration")
        if tuple(context_mask.shape) != tuple(context_values.shape):
            raise ValueError("incident-hazard context mask does not match context values")
        if tuple(station_codes.shape) != (batch,):
            raise ValueError("incident-hazard station codes must have shape [batch]")

    def forward(
        self,
        temporal_values: torch.Tensor,
        temporal_mask: torch.Tensor,
        evidence_values: torch.Tensor,
        evidence_mask: torch.Tensor,
        context_values: torch.Tensor,
        context_mask: torch.Tensor,
        station_codes: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        self._validate_inputs(
            temporal_values,
            temporal_mask,
            evidence_values,
            evidence_mask,
            context_values,
            context_mask,
            station_codes,
        )
        temporal_input = torch.cat(
            [
                torch.nan_to_num(temporal_values),
                torch.nan_to_num(temporal_mask),
            ],
            dim=-1,
        ).transpose(1, 2)
        temporal = torch_functional.gelu(self.temporal_input(temporal_input))
        for layer in self.temporal_layers:
            temporal = layer(temporal)
        temporal_embedding = temporal[:, :, -1]
        evidence = self.evidence_net(
            torch.cat(
                [torch.nan_to_num(evidence_values), torch.nan_to_num(evidence_mask)],
                dim=-1,
            )
        )
        context = self.context_net(
            torch.cat(
                [torch.nan_to_num(context_values), torch.nan_to_num(context_mask)],
                dim=-1,
            )
        )
        safe_station_codes = torch.clamp(station_codes.long(), min=0, max=self.n_stations - 1)
        station = self.station_embedding(safe_station_codes)
        evidence_embedding = torch.cat([evidence, context, station], dim=-1)
        fusion = torch.cat([temporal_embedding, evidence_embedding], dim=-1)
        temporal_logit = self.temporal_head(temporal_embedding).squeeze(-1)
        evidence_logit = self.evidence_head(evidence_embedding).squeeze(-1)
        station_logit = self.station_head(station).squeeze(-1)
        alpha = torch.sigmoid(self.gate(fusion).squeeze(-1))
        incident_logit = (
            alpha * temporal_logit
            + (1.0 - alpha) * evidence_logit
            + station_logit
            + self.fusion_residual(fusion).squeeze(-1)
        )
        return {
            "incident_logit": incident_logit,
            "incident_probability": torch.sigmoid(incident_logit),
            "temporal_logit": temporal_logit,
            "evidence_logit": evidence_logit,
            "station_logit": station_logit,
            "alpha": alpha,
        }


def _incident_hazard_seed(seed: int) -> None:
    resolved = int(seed)
    np.random.seed(resolved)
    torch.manual_seed(resolved)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(resolved)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False


def _incident_hazard_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _incident_hazard_loader(
    bundle: IncidentHazardTensorBundle,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    tensors = TensorDataset(
        torch.from_numpy(bundle.temporal_values),
        torch.from_numpy(bundle.temporal_mask),
        torch.from_numpy(bundle.evidence_values),
        torch.from_numpy(bundle.evidence_mask),
        torch.from_numpy(bundle.context_values),
        torch.from_numpy(bundle.context_mask),
        torch.from_numpy(bundle.station_codes),
        torch.from_numpy(bundle.y),
    )
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return DataLoader(
        tensors,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        generator=generator,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


def _incident_hazard_probability_from_bundle(
    model: CausalIncidentHazardRGFN,
    bundle: IncidentHazardTensorBundle,
    *,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    device = next(model.parameters()).device
    loader = _incident_hazard_loader(
        bundle,
        batch_size=batch_size,
        shuffle=False,
        seed=0,
    )
    probabilities: list[np.ndarray] = []
    logits: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for values in loader:
            temporal, temporal_mask, evidence, evidence_mask, context, context_mask, station, _ = values
            output = model(
                temporal.to(device),
                temporal_mask.to(device),
                evidence.to(device),
                evidence_mask.to(device),
                context.to(device),
                context_mask.to(device),
                station.to(device),
            )
            probabilities.append(output["incident_probability"].detach().cpu().numpy())
            logits.append(output["incident_logit"].detach().cpu().numpy())
    return (
        np.concatenate(probabilities).astype(float) if probabilities else np.empty(0, dtype=float),
        np.concatenate(logits).astype(float) if logits else np.empty(0, dtype=float),
    )


def _incident_hazard_threshold_trace(
    validation_y: np.ndarray,
    validation_probability: np.ndarray,
    *,
    weight_multiplier: float,
    base_weight: float,
    best_epoch: int,
    validation_loss: float,
    thresholds: tuple[float, ...],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for threshold in thresholds:
        metrics = forecast_classification_metrics(
            validation_y,
            validation_probability,
            float(threshold),
        )
        rows.append(
            {
                "weight_multiplier": float(weight_multiplier),
                "training_base_positive_class_weight": float(base_weight),
                "positive_class_weight": float(base_weight * weight_multiplier),
                "threshold": float(threshold),
                "best_epoch": int(best_epoch),
                "validation_loss": float(validation_loss),
                "test_metrics_accessed_during_selection": False,
                **{f"validation_{name}": value for name, value in metrics.items()},
            }
        )
    return rows


def fit_incident_hazard_rgfn(
    feature_frame: pd.DataFrame,
    train_partition: pd.DataFrame,
    validation_partition: pd.DataFrame,
    *,
    config: IncidentHazardRgfnConfig | None = None,
    weight_multipliers: tuple[float, ...] = INCIDENT_HAZARD_WEIGHT_MULTIPLIERS,
    thresholds: tuple[float, ...] = INCIDENT_HAZARD_THRESHOLDS,
) -> IncidentHazardModelSelection:
    resolved = config or IncidentHazardRgfnConfig()
    if not weight_multipliers or not thresholds:
        raise ValueError("incident-hazard selection grids must be non-empty")
    train = attach_incident_hazard_features(train_partition, feature_frame)
    validation = attach_incident_hazard_features(validation_partition, feature_frame)
    normalizer = fit_incident_hazard_normalizer(feature_frame, train)
    train_bundle = build_incident_hazard_tensor_bundle(
        feature_frame,
        train,
        normalizer,
        window_hours=resolved.window_hours,
    )
    validation_bundle = build_incident_hazard_tensor_bundle(
        feature_frame,
        validation,
        normalizer,
        window_hours=resolved.window_hours,
    )
    base_weight = forecast_training_positive_class_weight(train_bundle.y.astype(int))
    device = _incident_hazard_device()
    selection_rows: list[dict[str, object]] = []
    history_rows: list[dict[str, object]] = []
    trained: dict[float, CausalIncidentHazardRGFN] = {}
    validation_probabilities: dict[float, np.ndarray] = {}
    for index, multiplier in enumerate(weight_multipliers):
        multiplier = float(multiplier)
        if multiplier <= 0.0:
            raise ValueError("incident-hazard class-weight multipliers must be positive")
        _incident_hazard_seed(resolved.seed + index)
        model = CausalIncidentHazardRGFN(
            resolved,
            n_stations=len(normalizer.station_to_code) + 1,
        ).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(resolved.learning_rate),
            weight_decay=float(resolved.weight_decay),
        )
        criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(base_weight * multiplier, dtype=torch.float32, device=device)
        )
        train_loader = _incident_hazard_loader(
            train_bundle,
            batch_size=resolved.batch_size,
            shuffle=True,
            seed=resolved.seed + index,
        )
        best_state: dict[str, torch.Tensor] | None = None
        best_loss = float("inf")
        best_epoch = 0
        stale_epochs = 0
        for epoch in range(1, int(resolved.max_epochs) + 1):
            model.train()
            train_losses: list[float] = []
            for values in train_loader:
                temporal, temporal_mask, evidence, evidence_mask, context, context_mask, station, target = values
                optimizer.zero_grad(set_to_none=True)
                output = model(
                    temporal.to(device),
                    temporal_mask.to(device),
                    evidence.to(device),
                    evidence_mask.to(device),
                    context.to(device),
                    context_mask.to(device),
                    station.to(device),
                )
                loss = criterion(output["incident_logit"], target.to(device))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                train_losses.append(float(loss.detach().cpu()))
            _, validation_logits = _incident_hazard_probability_from_bundle(
                model,
                validation_bundle,
                batch_size=resolved.batch_size,
            )
            validation_targets = torch.from_numpy(validation_bundle.y).to(device)
            validation_logit_tensor = torch.from_numpy(validation_logits.astype(np.float32)).to(device)
            validation_loss = float(
                torch_functional.binary_cross_entropy_with_logits(
                    validation_logit_tensor,
                    validation_targets,
                ).detach().cpu()
            )
            history_rows.append(
                {
                    "weight_multiplier": multiplier,
                    "epoch": epoch,
                    "training_loss": float(np.mean(train_losses)) if train_losses else np.nan,
                    "validation_loss": validation_loss,
                }
            )
            if validation_loss < best_loss - 1e-6:
                best_loss = validation_loss
                best_epoch = epoch
                best_state = deepcopy(model.state_dict())
                stale_epochs = 0
            else:
                stale_epochs += 1
                if stale_epochs >= int(resolved.patience):
                    break
        if best_state is None:
            raise RuntimeError("incident-hazard training produced no validation state")
        model.load_state_dict(best_state)
        validation_probability, _ = _incident_hazard_probability_from_bundle(
            model,
            validation_bundle,
            batch_size=resolved.batch_size,
        )
        trained[multiplier] = model
        validation_probabilities[multiplier] = validation_probability
        selection_rows.extend(
            _incident_hazard_threshold_trace(
                validation_bundle.y.astype(int),
                validation_probability,
                weight_multiplier=multiplier,
                base_weight=base_weight,
                best_epoch=best_epoch,
                validation_loss=best_loss,
                thresholds=thresholds,
            )
        )
    trace = pd.DataFrame(selection_rows)
    selected = select_validation_maximin(trace)
    selected_multiplier = float(selected["weight_multiplier"])
    selected_threshold = float(selected["threshold"])
    validation_probability = validation_probabilities[selected_multiplier]
    validation_metrics = forecast_classification_metrics(
        validation_bundle.y.astype(int),
        validation_probability,
        selected_threshold,
    )
    trace["selected"] = (
        trace["weight_multiplier"].eq(selected_multiplier)
        & trace["threshold"].eq(selected_threshold)
    )
    return IncidentHazardModelSelection(
        model=trained[selected_multiplier],
        normalizer=normalizer,
        config=resolved,
        positive_class_weight=float(selected["positive_class_weight"]),
        weight_multiplier=selected_multiplier,
        threshold=selected_threshold,
        validation_probability=validation_probability,
        validation_metrics=validation_metrics,
        selection_trace=trace,
        training_history=pd.DataFrame(history_rows),
    )


def incident_hazard_rgfn_probability(
    selection: IncidentHazardModelSelection,
    feature_frame: pd.DataFrame,
    partition: pd.DataFrame,
) -> np.ndarray:
    attached = attach_incident_hazard_features(partition, feature_frame)
    bundle = build_incident_hazard_tensor_bundle(
        feature_frame,
        attached,
        selection.normalizer,
        window_hours=selection.config.window_hours,
    )
    probability, _ = _incident_hazard_probability_from_bundle(
        selection.model,
        bundle,
        batch_size=selection.config.batch_size,
    )
    return probability
