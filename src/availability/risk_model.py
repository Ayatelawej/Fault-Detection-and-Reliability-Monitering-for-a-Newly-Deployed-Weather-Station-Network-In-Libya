from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

from src.availability.risk_dataset import FEATURE_COLUMNS
from src.model.binary_metrics import binary_metrics, choose_threshold_max_f1

MODEL_NAMES = ("majority", "flicker", "logistic_regression", "gradient_boosted_trees")
LEARNED_MODEL_NAMES = ("logistic_regression", "gradient_boosted_trees")


@dataclass
class RiskPrediction:
    model: str
    threshold: float
    prob: np.ndarray
    pred: np.ndarray


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
