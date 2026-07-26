from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from src.availability.risk_dataset import FEATURE_COLUMNS, HORIZONS, RiskDataset, build_risk_dataset
from src.availability.risk_model import LEARNED_MODEL_NAMES, flicker_predict, learned_prediction, majority_predict, metric_row
from src.config.paths import AVAILABILITY_EVENTS_PATH

CUTOFF_UTC = pd.Timestamp("2026-03-16T00:00:00Z")
VALIDATION_FRACTION = 0.15


def split_train_validation_test(frame: pd.DataFrame, cutoff: pd.Timestamp = CUTOFF_UTC, validation_fraction: float = VALIDATION_FRACTION) -> dict[str, pd.DataFrame]:
    ordered = frame.sort_values("hour_utc", kind="mergesort").reset_index(drop=True)
    train_all = ordered.loc[ordered["hour_utc"].lt(cutoff)].copy()
    test = ordered.loc[ordered["hour_utc"].ge(cutoff)].copy()
    n_validation = max(1, int(np.ceil(len(train_all) * validation_fraction))) if len(train_all) else 0
    validation = train_all.tail(n_validation).copy()
    train = train_all.iloc[: max(0, len(train_all) - n_validation)].copy()
    return {"train": train, "validation": validation, "test": test}


def _safe_auc(y_true: np.ndarray, score: np.ndarray, kind: str) -> float:
    if len(np.unique(y_true.astype(int))) < 2:
        return np.nan
    if kind == "pr":
        return float(average_precision_score(y_true, score))
    return float(roc_auc_score(y_true, score))


def evaluate_predictions(
    horizon: int,
    split: dict[str, pd.DataFrame],
    model_name: str,
    prob: np.ndarray,
    pred: np.ndarray,
    threshold: float,
) -> dict[str, object]:
    test = split["test"]
    y_true = test["y"].to_numpy(dtype=int)
    row = metric_row(y_true, type("Prediction", (), {"model": model_name, "threshold": threshold, "prob": prob, "pred": pred})())
    row["pr_auc"] = _safe_auc(y_true, prob, "pr")
    row["roc_auc"] = _safe_auc(y_true, prob, "roc")
    row["horizon_h"] = int(horizon)
    row["n_train"] = int(len(split["train"]))
    row["n_validation"] = int(len(split["validation"]))
    row["n_test"] = int(len(test))
    row["test_positive_rate"] = float(y_true.mean()) if len(y_true) else np.nan
    row["n_test_positive"] = int(y_true.sum())
    return row


def evaluate_horizon(frame: pd.DataFrame, horizon: int, seed: int = 2026) -> dict[str, object]:
    split = split_train_validation_test(frame)
    rows = []
    predictions = {}
    train_y = split["train"]["y"].to_numpy(dtype=int)
    majority = majority_predict(train_y, split["test"])
    flicker = flicker_predict(split["test"])
    for prediction in [majority, flicker]:
        rows.append(evaluate_predictions(horizon, split, prediction.model, prediction.prob, prediction.pred, prediction.threshold))
        predictions[prediction.model] = prediction
    for model_name in LEARNED_MODEL_NAMES:
        prediction = learned_prediction(model_name, split["train"], split["validation"], split["test"], seed)
        rows.append(evaluate_predictions(horizon, split, prediction.model, prediction.prob, prediction.pred, prediction.threshold))
        predictions[prediction.model] = prediction
    return {
        "metrics": pd.DataFrame(rows).loc[:, [
            "horizon_h",
            "model",
            "n_train",
            "n_validation",
            "n_test",
            "n_test_positive",
            "test_positive_rate",
            "threshold",
            "precision",
            "recall",
            "f1",
            "pr_auc",
            "roc_auc",
        ]],
        "split": split,
        "predictions": predictions,
    }


def event_recall(
    test_frame: pd.DataFrame,
    events: pd.DataFrame,
    horizon: int,
    pred: np.ndarray,
    cutoff: pd.Timestamp = CUTOFF_UTC,
) -> dict[str, object]:
    frame = test_frame.loc[:, ["station_id", "hour_utc"]].copy()
    frame["pred"] = np.asarray(pred, dtype=int)
    events = events.copy()
    events["start_utc"] = pd.to_datetime(events["start_utc"], utc=True, errors="coerce")
    events = events.loc[events["start_utc"].ge(cutoff)].copy()
    leads = []
    recalled = 0
    for row in events.itertuples(index=False):
        station = str(row.station_id)
        start = row.start_utc
        begin = start - pd.Timedelta(hours=int(horizon))
        window = frame.loc[
            frame["station_id"].eq(station)
            & frame["hour_utc"].gt(begin)
            & frame["hour_utc"].le(start)
        ]
        positives = window.loc[window["pred"].eq(1)].sort_values("hour_utc", kind="mergesort")
        if not positives.empty:
            recalled += 1
            leads.append(float((start - positives["hour_utc"].iloc[0]) / pd.Timedelta(hours=1)))
    n_events = int(len(events))
    return {
        "n_test_events": n_events,
        "event_recall": float(recalled / n_events) if n_events else np.nan,
        "median_lead_time_h": float(np.median(leads)) if leads else np.nan,
    }


def evaluate_all(
    dataset: RiskDataset | None = None,
    events: pd.DataFrame | None = None,
    seed: int = 2026,
) -> dict[str, pd.DataFrame]:
    risk = build_risk_dataset() if dataset is None else dataset
    events_frame = pd.read_parquet(AVAILABILITY_EVENTS_PATH) if events is None else events.copy()
    metric_frames = []
    event_rows = []
    for horizon in HORIZONS:
        horizon_frame = risk.for_horizon(horizon)
        result = evaluate_horizon(horizon_frame, horizon, seed)
        metric_frames.append(result["metrics"])
        for model_name, prediction in result["predictions"].items():
            row = event_recall(
                result["split"]["test"],
                events_frame,
                horizon,
                prediction.pred,
            )
            event_rows.append({"horizon_h": horizon, "model": model_name, **row})
    return {
        "hour_metrics": pd.concat(metric_frames, ignore_index=True),
        "event_metrics": pd.DataFrame(event_rows),
    }
