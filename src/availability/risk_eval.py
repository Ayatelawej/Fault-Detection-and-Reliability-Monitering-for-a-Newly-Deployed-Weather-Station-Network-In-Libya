from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)

from src.availability.risk_dataset import (
    FEATURE_COLUMNS,
    FaultRiskDataset,
    HORIZONS,
    IncidentHazardDataset,
    RiskDataset,
    build_risk_dataset,
)
from src.availability.risk_model import (
    LEARNED_MODEL_NAMES,
    flicker_predict,
    learned_prediction,
    majority_predict,
    metric_row,
)
from src.config.paths import AVAILABILITY_EVENTS_PATH

TRAIN_TIMESTAMP_FRACTION = 0.70
VALIDATION_TIMESTAMP_FRACTION = 0.15
MIN_MEANINGFUL_POSITIVES = 20


def _require_columns(frame: pd.DataFrame, required: list[str]) -> None:
    missing = sorted(set(required).difference(frame.columns))
    if missing:
        raise KeyError(missing)


def _prepare_split_frame(frame: pd.DataFrame, horizon: int) -> pd.DataFrame:
    _require_columns(frame, ["station_id", "hour_utc", "y"])
    out = frame.copy(deep=True)
    out["station_id"] = out["station_id"].astype(str)
    out["hour_utc"] = pd.to_datetime(out["hour_utc"], utc=True, errors="coerce")
    out["y"] = pd.to_numeric(out["y"], errors="coerce").fillna(0).astype(int)
    if "label_end_utc" not in out.columns:
        out["label_end_utc"] = out["hour_utc"] + pd.Timedelta(hours=int(horizon))
    else:
        out["label_end_utc"] = pd.to_datetime(
            out["label_end_utc"],
            utc=True,
            errors="coerce",
        )
    out = out.loc[
        out["station_id"].notna()
        & out["hour_utc"].notna()
        & out["label_end_utc"].notna()
    ].copy()
    if out.duplicated(["station_id", "hour_utc"]).any():
        raise ValueError("risk split input contains duplicate station-hour rows")
    return out.sort_values(["hour_utc", "station_id"], kind="mergesort").reset_index(
        drop=True
    )


def _prepare_timestamp_split_frame(
    frame: pd.DataFrame,
    *,
    target_columns: tuple[str, ...],
) -> pd.DataFrame:
    _require_columns(frame, ["station_id", "hour_utc", "label_end_utc"])
    _require_columns(frame, list(target_columns))
    out = frame.copy(deep=True)
    for column in target_columns:
        if not pd.api.types.is_numeric_dtype(out[column]):
            raise TypeError(f"timestamp split target must be numeric: {column}")
    out["station_id"] = out["station_id"].astype(str)
    out["hour_utc"] = pd.to_datetime(out["hour_utc"], utc=True, errors="coerce")
    out["label_end_utc"] = pd.to_datetime(
        out["label_end_utc"],
        utc=True,
        errors="coerce",
    )
    out = out.loc[
        out["station_id"].notna()
        & out["hour_utc"].notna()
        & out["label_end_utc"].notna()
    ].copy()
    if out.duplicated(["station_id", "hour_utc"]).any():
        raise ValueError("timestamp split input contains duplicate station-hour rows")
    return out.sort_values(["hour_utc", "station_id"], kind="mergesort").reset_index(
        drop=True
    )


def _label_span_hours(frame: pd.DataFrame, horizon: int) -> int:
    spans = (frame["label_end_utc"] - frame["hour_utc"]).dt.total_seconds() / 3600.0
    if spans.empty or spans.isna().any() or (spans < 0.0).any():
        raise ValueError("risk labels must have a non-negative, known label span")
    maximum = float(spans.max())
    rounded = int(round(maximum))
    if not np.isclose(maximum, rounded):
        raise ValueError("risk label span must resolve to whole clock-hours")
    if rounded < int(horizon):
        raise ValueError("risk label span cannot be shorter than its prediction horizon")
    return rounded


def _label_span_from_endpoints(frame: pd.DataFrame) -> int:
    spans = (frame["label_end_utc"] - frame["hour_utc"]).dt.total_seconds() / 3600.0
    if spans.empty or spans.isna().any() or (spans < 0.0).any():
        raise ValueError("timestamp split labels must have a non-negative, known span")
    maximum = float(spans.max())
    rounded = int(round(maximum))
    if not np.isclose(maximum, rounded):
        raise ValueError("timestamp split label span must resolve to whole clock-hours")
    return rounded


def _timestamp_boundaries(
    timestamps: pd.DatetimeIndex,
    *,
    train_fraction: float,
    validation_fraction: float,
    validation_start_utc: pd.Timestamp | None,
    test_start_utc: pd.Timestamp | None,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    if (validation_start_utc is None) != (test_start_utc is None):
        raise ValueError(
            "validation_start_utc and test_start_utc must be supplied together"
        )
    if validation_start_utc is not None and test_start_utc is not None:
        validation_start = pd.Timestamp(validation_start_utc)
        test_start = pd.Timestamp(test_start_utc)
        validation_start = (
            validation_start.tz_localize("UTC")
            if validation_start.tzinfo is None
            else validation_start.tz_convert("UTC")
        )
        test_start = (
            test_start.tz_localize("UTC")
            if test_start.tzinfo is None
            else test_start.tz_convert("UTC")
        )
        if validation_start >= test_start:
            raise ValueError("validation boundary must precede test boundary")
        return validation_start, test_start

    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must lie between zero and one")
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must lie between zero and one")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("train and validation fractions must leave a test period")
    if len(timestamps) < 3:
        raise ValueError("at least three unique timestamps are required for a split")

    n_train = max(1, int(np.floor(len(timestamps) * train_fraction)))
    n_validation = max(1, int(np.floor(len(timestamps) * validation_fraction)))
    if n_train + n_validation >= len(timestamps):
        raise ValueError("timestamp fractions leave no timestamp for the test period")
    return timestamps[n_train], timestamps[n_train + n_validation]


def split_timestamp_partitions(
    frame: pd.DataFrame,
    *,
    target_columns: tuple[str, ...] = (),
    train_fraction: float = TRAIN_TIMESTAMP_FRACTION,
    validation_fraction: float = VALIDATION_TIMESTAMP_FRACTION,
    validation_start_utc: pd.Timestamp | None = None,
    test_start_utc: pd.Timestamp | None = None,
    horizon_h: int | None = None,
) -> dict[str, object]:
    ordered = _prepare_timestamp_split_frame(
        frame,
        target_columns=tuple(target_columns),
    )
    label_span_hours = _label_span_from_endpoints(ordered)
    timestamps = pd.DatetimeIndex(ordered["hour_utc"].drop_duplicates().sort_values())
    validation_start, test_start = _timestamp_boundaries(
        timestamps,
        train_fraction=train_fraction,
        validation_fraction=validation_fraction,
        validation_start_utc=validation_start_utc,
        test_start_utc=test_start_utc,
    )
    train_before_purge = ordered.loc[ordered["hour_utc"].lt(validation_start)].copy()
    validation_before_purge = ordered.loc[
        ordered["hour_utc"].ge(validation_start)
        & ordered["hour_utc"].lt(test_start)
    ].copy()
    test = ordered.loc[ordered["hour_utc"].ge(test_start)].copy()

    train_purge_start = validation_start - pd.Timedelta(hours=label_span_hours)
    validation_purge_start = test_start - pd.Timedelta(hours=label_span_hours)
    purged_before_validation = train_before_purge.loc[
        train_before_purge["hour_utc"].ge(train_purge_start)
    ].copy()
    purged_before_validation["purge_boundary"] = "train_to_validation"
    purged_before_test = validation_before_purge.loc[
        validation_before_purge["hour_utc"].ge(validation_purge_start)
    ].copy()
    purged_before_test["purge_boundary"] = "validation_to_test"

    train = train_before_purge.loc[
        train_before_purge["hour_utc"].lt(train_purge_start)
    ].copy()
    validation = validation_before_purge.loc[
        validation_before_purge["hour_utc"].lt(validation_purge_start)
    ].copy()
    purged = pd.concat(
        [purged_before_validation, purged_before_test],
        ignore_index=True,
    )
    metadata = {
        "label_span_hours": label_span_hours,
        "train_fraction": float(train_fraction),
        "validation_fraction": float(validation_fraction),
        "test_fraction": float(1.0 - train_fraction - validation_fraction),
        "validation_start_utc": validation_start,
        "test_start_utc": test_start,
        "train_purge_start_utc": train_purge_start,
        "validation_purge_start_utc": validation_purge_start,
        "purged_before_validation_rows": int(len(purged_before_validation)),
        "purged_before_test_rows": int(len(purged_before_test)),
    }
    if horizon_h is not None:
        metadata["horizon_h"] = int(horizon_h)
    result = {
        "train": train.reset_index(drop=True),
        "validation": validation.reset_index(drop=True),
        "test": test.reset_index(drop=True),
        "purged": purged.reset_index(drop=True),
        "metadata": metadata,
    }
    _assert_purged_split_invariants(result)
    return result


def split_train_validation_test(
    frame: pd.DataFrame,
    horizon: int,
    *,
    train_fraction: float = TRAIN_TIMESTAMP_FRACTION,
    validation_fraction: float = VALIDATION_TIMESTAMP_FRACTION,
    validation_start_utc: pd.Timestamp | None = None,
    test_start_utc: pd.Timestamp | None = None,
) -> dict[str, object]:
    horizon = int(horizon)
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    ordered = _prepare_split_frame(frame, horizon)
    label_span_hours = _label_span_hours(ordered, horizon)
    result = split_timestamp_partitions(
        ordered,
        target_columns=("y",),
        train_fraction=train_fraction,
        validation_fraction=validation_fraction,
        validation_start_utc=validation_start_utc,
        test_start_utc=test_start_utc,
        horizon_h=horizon,
    )
    metadata = result["metadata"]
    if not isinstance(metadata, dict):
        raise TypeError("split metadata must be a dictionary")
    if int(metadata["label_span_hours"]) != label_span_hours:
        raise RuntimeError("binary risk split label span changed during delegation")
    return result


def _assert_purged_split_invariants(split: dict[str, object]) -> None:
    metadata = split["metadata"]
    if not isinstance(metadata, dict):
        raise TypeError("split metadata must be a dictionary")
    frames = []
    for partition in ["train", "validation", "test", "purged"]:
        frame = split[partition]
        if not isinstance(frame, pd.DataFrame):
            raise TypeError(f"{partition} split must be a DataFrame")
        frames.append(frame.assign(_partition=partition))
    membership = pd.concat(frames, ignore_index=True)
    if membership.groupby("hour_utc")["_partition"].nunique().gt(1).any():
        raise RuntimeError("a timestamp appears in more than one risk partition")
    train = split["train"]
    validation = split["validation"]
    if not isinstance(train, pd.DataFrame) or not isinstance(validation, pd.DataFrame):
        raise TypeError("risk partitions must be DataFrames")
    if not train["label_end_utc"].lt(metadata["validation_start_utc"]).all():
        raise RuntimeError("a training label reaches the validation partition")
    if not validation["label_end_utc"].lt(metadata["test_start_utc"]).all():
        raise RuntimeError("a validation label reaches the test partition")


def _support_status(n_positive: int) -> str:
    if n_positive == 0:
        return "no_positive_examples"
    if n_positive < MIN_MEANINGFUL_POSITIVES:
        return f"limited_support_lt_{MIN_MEANINGFUL_POSITIVES}"
    return "adequate_positive_support"


def summarize_purged_split(split: dict[str, object]) -> tuple[pd.DataFrame, pd.DataFrame]:
    metadata = split["metadata"]
    if not isinstance(metadata, dict):
        raise TypeError("split metadata must be a dictionary")
    partition_rows = []
    for partition in ["train", "validation", "test"]:
        frame = split[partition]
        if not isinstance(frame, pd.DataFrame):
            raise TypeError(f"{partition} split must be a DataFrame")
        positives = int(frame["y"].sum())
        negatives = int(len(frame) - positives)
        partition_rows.append(
            {
                "horizon_h": int(metadata["horizon_h"]),
                "partition": partition,
                "n_rows": int(len(frame)),
                "n_timestamps": int(frame["hour_utc"].nunique()),
                "n_positive": positives,
                "positive_rate": float(positives / len(frame)) if len(frame) else np.nan,
                "n_negative": negatives,
                "negative_to_positive_ratio": (
                    float(negatives / positives) if positives else np.inf
                ),
                "positive_support_status": _support_status(positives),
                "validation_start_utc": metadata["validation_start_utc"],
                "test_start_utc": metadata["test_start_utc"],
            }
        )
    purge_rows = []
    purged = split["purged"]
    if not isinstance(purged, pd.DataFrame):
        raise TypeError("purged split must be a DataFrame")
    for boundary, count_key in [
        ("train_to_validation", "purged_before_validation_rows"),
        ("validation_to_test", "purged_before_test_rows"),
    ]:
        frame = purged.loc[purged["purge_boundary"].eq(boundary)]
        purge_rows.append(
            {
                "horizon_h": int(metadata["horizon_h"]),
                "purge_boundary": boundary,
                "n_rows_purged": int(metadata[count_key]),
                "n_timestamps_purged": int(frame["hour_utc"].nunique()),
                "first_purged_hour_utc": frame["hour_utc"].min() if not frame.empty else pd.NaT,
                "last_purged_hour_utc": frame["hour_utc"].max() if not frame.empty else pd.NaT,
            }
        )
    return pd.DataFrame(partition_rows), pd.DataFrame(purge_rows)


def build_label_split_characteristics(
    dataset: RiskDataset | FaultRiskDataset | IncidentHazardDataset,
    *,
    boundary_metadata_by_horizon: dict[int, dict[str, object]] | None = None,
    label_changes: pd.DataFrame | None = None,
) -> dict[str, object]:
    partition_frames = []
    purge_frames = []
    splits: dict[int, dict[str, object]] = {}
    for horizon in dataset.horizons:
        boundary_metadata = (
            None
            if boundary_metadata_by_horizon is None
            else boundary_metadata_by_horizon.get(int(horizon))
        )
        split = split_train_validation_test(
            dataset.for_horizon(horizon),
            horizon,
            **(
                {}
                if boundary_metadata is None
                else {
                    "validation_start_utc": boundary_metadata[
                        "validation_start_utc"
                    ],
                    "test_start_utc": boundary_metadata["test_start_utc"],
                }
            ),
        )
        partition_summary, purge_summary = summarize_purged_split(split)
        partition_frames.append(partition_summary)
        purge_frames.append(purge_summary)
        splits[int(horizon)] = split
    return {
        "partition_summary": pd.concat(partition_frames, ignore_index=True),
        "purge_summary": pd.concat(purge_frames, ignore_index=True),
        "label_changes": (
            (
                dataset.label_change_summary()
                if isinstance(dataset, RiskDataset)
                else dataset.construction_summary()
            )
            if label_changes is None
            else label_changes
        ),
        "splits": splits,
    }


def summarize_station_positive_support(
    splits: dict[int, dict[str, object]],
    near_zero_threshold: int = 5,
) -> pd.DataFrame:
    rows = []
    for horizon, split in splits.items():
        for partition in ["train", "validation", "test"]:
            frame = split[partition]
            if not isinstance(frame, pd.DataFrame):
                raise TypeError(f"{partition} split must be a DataFrame")
            summary = (
                frame.groupby("station_id", as_index=False)["y"]
                .agg(n_rows="size", positive_rows="sum")
                .sort_values("station_id", kind="mergesort")
            )
            for row in summary.itertuples(index=False):
                rows.append(
                    {
                        "horizon_h": int(horizon),
                        "partition": partition,
                        "station_id": str(row.station_id),
                        "n_rows": int(row.n_rows),
                        "positive_rows": int(row.positive_rows),
                        "positive_rate": (
                            float(row.positive_rows / row.n_rows)
                            if row.n_rows
                            else np.nan
                        ),
                        "near_zero_positives_le": int(near_zero_threshold),
                        "near_zero_positive_support": bool(
                            row.positive_rows <= near_zero_threshold
                        ),
                    }
                )
    return pd.DataFrame(rows)


def _safe_auc(y_true: np.ndarray, score: np.ndarray, kind: str) -> float:
    if len(np.unique(y_true.astype(int))) < 2:
        return np.nan
    if kind == "pr":
        return float(average_precision_score(y_true, score))
    return float(roc_auc_score(y_true, score))


def regression_metrics(
    y_true: np.ndarray | pd.Series,
    prediction: np.ndarray | pd.Series,
) -> dict[str, float]:
    actual = np.asarray(y_true, dtype=float).reshape(-1)
    predicted = np.asarray(prediction, dtype=float).reshape(-1)
    if actual.shape != predicted.shape:
        raise ValueError("regression target and prediction must have equal shapes")
    if not len(actual):
        raise ValueError("regression metrics require at least one observation")
    if not (np.isfinite(actual).all() and np.isfinite(predicted).all()):
        raise ValueError("regression metrics require finite values")
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        "r2": float(r2_score(actual, predicted)) if len(actual) > 1 else np.nan,
    }


def regression_error_improvement_percent(
    baseline_error: float,
    candidate_error: float,
) -> float:
    baseline = float(baseline_error)
    candidate = float(candidate_error)
    if not (np.isfinite(baseline) and np.isfinite(candidate)) or np.isclose(baseline, 0.0):
        return np.nan
    return float(100.0 * (baseline - candidate) / baseline)


def evaluate_predictions(
    horizon: int,
    split: dict[str, object],
    model_name: str,
    prob: np.ndarray,
    pred: np.ndarray,
    threshold: float,
) -> dict[str, object]:
    test = split["test"]
    if not isinstance(test, pd.DataFrame):
        raise TypeError("test split must be a DataFrame")
    y_true = test["y"].to_numpy(dtype=int)
    row = metric_row(
        y_true,
        type(
            "Prediction",
            (),
            {
                "model": model_name,
                "threshold": threshold,
                "prob": prob,
                "pred": pred,
            },
        )(),
    )
    row["pr_auc"] = _safe_auc(y_true, prob, "pr")
    row["roc_auc"] = _safe_auc(y_true, prob, "roc")
    row["horizon_h"] = int(horizon)
    for partition in ["train", "validation", "test"]:
        part = split[partition]
        if not isinstance(part, pd.DataFrame):
            raise TypeError(f"{partition} split must be a DataFrame")
        row[f"n_{partition}"] = int(len(part))
    row["test_positive_rate"] = float(y_true.mean()) if len(y_true) else np.nan
    row["n_test_positive"] = int(y_true.sum())
    return row


def evaluate_horizon(
    frame: pd.DataFrame,
    horizon: int,
    seed: int = 2026,
) -> dict[str, object]:
    split = split_train_validation_test(frame, horizon)
    train = split["train"]
    validation = split["validation"]
    test = split["test"]
    if not all(isinstance(part, pd.DataFrame) for part in [train, validation, test]):
        raise TypeError("risk partitions must be DataFrames")
    rows = []
    predictions = {}
    train_y = train["y"].to_numpy(dtype=int)
    majority = majority_predict(train_y, test)
    flicker = flicker_predict(test)
    for prediction in [majority, flicker]:
        rows.append(
            evaluate_predictions(
                horizon,
                split,
                prediction.model,
                prediction.prob,
                prediction.pred,
                prediction.threshold,
            )
        )
        predictions[prediction.model] = prediction
    for model_name in LEARNED_MODEL_NAMES:
        prediction = learned_prediction(model_name, train, validation, test, seed)
        rows.append(
            evaluate_predictions(
                horizon,
                split,
                prediction.model,
                prediction.prob,
                prediction.pred,
                prediction.threshold,
            )
        )
        predictions[prediction.model] = prediction
    return {
        "metrics": pd.DataFrame(rows).loc[
            :,
            [
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
            ],
        ],
        "split": split,
        "predictions": predictions,
    }


def event_recall(
    test_frame: pd.DataFrame,
    events: pd.DataFrame,
    horizon: int,
    pred: np.ndarray,
    test_start_utc: pd.Timestamp | None = None,
) -> dict[str, object]:
    frame = test_frame.loc[:, ["station_id", "hour_utc"]].copy()
    frame["hour_utc"] = pd.to_datetime(frame["hour_utc"], utc=True, errors="coerce")
    frame["pred"] = np.asarray(pred, dtype=int)
    start = (
        pd.to_datetime(frame["hour_utc"], utc=True, errors="coerce").min()
        if test_start_utc is None
        else pd.Timestamp(test_start_utc)
    )
    if pd.notna(start) and start.tzinfo is None:
        start = start.tz_localize("UTC")
    elif pd.notna(start):
        start = start.tz_convert("UTC")
    events = events.copy()
    events["start_utc"] = pd.to_datetime(
        events["start_utc"],
        utc=True,
        errors="coerce",
    )
    events = events.loc[events["start_utc"].ge(start)].copy()
    leads = []
    recalled = 0
    for row in events.itertuples(index=False):
        station = str(row.station_id)
        event_start = row.start_utc
        begin = event_start - pd.Timedelta(hours=int(horizon))
        window = frame.loc[
            frame["station_id"].eq(station)
            & frame["hour_utc"].gt(begin)
            & frame["hour_utc"].le(event_start)
        ]
        positives = window.loc[window["pred"].eq(1)].sort_values(
            "hour_utc",
            kind="mergesort",
        )
        if not positives.empty:
            recalled += 1
            leads.append(
                float(
                    (event_start - positives["hour_utc"].iloc[0])
                    / pd.Timedelta(hours=1)
                )
            )
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
    events_frame = (
        pd.read_parquet(AVAILABILITY_EVENTS_PATH)
        if events is None
        else events.copy()
    )
    metric_frames = []
    event_rows = []
    for horizon in HORIZONS:
        horizon_frame = risk.for_horizon(horizon)
        result = evaluate_horizon(horizon_frame, horizon, seed)
        metric_frames.append(result["metrics"])
        metadata = result["split"]["metadata"]
        if not isinstance(metadata, dict):
            raise TypeError("split metadata must be a dictionary")
        for model_name, prediction in result["predictions"].items():
            row = event_recall(
                result["split"]["test"],
                events_frame,
                horizon,
                prediction.pred,
                test_start_utc=metadata["test_start_utc"],
            )
            event_rows.append({"horizon_h": horizon, "model": model_name, **row})
    return {
        "hour_metrics": pd.concat(metric_frames, ignore_index=True),
        "event_metrics": pd.DataFrame(event_rows),
    }
