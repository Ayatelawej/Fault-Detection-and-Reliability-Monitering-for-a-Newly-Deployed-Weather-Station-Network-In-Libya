from __future__ import annotations

import numpy as np
import pandas as pd

from src.model.feature_spec import (
    CONTINUOUS_FEATURES,
    RULE_EVIDENCE_FLAGS,
    STATIC_FEATURES,
    rule_evidence_feature_names,
)


TRAILING_HOURS = 7
RULE_SUMMARY_SOURCES = (
    "r_pressure",
    "offset_level_pressure",
    "z_spatial_pressure",
    "ext_abs_z_array_mean",
)
RULE_SUMMARY_STATS = ("mean", "min", "max", "absmax")
CAUSAL_RULE_EVIDENCE_FEATURE_NAMES = tuple(
    f"rbp_{source}_{stat}"
    for stat in RULE_SUMMARY_STATS
    for source in RULE_SUMMARY_SOURCES
)
CAUSAL_STATIC_FEATURE_NAMES = (
    "trailing_max_non_group_detector_run",
    "trailing_observed_fraction_hours_ge2_sensor_groups",
    "trailing_distinct_sensor_groups",
    "trailing_pressure_agreement_strength_mean",
    "trailing_pressure_agreement_sign_fraction",
    "trailing_abs_r_pressure_mean",
)
GROUP_PREFIX = "stat_sensor_group_flag_"
FLAG_PREFIX = "stat_flag_"
TIME_COLUMNS = ("hour", "hour_utc", "time_utc", "timestamp")


def _time_column(columns: list[str]) -> str:
    for column in TIME_COLUMNS:
        if column in columns:
            return column
    raise KeyError("raw hourly values need a timestamp column")


def _feature_names(
    examples: dict[str, np.ndarray],
    key: str,
    expected: tuple[str, ...] | list[str],
) -> np.ndarray:
    values = examples.get(key)
    if values is None:
        return np.asarray(expected, dtype=object)
    names = np.asarray(values, dtype=object)
    if len(names) != len(expected):
        raise ValueError(f"{key} does not match its base width")
    return names.copy()


def _validate_examples(examples: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    required = {
        "X_cont",
        "mask",
        "time_since_last",
        "static",
        "rule_evidence",
        "station_id",
        "hour",
    }
    missing = sorted(required.difference(examples))
    if missing:
        raise KeyError(f"hourly examples are missing: {missing}")
    x_cont = np.asarray(examples["X_cont"], dtype=np.float32)
    mask = np.asarray(examples["mask"], dtype=np.float32)
    static = np.asarray(examples["static"], dtype=np.float32)
    rule_evidence = np.asarray(examples["rule_evidence"], dtype=np.float32)
    count = x_cont.shape[0]
    if x_cont.ndim != 3 or x_cont.shape[1] != TRAILING_HOURS or x_cont.shape[2] != len(CONTINUOUS_FEATURES):
        raise ValueError("causal tuning features require seven hourly continuous values")
    if mask.ndim != 3 or mask.shape[:2] != (count, TRAILING_HOURS):
        raise ValueError("hourly masks do not match the seven-hour input")
    if mask.shape[2] not in (1, len(CONTINUOUS_FEATURES)):
        raise ValueError("hourly masks must be per-hour or per-feature")
    if np.asarray(examples["time_since_last"]).shape != (count, TRAILING_HOURS, 1):
        raise ValueError("elapsed-hour values do not match the hourly mask")
    if static.shape != (count, len(STATIC_FEATURES)):
        raise ValueError("causal tuning features require the three base static values")
    if rule_evidence.shape != (count, len(RULE_EVIDENCE_FLAGS) * 2):
        raise ValueError("causal tuning features require the base rule evidence values")
    if len(np.asarray(examples["station_id"])) != count or len(np.asarray(examples["hour"])) != count:
        raise ValueError("station-hour keys do not match hourly examples")
    return x_cont, mask, static


def _masked_continuous(x_cont: np.ndarray, mask: np.ndarray) -> np.ndarray:
    observed = mask.astype(bool)
    if observed.shape[2] == 1:
        observed = observed[:, :, 0:1]
    return np.where(observed, x_cont, np.nan).astype(np.float32)


def _safe_mean(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    count = valid.sum(axis=1, dtype=np.int64)
    total = np.where(valid, values, 0.0).sum(axis=1, dtype=np.float64)
    return np.divide(
        total,
        count,
        out=np.full(total.shape, np.nan, dtype=np.float64),
        where=count > 0,
    ).astype(np.float32)


def _rule_summaries(x_cont: np.ndarray, mask: np.ndarray) -> np.ndarray:
    source_indices = [CONTINUOUS_FEATURES.index(name) for name in RULE_SUMMARY_SOURCES]
    values = _masked_continuous(x_cont, mask)[:, :, source_indices]
    finite = np.isfinite(values)
    count = finite.sum(axis=1, dtype=np.int64)
    means = _safe_mean(values, finite)
    minimums = np.where(finite, values, np.inf).min(axis=1)
    maximums = np.where(finite, values, -np.inf).max(axis=1)
    absmaximums = np.where(finite, np.abs(values), -np.inf).max(axis=1)
    for summary in (minimums, maximums, absmaximums):
        summary[count == 0] = np.nan
    return np.concatenate(
        [means, minimums, maximums, absmaximums],
        axis=1,
    ).astype(np.float32)


def _pressure_summaries(x_cont: np.ndarray, mask: np.ndarray) -> np.ndarray:
    values = _masked_continuous(x_cont, mask)
    external = values[:, :, CONTINUOUS_FEATURES.index("r_pressure")]
    spatial = values[:, :, CONTINUOUS_FEATURES.index("spatial_offset_level_pressure")]
    valid = (
        np.isfinite(external)
        & np.isfinite(spatial)
        & (np.abs(external) > 1e-9)
        & (np.abs(spatial) > 1e-9)
    )
    agree = valid & (np.sign(external) == np.sign(spatial))
    valid_count = valid.sum(axis=1, dtype=np.int64)
    agree_count = agree.sum(axis=1, dtype=np.int64)
    strength_values = np.minimum(np.abs(external), np.abs(spatial))
    strength = _safe_mean(strength_values, agree)
    sign_fraction = np.divide(
        agree_count,
        valid_count,
        out=np.full(len(values), np.nan, dtype=np.float64),
        where=valid_count > 0,
    ).astype(np.float32)
    abs_external = _safe_mean(np.abs(external), np.isfinite(external))
    return np.column_stack([strength, sign_fraction, abs_external]).astype(np.float32)


def _prepare_raw(raw_hourly: pd.DataFrame) -> pd.DataFrame:
    time_column = _time_column(list(raw_hourly.columns))
    required = {"station_id", time_column, "data_present"}
    missing = sorted(required.difference(raw_hourly.columns))
    if missing:
        raise KeyError(f"raw hourly values are missing: {missing}")
    detector_columns = sorted(
        column for column in raw_hourly.columns if column.startswith(FLAG_PREFIX)
    )
    group_columns = sorted(
        column for column in raw_hourly.columns if column.startswith(GROUP_PREFIX)
    )
    keep = ["station_id", time_column, "data_present", *detector_columns, *group_columns]
    result = raw_hourly.loc[:, list(dict.fromkeys(keep))].copy()
    result = result.rename(columns={time_column: "hour"})
    result["station_id"] = result["station_id"].astype(str)
    result["hour"] = pd.to_datetime(result["hour"], utc=True, format="mixed")
    if result.duplicated(["station_id", "hour"]).any():
        raise ValueError("raw hourly values have duplicate station-hour rows")
    return result.sort_values(["station_id", "hour"]).reset_index(drop=True)


def _max_run(values: np.ndarray) -> np.ndarray:
    current = np.zeros(values.shape[0], dtype=np.int64)
    longest = np.zeros(values.shape[0], dtype=np.int64)
    for offset in range(values.shape[1]):
        current = np.where(values[:, offset], current + 1, 0)
        longest = np.maximum(longest, current)
    return longest.astype(np.float32)


def _station_raw_summaries(
    station_raw: pd.DataFrame,
    target_hours: pd.DatetimeIndex,
) -> np.ndarray:
    if station_raw.empty:
        raise ValueError("raw hourly values lack a station used by the tensor")
    start = target_hours.min() - pd.Timedelta(hours=TRAILING_HOURS - 1)
    grid = pd.date_range(start=start, end=target_hours.max(), freq="h")
    indexed = station_raw.set_index("hour").reindex(grid)
    target_rows = indexed.reindex(target_hours)
    if target_rows["data_present"].isna().any():
        raise ValueError("raw hourly values lack a tensor station-hour row")
    positions = grid.get_indexer(target_hours)
    offsets = np.arange(TRAILING_HOURS, dtype=np.int64)
    window_indices = positions[:, None] - (TRAILING_HOURS - 1) + offsets
    present = pd.to_numeric(indexed["data_present"], errors="coerce").fillna(0.0).gt(0.0).to_numpy(dtype=bool)
    window_present = present[window_indices]
    detector_columns = [
        column
        for column in indexed.columns
        if column.startswith(FLAG_PREFIX) and not column.startswith(GROUP_PREFIX)
    ]
    if detector_columns:
        detector_values = indexed.loc[:, detector_columns].apply(
            pd.to_numeric,
            errors="coerce",
        ).fillna(0.0).to_numpy(dtype=np.float32)
        detector_any = (detector_values > 0.0).any(axis=1)
        detector_any &= present
        detector_run = _max_run(detector_any[window_indices])
    else:
        detector_run = np.zeros(len(target_hours), dtype=np.float32)
    group_columns = [column for column in indexed.columns if column.startswith(GROUP_PREFIX)]
    if group_columns:
        group_values = indexed.loc[:, group_columns].apply(
            pd.to_numeric,
            errors="coerce",
        ).fillna(0.0).to_numpy(dtype=np.float32)
        group_values[~present] = 0.0
        window_groups = group_values[window_indices]
        group_counts = (window_groups > 0.0).sum(axis=2, dtype=np.int64)
        denominator = window_present.sum(axis=1, dtype=np.int64)
        fraction = np.divide(
            ((group_counts >= 2) & window_present).sum(axis=1, dtype=np.int64),
            denominator,
            out=np.zeros(len(target_hours), dtype=np.float64),
            where=denominator > 0,
        ).astype(np.float32)
        distinct = (window_groups > 0.0).any(axis=1).sum(axis=1, dtype=np.int64).astype(np.float32)
    else:
        fraction = np.zeros(len(target_hours), dtype=np.float32)
        distinct = np.zeros(len(target_hours), dtype=np.float32)
    return np.column_stack([detector_run, fraction, distinct]).astype(np.float32)


def _raw_summaries(examples: dict[str, np.ndarray], raw_hourly: pd.DataFrame) -> np.ndarray:
    raw = _prepare_raw(raw_hourly)
    stations = np.asarray(examples["station_id"], dtype=object).astype(str)
    hours = pd.DatetimeIndex(
        pd.to_datetime(pd.Series(examples["hour"]), utc=True, format="mixed")
    )
    result = np.zeros((len(stations), 3), dtype=np.float32)
    for station_id in np.unique(stations):
        sample_indices = np.flatnonzero(stations == station_id)
        station_hours = hours.take(sample_indices)
        station_raw = raw.loc[raw["station_id"].eq(station_id)]
        result[sample_indices] = _station_raw_summaries(station_raw, station_hours)
    return result


def augment_hourly_rgfn_examples(
    examples: dict[str, np.ndarray],
    raw_hourly: pd.DataFrame,
) -> dict[str, np.ndarray]:
    x_cont, mask, static = _validate_examples(examples)
    rule_summary = _rule_summaries(x_cont, mask)
    raw_summary = _raw_summaries(examples, raw_hourly)
    pressure_summary = _pressure_summaries(x_cont, mask)
    static_names = _feature_names(examples, "static_feature_names", STATIC_FEATURES)
    rule_names = _feature_names(
        examples,
        "rule_evidence_feature_names",
        rule_evidence_feature_names(),
    )
    result = dict(examples)
    result["static"] = np.concatenate(
        [static.copy(), raw_summary, pressure_summary],
        axis=1,
    ).astype(np.float32)
    result["rule_evidence"] = np.concatenate(
        [np.asarray(examples["rule_evidence"], dtype=np.float32).copy(), rule_summary],
        axis=1,
    ).astype(np.float32)
    result["static_feature_names"] = np.concatenate(
        [static_names, np.asarray(CAUSAL_STATIC_FEATURE_NAMES, dtype=object)],
    )
    result["rule_evidence_feature_names"] = np.concatenate(
        [rule_names, np.asarray(CAUSAL_RULE_EVIDENCE_FEATURE_NAMES, dtype=object)],
    )
    return result
