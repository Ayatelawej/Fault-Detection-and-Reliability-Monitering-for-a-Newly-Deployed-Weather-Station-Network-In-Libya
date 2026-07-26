from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.model.feature_spec import (
    COMPONENT_LABEL_NAMES,
    CONTINUOUS_FEATURES,
    MECHANISM_LABEL_NAMES,
    RULE_EVIDENCE_FLAGS,
    STATIC_FEATURES,
    rule_evidence_feature_names,
)
from src.rules.labelling import period_for_start


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_PATH = PROJECT_ROOT / "data" / "merged" / "station_hourly_merged.csv"
FEATURE_PATH = PROJECT_ROOT / "data" / "features" / "feature_matrix.parquet"
LABEL_PATH = PROJECT_ROOT / "data" / "labels" / "episode_labels.csv"
HOURLY_DATA_DIR = PROJECT_ROOT / "data" / "hourly_detection"
HOURLY_LABEL_PATH = HOURLY_DATA_DIR / "hourly_labels.csv"
SHORT_TENSOR_PATH = HOURLY_DATA_DIR / "hourly_detection_short.npz"
LONG_TENSOR_PATH = HOURLY_DATA_DIR / "hourly_detection_long.npz"
SHORT_WINDOW_HOURS = 7
LONG_WINDOW_HOURS = 49
MASK_MODE_PER_HOUR = "per_hour"
MASK_MODE_PER_FEATURE = "per_feature"
MASK_MODES = (MASK_MODE_PER_HOUR, MASK_MODE_PER_FEATURE)

LABEL_OUTPUT_COLUMNS = [
    "station_id",
    "hour",
    "fault_hour",
    "display_state",
    "mechanisms",
    "components",
    "detectors_fired",
]
EPISODE_REQUIRED_COLUMNS = [
    "episode_id",
    "station_id",
    "start_hour",
    "end_hour",
    "binary_fault",
    "label_state",
    "mechanisms",
    "components",
]
DETECTOR_GROUPS = (
    ("robust_zscore", "stat_flag_zscore_"),
    ("isolation_forest", "stat_flag_iforest_"),
    ("stuck_rolling_variance", "stat_flag_stuck_"),
    ("physical_limit", "stat_flag_physical_"),
)
PHYSICAL_SUSPECT_PREFIX = "stat_flag_physical_suspect_"
TIME_COLUMNS = ("hour", "hour_utc", "time_utc", "timestamp")


def _time_column(columns: list[str]) -> str:
    for column in TIME_COLUMNS:
        if column in columns:
            return column
    raise KeyError(f"missing timestamp column; expected one of {TIME_COLUMNS}")


def _tokens(value: object) -> set[str]:
    if value is None:
        return set()
    missing = pd.isna(value)
    if missing is pd.NA:
        return set()
    if isinstance(missing, (bool, np.bool_)) and missing:
        return set()
    return {item for item in str(value).split("|") if item}


def _serialise(values: set[str], order: tuple[str, ...]) -> str:
    return "|".join(item for item in order if item in values)


def _union_serialised(current: object, incoming: object, order: tuple[str, ...]) -> str:
    return _serialise(_tokens(current) | _tokens(incoming), order)


def _union_episode_ids(current: object, incoming: object) -> str:
    return "|".join(sorted(_tokens(current) | {str(incoming)}))


def detector_columns_by_group(columns: list[str]) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for name, prefix in DETECTOR_GROUPS:
        matched = [column for column in columns if column.startswith(prefix)]
        if name == "physical_limit":
            matched = [
                column
                for column in matched
                if not column.startswith(PHYSICAL_SUSPECT_PREFIX)
            ]
        result[name] = sorted(matched)
    return result


def _numeric(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    return frame.loc[:, columns].apply(pd.to_numeric, errors="coerce")


def prepare_hourly_frame(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    time_column = _time_column(list(result.columns))
    if time_column != "hour":
        result = result.rename(columns={time_column: "hour"})
    required = ["station_id", "hour", "data_present", *CONTINUOUS_FEATURES, *STATIC_FEATURES, *RULE_EVIDENCE_FLAGS]
    missing = sorted(set(required).difference(result.columns))
    if missing:
        raise KeyError(f"hourly feature fields missing: {missing}")
    result["station_id"] = result["station_id"].astype(str)
    result["hour"] = pd.to_datetime(result["hour"], utc=True, format="mixed")
    result["data_present"] = pd.to_numeric(result["data_present"], errors="coerce").fillna(0).astype(np.int8)
    if result.duplicated(["station_id", "hour"]).any():
        raise ValueError("hourly source has duplicate station-hour rows")
    return result.sort_values(["station_id", "hour"]).reset_index(drop=True)


def load_hourly_frame(
    source_path: Path = SOURCE_PATH,
    feature_path: Path = FEATURE_PATH,
) -> pd.DataFrame:
    source = pd.read_csv(source_path)
    source_time = _time_column(list(source.columns))
    source = source.rename(columns={source_time: "hour"})
    source_required = ["station_id", "hour", "data_present"]
    source_missing = sorted(set(source_required).difference(source.columns))
    if source_missing:
        raise KeyError(f"hourly source fields missing: {source_missing}")
    source = source.loc[:, [column for column in ["station_id", "hour", "data_present", "n_raw_records"] if column in source.columns]].copy()
    source["station_id"] = source["station_id"].astype(str)
    source["hour"] = pd.to_datetime(source["hour"], utc=True, format="mixed")

    features = pd.read_parquet(feature_path)
    feature_time = _time_column(list(features.columns))
    features = features.rename(columns={feature_time: "hour"})
    detector_columns = sorted(
        column for column in features.columns if column.startswith("stat_flag_")
    )
    required_features = [
        "station_id",
        "hour",
        *CONTINUOUS_FEATURES,
        *STATIC_FEATURES,
        *RULE_EVIDENCE_FLAGS,
        *detector_columns,
    ]
    feature_missing = sorted(set(required_features).difference(features.columns))
    if feature_missing:
        raise KeyError(f"hourly feature fields missing: {feature_missing}")
    features = features.loc[:, list(dict.fromkeys(required_features))].copy()
    features["station_id"] = features["station_id"].astype(str)
    features["hour"] = pd.to_datetime(features["hour"], utc=True, format="mixed")
    if features.duplicated(["station_id", "hour"]).any():
        raise ValueError("feature lookup has duplicate station-hour rows")

    merged = source.merge(
        features,
        on=["station_id", "hour"],
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    unmatched = int(merged["_merge"].ne("both").sum())
    if unmatched:
        raise ValueError(f"hourly source rows without feature lookup rows: {unmatched}")
    return prepare_hourly_frame(merged.drop(columns="_merge"))


def prepare_labelled_episodes(frame: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(set(EPISODE_REQUIRED_COLUMNS).difference(frame.columns))
    if missing:
        raise KeyError(f"labelled episode fields missing: {missing}")
    result = frame.loc[:, EPISODE_REQUIRED_COLUMNS].copy()
    result["episode_id"] = result["episode_id"].astype(str)
    result["station_id"] = result["station_id"].astype(str)
    result["start_hour"] = pd.to_datetime(result["start_hour"], utc=True, format="mixed")
    result["end_hour"] = pd.to_datetime(result["end_hour"], utc=True, format="mixed")
    result["binary_fault"] = pd.to_numeric(result["binary_fault"], errors="coerce")
    valid_binary = result["binary_fault"].isin([0, 1])
    if result["binary_fault"].isna().any() or not bool(valid_binary.all()):
        raise ValueError("labelled binary_fault must contain only zero or one")
    states = {"fault", "benign", "borderline_review"}
    unknown_states = sorted(set(result["label_state"]).difference(states))
    if unknown_states:
        raise ValueError(f"unknown label states: {unknown_states}")
    if not result["end_hour"].ge(result["start_hour"]).all():
        raise ValueError("labelled episode ends before it starts")
    fault_state = result["label_state"].eq("fault")
    if not result.loc[fault_state, "binary_fault"].eq(1).all():
        raise ValueError("fault episodes must have binary_fault equal to one")
    if not result.loc[~fault_state, "binary_fault"].eq(0).all():
        raise ValueError("non-fault episodes must have binary_fault equal to zero")
    mechanism_values = set().union(*[_tokens(value) for value in result.loc[fault_state, "mechanisms"]])
    component_values = set().union(*[_tokens(value) for value in result.loc[fault_state, "components"]])
    unknown_mechanisms = sorted(mechanism_values.difference(MECHANISM_LABEL_NAMES))
    unknown_components = sorted(component_values.difference(COMPONENT_LABEL_NAMES))
    if unknown_mechanisms or unknown_components:
        raise ValueError(
            f"unsupported fault labels: mechanisms={unknown_mechanisms}, components={unknown_components}"
        )
    return result.sort_values(["station_id", "start_hour", "end_hour", "episode_id"]).reset_index(drop=True)


def load_labelled_episodes(path: Path = LABEL_PATH) -> pd.DataFrame:
    return prepare_labelled_episodes(pd.read_csv(path))


def _detector_display_values(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    grouped = detector_columns_by_group(list(frame.columns))
    available_columns = [column for columns in grouped.values() for column in columns]
    if not available_columns:
        raise KeyError("no detector evidence fields are available")
    available = _numeric(frame, available_columns).notna().any(axis=1).to_numpy(dtype=bool)
    values = np.full(len(frame), "", dtype=object)
    any_fired = np.zeros(len(frame), dtype=bool)
    for name, _ in DETECTOR_GROUPS:
        columns = grouped[name]
        if not columns:
            continue
        fired = _numeric(frame, columns).fillna(0.0).gt(0.0).any(axis=1).to_numpy(dtype=bool)
        any_fired |= fired
        existing = values[fired]
        values[fired] = np.where(existing == "", name, existing + "|" + name)
    return available, values, any_fired


def build_hourly_labels(
    hourly_frame: pd.DataFrame,
    labelled_episodes: pd.DataFrame,
) -> pd.DataFrame:
    hourly = prepare_hourly_frame(hourly_frame)
    episodes = prepare_labelled_episodes(labelled_episodes)
    detector_available, detectors_fired, any_detector_fired = _detector_display_values(hourly)
    count = len(hourly)
    fault_member = np.zeros(count, dtype=bool)
    review_member = np.zeros(count, dtype=bool)
    mechanisms = np.full(count, "", dtype=object)
    components = np.full(count, "", dtype=object)
    source_episode_ids = np.full(count, "", dtype=object)

    episode_by_station = {
        station_id: group
        for station_id, group in episodes.groupby("station_id", sort=False)
    }
    for station_id, station_hourly in hourly.groupby("station_id", sort=False):
        station_episodes = episode_by_station.get(station_id)
        if station_episodes is None:
            continue
        station_indices = station_hourly.index.to_numpy(dtype=np.int64)
        station_hours = station_hourly["hour"]
        for _, episode in station_episodes.iterrows():
            covered = station_hours.between(
                episode["start_hour"],
                episode["end_hour"],
                inclusive="both",
            )
            indices = station_indices[covered.to_numpy(dtype=bool)]
            if not len(indices):
                continue
            if episode["label_state"] == "borderline_review":
                review_member[indices] = True
                continue
            if episode["label_state"] != "fault":
                continue
            fault_member[indices] = True
            for index in indices:
                mechanisms[index] = _union_serialised(
                    mechanisms[index],
                    episode["mechanisms"],
                    MECHANISM_LABEL_NAMES,
                )
                components[index] = _union_serialised(
                    components[index],
                    episode["components"],
                    COMPONENT_LABEL_NAMES,
                )
                source_episode_ids[index] = _union_episode_ids(
                    source_episode_ids[index],
                    episode["episode_id"],
                )

    source_present = hourly["data_present"].eq(1).to_numpy(dtype=bool)
    trainable = source_present & detector_available & ~review_member
    display_state = np.full(count, "clean", dtype=object)
    display_state[~trainable] = "excluded"
    display_state[trainable & any_detector_fired & ~fault_member] = "benign"
    display_state[trainable & fault_member] = "fault"
    fault_hour = pd.Series(pd.NA, index=hourly.index, dtype="Int64")
    fault_hour.loc[trainable] = (trainable & fault_member)[trainable].astype(np.int8)
    non_fault = display_state != "fault"
    mechanisms[non_fault] = ""
    components[non_fault] = ""
    source_episode_ids[non_fault] = ""
    result = pd.DataFrame(
        {
            "station_id": hourly["station_id"].to_numpy(dtype=object),
            "hour": hourly["hour"].to_numpy(),
            "fault_hour": fault_hour,
            "display_state": display_state,
            "mechanisms": mechanisms,
            "components": components,
            "detectors_fired": detectors_fired,
            "source_episode_ids": source_episode_ids,
            "training_eligible": trainable,
            "data_present": source_present,
            "detector_available": detector_available,
        }
    )
    result["period"] = result["hour"].map(period_for_start)
    return result


def label_output_frame(labels: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(set(LABEL_OUTPUT_COLUMNS).difference(labels.columns))
    if missing:
        raise KeyError(f"hourly label fields missing: {missing}")
    result = labels.loc[:, LABEL_OUTPUT_COLUMNS].copy()
    result["hour"] = pd.to_datetime(result["hour"], utc=True).astype(str)
    return result


def write_hourly_labels(labels: pd.DataFrame, path: Path = HOURLY_LABEL_PATH) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    label_output_frame(labels).to_csv(destination, index=False)
    return destination


def _time_since_last(present: np.ndarray, window_hours: int) -> np.ndarray:
    values = np.zeros(len(present), dtype=np.float32)
    previous = None
    for index, value in enumerate(present.astype(bool)):
        if value:
            previous = index
            continue
        values[index] = float(window_hours if previous is None else index - previous)
    return values


def _station_static(frame: pd.DataFrame) -> np.ndarray:
    values = frame.loc[:, STATIC_FEATURES].dropna(how="all")
    if values.empty:
        return np.full(len(STATIC_FEATURES), np.nan, dtype=np.float32)
    return values.iloc[0].astype(float).to_numpy(dtype=np.float32)


def _multi_target(values: pd.Series, names: tuple[str, ...]) -> np.ndarray:
    target = np.zeros((len(values), len(names)), dtype=np.float32)
    lookup = {name: index for index, name in enumerate(names)}
    for row_index, value in enumerate(values):
        for token in _tokens(value):
            target[row_index, lookup[token]] = 1.0
    return target


def build_hourly_examples(
    hourly_frame: pd.DataFrame,
    labels: pd.DataFrame,
    window_hours: int,
    mask_mode: str = MASK_MODE_PER_HOUR,
) -> dict[str, np.ndarray]:
    if window_hours < 1:
        raise ValueError("window_hours must be positive")
    resolved_mask_mode = str(mask_mode)
    if resolved_mask_mode not in MASK_MODES:
        raise ValueError(f"unknown hourly mask mode: {mask_mode}")
    hourly = prepare_hourly_frame(hourly_frame)
    required_labels = [
        "station_id",
        "hour",
        "fault_hour",
        "display_state",
        "mechanisms",
        "components",
        "detectors_fired",
        "source_episode_ids",
        "training_eligible",
    ]
    missing = sorted(set(required_labels).difference(labels.columns))
    if missing:
        raise KeyError(f"hourly labels lack fields needed for tensors: {missing}")
    label_frame = labels.copy()
    label_frame["station_id"] = label_frame["station_id"].astype(str)
    label_frame["hour"] = pd.to_datetime(label_frame["hour"], utc=True, format="mixed")
    label_frame = label_frame.sort_values(["station_id", "hour"]).reset_index(drop=True)
    hourly_keys = hourly.loc[:, ["station_id", "hour"]].reset_index(drop=True)
    label_keys = label_frame.loc[:, ["station_id", "hour"]]
    if not hourly_keys.equals(label_keys):
        raise ValueError("hourly features and labels do not share identical station-hour keys")

    targets = label_frame.loc[label_frame["training_eligible"].astype(bool)].copy()
    targets = targets.sort_values(["station_id", "hour"]).reset_index(drop=True)
    total = len(targets)
    feature_count = len(CONTINUOUS_FEATURES)
    rule_count = len(RULE_EVIDENCE_FLAGS)
    x_cont = np.full((total, window_hours, feature_count), np.nan, dtype=np.float32)
    mask = np.zeros((total, window_hours, 1), dtype=np.float32)
    time_since_last = np.zeros((total, window_hours, 1), dtype=np.float32)
    static = np.full((total, len(STATIC_FEATURES)), np.nan, dtype=np.float32)
    rule_evidence = np.zeros((total, rule_count * 2), dtype=np.float32)
    rows_present = np.zeros(total, dtype=np.int64)

    target_by_station = {
        station_id: group
        for station_id, group in targets.groupby("station_id", sort=False)
    }
    cursor = 0
    for station_id, station_targets in target_by_station.items():
        station_hourly = hourly.loc[hourly["station_id"].eq(station_id)].copy()
        observed = station_hourly.loc[station_hourly["data_present"].eq(1)].copy()
        if observed.empty:
            raise ValueError(f"training target belongs to a station without observed rows: {station_id}")
        observed = observed.sort_values("hour")
        grid = pd.date_range(
            start=observed["hour"].iloc[0] - pd.Timedelta(hours=window_hours - 1),
            end=observed["hour"].iloc[-1],
            freq="h",
            tz="UTC",
        )
        indexed = observed.set_index("hour").reindex(grid)
        present = indexed["data_present"].fillna(0).eq(1).to_numpy(dtype=bool)
        positions = grid.get_indexer(pd.DatetimeIndex(station_targets["hour"]))
        if (positions < 0).any():
            raise ValueError(f"training target was not found in its station grid: {station_id}")
        offsets = np.arange(window_hours, dtype=np.int64)
        window_indices = positions[:, None] - (window_hours - 1) + offsets
        count = len(station_targets)
        next_cursor = cursor + count
        continuous_values = indexed.loc[:, CONTINUOUS_FEATURES].astype(float).to_numpy(dtype=np.float32)
        x_cont[cursor:next_cursor] = continuous_values[window_indices]
        window_present = present[window_indices]
        mask[cursor:next_cursor, :, 0] = window_present.astype(np.float32)
        elapsed = _time_since_last(present, window_hours)
        time_since_last[cursor:next_cursor, :, 0] = elapsed[window_indices]
        rows_present[cursor:next_cursor] = window_present.sum(axis=1, dtype=np.int64)
        static[cursor:next_cursor] = _station_static(observed)
        rule_values = indexed.loc[:, RULE_EVIDENCE_FLAGS].apply(
            pd.to_numeric,
            errors="coerce",
        ).fillna(0.0).to_numpy(dtype=np.float32)
        window_rules = rule_values[window_indices]
        rule_evidence[cursor:next_cursor, 0::2] = window_rules.max(axis=1)
        denominator = window_present.sum(axis=1, dtype=np.float32)
        numerator = (window_rules * window_present[:, :, None]).sum(axis=1)
        rule_evidence[cursor:next_cursor, 1::2] = np.divide(
            numerator,
            denominator[:, None],
            out=np.zeros_like(numerator, dtype=np.float32),
            where=denominator[:, None] > 0.0,
        )
        cursor = next_cursor
    if cursor != total:
        raise RuntimeError("hourly tensor assembly did not cover every training target")

    y_binary = targets["fault_hour"].astype(np.int64).to_numpy()
    y_mechanism = _multi_target(targets["mechanisms"], MECHANISM_LABEL_NAMES)
    y_component = _multi_target(targets["components"], COMPONENT_LABEL_NAMES)
    result = {
        "X_cont": x_cont,
        "mask": mask,
        "time_since_last": time_since_last,
        "static": static,
        "rule_evidence": rule_evidence,
        "y_binary": y_binary,
        "y_mechanism": y_mechanism,
        "y_component": y_component,
        "mechanism_target_available": (y_binary == 1) & y_mechanism.astype(bool).any(axis=1),
        "component_target_available": (y_binary == 1) & y_component.astype(bool).any(axis=1),
        "mechanism_label_names": np.asarray(MECHANISM_LABEL_NAMES, dtype=object),
        "component_label_names": np.asarray(COMPONENT_LABEL_NAMES, dtype=object),
        "station_id": targets["station_id"].astype(str).to_numpy(dtype=object),
        "hour": targets["hour"].astype(str).to_numpy(dtype=object),
        "display_state": targets["display_state"].astype(str).to_numpy(dtype=object),
        "mechanisms": targets["mechanisms"].fillna("").astype(str).to_numpy(dtype=object),
        "components": targets["components"].fillna("").astype(str).to_numpy(dtype=object),
        "detectors_fired": targets["detectors_fired"].fillna("").astype(str).to_numpy(dtype=object),
        "source_episode_ids": targets["source_episode_ids"].fillna("").astype(str).to_numpy(dtype=object),
        "rows_present": rows_present,
        "rows_with_any_feature": rows_present.copy(),
        "window_hours": np.asarray([window_hours], dtype=np.int64),
        "continuous_feature_names": np.asarray(CONTINUOUS_FEATURES, dtype=object),
        "static_feature_names": np.asarray(STATIC_FEATURES, dtype=object),
        "rule_evidence_feature_names": np.asarray(rule_evidence_feature_names(), dtype=object),
        "mask_feature_names": np.asarray(["row_present"], dtype=object),
    }
    if resolved_mask_mode == MASK_MODE_PER_FEATURE:
        return add_per_feature_mask(result)
    return result


def add_per_feature_mask(examples: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    result = dict(examples)
    x_cont = result["X_cont"]
    result["mask_per_hour"] = result["mask"]
    result["mask_per_hour_feature_names"] = result.get(
        "mask_feature_names", np.asarray(["row_present"], dtype=object)
    )
    result["mask"] = (~np.isnan(x_cont)).astype(np.float32)
    result["mask_feature_names"] = result.get(
        "continuous_feature_names", np.asarray(CONTINUOUS_FEATURES, dtype=object)
    )
    return result


def write_hourly_tensor(examples: dict[str, np.ndarray], path: Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, **examples)
    return destination


def _table(frame: pd.DataFrame) -> str:
    return frame.to_string(index=False)


def _fault_label_counts(labels: pd.DataFrame, column: str, names: tuple[str, ...]) -> pd.DataFrame:
    counts = {name: 0 for name in names}
    for value in labels.loc[labels["display_state"].eq("fault"), column]:
        for token in _tokens(value):
            counts[token] += 1
    return pd.DataFrame(
        {
            column: list(names),
            "fault_hours": [counts[name] for name in names],
        }
    )


def hourly_report(labels: pd.DataFrame, labelled_episodes: pd.DataFrame) -> str:
    states = ("fault", "benign", "clean", "excluded")
    total = len(labels)
    state_counts = labels["display_state"].value_counts()
    state_table = pd.DataFrame(
        {
            "display_state": states,
            "station_hours": [int(state_counts.get(state, 0)) for state in states],
        }
    )
    state_table["percent"] = state_table["station_hours"].mul(100.0).div(total).round(3)

    trainable = labels.loc[labels["training_eligible"].astype(bool)]
    balance_rows = []
    for period in ("overall", "pre_april", "april_may", "june"):
        subset = trainable if period == "overall" else trainable.loc[trainable["period"].eq(period)]
        faults = int(subset["fault_hour"].eq(1).sum())
        not_faults = int(subset["fault_hour"].eq(0).sum())
        ratio = "n/a" if faults == 0 else f"{not_faults / faults:.2f}:1"
        balance_rows.append(
            {
                "period": period,
                "not_fault_hours": not_faults,
                "fault_hours": faults,
                "not_fault_to_fault": ratio,
            }
        )
    balance_table = pd.DataFrame(balance_rows)

    episodes = prepare_labelled_episodes(labelled_episodes)
    faults = episodes.loc[episodes["label_state"].eq("fault")].copy()
    durations = ((faults["end_hour"] - faults["start_hour"]).dt.total_seconds() / 3600.0 + 1.0).astype(int)
    duration_rows = []
    for name, condition in (
        ("1h", durations.eq(1)),
        ("2-3h", durations.between(2, 3)),
        ("4-6h", durations.between(4, 6)),
        ("7-12h", durations.between(7, 12)),
        ("13-23h", durations.between(13, 23)),
        ("24h+", durations.ge(24)),
    ):
        duration_rows.append({"duration": name, "fault_episodes": int(condition.sum())})
    duration_table = pd.DataFrame(duration_rows)

    labeled_fault_hours = int(labels["display_state"].eq("fault").sum())
    duration_summary = pd.DataFrame(
        [
            {
                "fault_episodes": int(len(faults)),
                "median_hours": float(durations.median()) if len(durations) else np.nan,
                "mean_hours": round(float(durations.mean()), 3) if len(durations) else np.nan,
                "training_ready_fault_hours": labeled_fault_hours,
                "hours_per_fault_episode": round(labeled_fault_hours / len(faults), 3) if len(faults) else np.nan,
            }
        ]
    )

    station_table = (
        labels.loc[labels["display_state"].eq("fault")]
        .groupby("station_id", as_index=False)
        .size()
        .rename(columns={"size": "fault_hours"})
    )
    stations = pd.DataFrame({"station_id": sorted(labels["station_id"].astype(str).unique())})
    station_table = stations.merge(station_table, on="station_id", how="left")
    station_table["fault_hours"] = station_table["fault_hours"].fillna(0).astype(int)
    station_table["near_zero_5h_or_less"] = station_table["fault_hours"].le(5)
    station_table = station_table.sort_values(["fault_hours", "station_id"]).reset_index(drop=True)

    benign_clean = pd.DataFrame(
        {
            "display_state": ["benign", "clean"],
            "not_fault_hours": [
                int(labels["display_state"].eq("benign").sum()),
                int(labels["display_state"].eq("clean").sum()),
            ],
        }
    )
    excluded_with_label = int(
        labels.loc[
            labels["display_state"].eq("excluded"),
            "fault_hour",
        ].notna().sum()
    )
    unavailable = int((~labels["data_present"].astype(bool)).sum())
    no_detector = int(
        labels["data_present"].astype(bool).mul(~labels["detector_available"].astype(bool)).sum()
    )
    parts = [
        "HOURLY DETECTION DATASET REPORT",
        "",
        "1. DISPLAY STATE DISTRIBUTION",
        f"total_station_hours={total}",
        _table(state_table),
        "",
        "2. TWO-CLASS TRAINING BALANCE",
        _table(balance_table),
        "",
        "3. FAULT EPISODE DURATION",
        _table(duration_table),
        _table(duration_summary),
        "",
        "4. FAULT HOURS AND FAULT EPISODES",
        _table(duration_summary.loc[:, ["training_ready_fault_hours", "fault_episodes", "hours_per_fault_episode"]]),
        "",
        "5. FAULT HOURS BY MECHANISM",
        _table(_fault_label_counts(labels, "mechanisms", MECHANISM_LABEL_NAMES)),
        "",
        "FAULT HOURS BY COMPONENT",
        _table(_fault_label_counts(labels, "components", COMPONENT_LABEL_NAMES)),
        "",
        "6. FAULT HOURS BY STATION",
        _table(station_table),
        "",
        "7. NOT-FAULT DISPLAY STATES",
        _table(benign_clean),
        "",
        "8. EXCLUDED TRAINING LABEL CHECK",
        f"excluded_hours_with_training_label={excluded_with_label}",
        f"source_hours_without_measurements={unavailable}",
        f"measured_hours_without_detector_availability={no_detector}",
        "",
        "WINDOW ASSEMBLY CHECK",
        f"short_window_hours={SHORT_WINDOW_HOURS}",
        f"long_window_hours={LONG_WINDOW_HOURS}",
        "later_hour_rows_in_each_window=0",
        "window_end_alignment=current_hour",
        "source_feature_snapshot=existing_feature_matrix",
    ]
    return "\n".join(parts)
