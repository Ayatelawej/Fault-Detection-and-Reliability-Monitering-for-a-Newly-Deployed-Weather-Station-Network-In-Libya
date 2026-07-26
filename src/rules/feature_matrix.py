from __future__ import annotations

from functools import reduce
from pathlib import Path

import pandas as pd

from src.config.paths import MERGED_DATASET_PATH, STATION_REGISTRY_PATH
from src.rules.channel_handlers import sensor_group_for_channel
from src.rules.config import (
    EXTERNAL_FEATURES_PATH,
    FEATURE_MATRIX_PATH,
    SENSOR_GROUP_PREFIXES,
    SPATIAL_FEATURES_PATH,
    SPATIAL_NEIGHBORS_PATH,
    STATISTICAL_FEATURES_PATH,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
KEY_COLUMNS = ["station_id", "time_utc"]
STATISTICAL_SCORE_PATH = PROJECT_ROOT / "data/processed/statistical_anomaly_scores.parquet"
STATISTICAL_FEATURES_FILE = PROJECT_ROOT / STATISTICAL_FEATURES_PATH
EXTERNAL_FEATURES_FILE = PROJECT_ROOT / EXTERNAL_FEATURES_PATH
SPATIAL_FEATURES_FILE = PROJECT_ROOT / SPATIAL_FEATURES_PATH
FEATURE_MATRIX_FILE = PROJECT_ROOT / FEATURE_MATRIX_PATH
SPATIAL_NEIGHBORS_FILE = PROJECT_ROOT / SPATIAL_NEIGHBORS_PATH
STATISTICAL_METRICS = [
    "zscore",
    "rolling_variance",
    "iforest_score",
    "flag_zscore",
    "flag_stuck",
    "flag_iforest",
    "flag_physical",
    "flag_physical_suspect",
    "flag",
]
FORBIDDEN_COLUMN_TERMS = ["label", "family", "tier", "verdict", "episode", "queue"]


def _normalize_keys(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    if "time_utc" not in result.columns and "hour_utc" in result.columns:
        result = result.rename(columns={"hour_utc": "time_utc"})
    missing = [column for column in KEY_COLUMNS if column not in result.columns]
    if missing:
        raise KeyError(missing)
    result["station_id"] = result["station_id"].astype(str)
    result["time_utc"] = pd.to_datetime(result["time_utc"], utc=True)
    return result


def _validate_unique(frame: pd.DataFrame, keys: list[str], name: str) -> None:
    if frame.duplicated(keys).any():
        raise ValueError(f"{name} has duplicate keys")


def _load_frame(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    return _normalize_keys(frame)


def load_external_features(path: Path = EXTERNAL_FEATURES_FILE) -> pd.DataFrame:
    return _load_frame(path)


def load_spatial_features(path: Path = SPATIAL_FEATURES_FILE) -> pd.DataFrame:
    return _load_frame(path)


def load_statistical_features(path: Path = STATISTICAL_FEATURES_FILE) -> pd.DataFrame:
    return _load_frame(path)


def load_registry(path: Path = STATION_REGISTRY_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def load_neighbors(path: Path = SPATIAL_NEIGHBORS_FILE) -> pd.DataFrame:
    return pd.read_csv(path)


def _score_metric_wide(scores: pd.DataFrame, metric: str) -> pd.DataFrame:
    values = scores.loc[:, KEY_COLUMNS + ["channel", metric]].copy()
    if metric.startswith("flag"):
        values[metric] = values[metric].astype(float)
    pivot = values.pivot(index=KEY_COLUMNS, columns="channel", values=metric)
    pivot.columns = [f"{metric}_{column}" for column in pivot.columns]
    return pivot.reset_index()


def _sensor_group_flags(scores: pd.DataFrame) -> pd.DataFrame:
    groups = sorted(set(SENSOR_GROUP_PREFIXES.values()) | {"other"})
    frame = scores.loc[:, KEY_COLUMNS + ["channel", "flag"]].copy()
    frame["sensor_group"] = frame["channel"].map(sensor_group_for_channel)
    frame["flag"] = frame["flag"].astype(float)
    grouped = (
        frame.groupby(KEY_COLUMNS + ["sensor_group"], dropna=False)["flag"]
        .max()
        .unstack("sensor_group")
    )
    grouped = grouped.reindex(columns=groups)
    grouped.columns = [f"sensor_group_flag_{column}" for column in grouped.columns]
    return grouped.reset_index()


def materialize_statistical_features(
    scores: pd.DataFrame | None = None,
    merged: pd.DataFrame | None = None,
    output_path: Path | None = STATISTICAL_FEATURES_FILE,
) -> pd.DataFrame:
    if scores is None:
        scores = pd.read_parquet(STATISTICAL_SCORE_PATH)
    if merged is None:
        merged = pd.read_csv(MERGED_DATASET_PATH, usecols=["station_id", "hour_utc"])
    scores = _normalize_keys(scores)
    merged = _normalize_keys(merged)
    _validate_unique(merged, KEY_COLUMNS, "merged hourly frame")
    _validate_unique(scores, KEY_COLUMNS + ["channel"], "statistical score frame")
    frames = [merged.loc[:, KEY_COLUMNS].copy()]
    for metric in STATISTICAL_METRICS:
        if metric in scores.columns:
            frames.append(_score_metric_wide(scores, metric))
    frames.append(_sensor_group_flags(scores))
    features = reduce(
        lambda left, right: left.merge(right, on=KEY_COLUMNS, how="left", validate="one_to_one"),
        frames,
    )
    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        features.to_parquet(output_path, index=False)
    return features


def prefix_statistical_columns(frame: pd.DataFrame) -> pd.DataFrame:
    result = _normalize_keys(frame)
    rename = {
        column: f"stat_{column}"
        for column in result.columns
        if column not in KEY_COLUMNS
    }
    return result.rename(columns=rename)


def _merge_one(left: pd.DataFrame, right: pd.DataFrame, keys: list[str], name: str) -> pd.DataFrame:
    _validate_unique(right, keys, name)
    collisions = sorted(set(left.columns).intersection(right.columns).difference(keys))
    if collisions:
        raise ValueError(f"{name} column collision: {collisions}")
    return left.merge(right, on=keys, how="left", validate="one_to_one")


def station_context(
    registry: pd.DataFrame,
    neighbors: pd.DataFrame,
    station_ids: pd.Series,
) -> pd.DataFrame:
    stations = pd.DataFrame({"station_id": sorted(station_ids.astype(str).unique())})
    elevation = registry.loc[:, ["station_id", "elevation"]].copy()
    elevation["station_id"] = elevation["station_id"].astype(str)
    elevation = elevation.rename(columns={"elevation": "ctx_elevation"})
    counts = (
        neighbors.groupby("station_id")
        .size()
        .rename("ctx_n_neighbors")
        .reset_index()
    )
    counts["station_id"] = counts["station_id"].astype(str)
    context = stations.merge(elevation, on="station_id", how="left", validate="one_to_one")
    context = context.merge(counts, on="station_id", how="left", validate="one_to_one")
    context["ctx_n_neighbors"] = context["ctx_n_neighbors"].fillna(0).astype(int)
    return context


def feature_has_forbidden_columns(frame: pd.DataFrame) -> bool:
    columns = [str(column).lower() for column in frame.columns]
    return any(term in column for column in columns for term in FORBIDDEN_COLUMN_TERMS)


def build_feature_matrix(
    external_features: pd.DataFrame | None = None,
    statistical_features: pd.DataFrame | None = None,
    spatial_features: pd.DataFrame | None = None,
    registry: pd.DataFrame | None = None,
    neighbors: pd.DataFrame | None = None,
    return_metadata: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, dict[str, object]]:
    external = load_external_features() if external_features is None else _normalize_keys(external_features)
    statistical = (
        load_statistical_features()
        if statistical_features is None
        else _normalize_keys(statistical_features)
    )
    spatial = load_spatial_features() if spatial_features is None else _normalize_keys(spatial_features)
    registry_frame = load_registry() if registry is None else registry.copy()
    neighbors_frame = load_neighbors() if neighbors is None else neighbors.copy()
    _validate_unique(external, KEY_COLUMNS, "external features")
    _validate_unique(statistical, KEY_COLUMNS, "statistical features")
    _validate_unique(spatial, KEY_COLUMNS, "spatial features")
    frame = external.copy()
    source_columns = {
        "external": [column for column in external.columns if column not in KEY_COLUMNS],
    }
    row_counts = [{"step": "external_spine", "rows": len(frame)}]
    stat_prefixed = prefix_statistical_columns(statistical)
    frame = _merge_one(frame, stat_prefixed, KEY_COLUMNS, "statistical features")
    source_columns["statistical"] = [
        column for column in stat_prefixed.columns if column not in KEY_COLUMNS
    ]
    row_counts.append({"step": "after_statistical_join", "rows": len(frame)})
    frame = _merge_one(frame, spatial, KEY_COLUMNS, "spatial features")
    source_columns["spatial"] = [column for column in spatial.columns if column not in KEY_COLUMNS]
    row_counts.append({"step": "after_spatial_join", "rows": len(frame)})
    context = station_context(registry_frame, neighbors_frame, frame["station_id"])
    frame = frame.merge(context, on="station_id", how="left", validate="many_to_one")
    source_columns["context"] = [column for column in context.columns if column != "station_id"]
    row_counts.append({"step": "after_context_join", "rows": len(frame)})
    if feature_has_forbidden_columns(frame):
        raise ValueError("feature matrix contains forbidden columns")
    metadata = {
        "row_counts": row_counts,
        "source_columns": source_columns,
    }
    if return_metadata:
        return frame, metadata
    return frame


def write_feature_matrix(
    frame: pd.DataFrame,
    path: Path = FEATURE_MATRIX_FILE,
) -> pd.DataFrame:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    return frame
