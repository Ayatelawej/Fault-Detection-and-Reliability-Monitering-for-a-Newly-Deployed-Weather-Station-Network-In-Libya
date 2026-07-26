from __future__ import annotations

from itertools import permutations
from pathlib import Path

import numpy as np
import pandas as pd

from src.config.paths import MERGED_DATASET_PATH, STATION_REGISTRY_PATH
from src.rules.config import (
    EXTERNAL_END_DATE,
    EXTERNAL_START_DATE,
    SPATIAL_BASELINE_MIN_HOURS,
    SPATIAL_BASELINE_WINDOW_HOURS,
    SPATIAL_CHANNEL_COLUMNS,
    SPATIAL_CHANNELS,
    SPATIAL_EXCLUDED_NEIGHBORS,
    SPATIAL_MAD_FLOORS,
    SPATIAL_MIN_NEIGHBORS,
    SPATIAL_MIN_NEIGHBORS_PRESENT,
    SPATIAL_NEIGHBOR_RADIUS_KM,
    SPATIAL_NEIGHBORS_PATH,
    SPATIAL_RESIDUALS_PATH,
)
from src.rules.external_residuals import rolling_median_mad

PROJECT_ROOT = Path(__file__).resolve().parents[2]
NEIGHBORS_PATH = PROJECT_ROOT / SPATIAL_NEIGHBORS_PATH
RESIDUALS_PATH = PROJECT_ROOT / SPATIAL_RESIDUALS_PATH
EARTH_RADIUS_KM = 6371.0


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    lat1_r = np.radians(lat1)
    lat2_r = np.radians(lat2)
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat1_r) * np.cos(lat2_r) * np.sin(dlon / 2.0) ** 2
    )
    return float(EARTH_RADIUS_KM * 2.0 * np.arcsin(np.sqrt(a)))


def load_registry(path: Path = STATION_REGISTRY_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def load_merged(path: Path = MERGED_DATASET_PATH) -> pd.DataFrame:
    frame = pd.read_csv(path, low_memory=False)
    frame["hour_utc"] = pd.to_datetime(frame["hour_utc"], utc=True)
    return frame


def full_time_index(
    start_date: str = EXTERNAL_START_DATE,
    end_date: str = EXTERNAL_END_DATE,
) -> pd.DatetimeIndex:
    start = pd.Timestamp(start_date, tz="UTC")
    end = pd.Timestamp(end_date, tz="UTC") + pd.Timedelta(hours=23)
    return pd.date_range(start, end, freq="h", tz="UTC")


def build_neighbor_graph(
    registry: pd.DataFrame,
    radius_km: float = SPATIAL_NEIGHBOR_RADIUS_KM,
) -> pd.DataFrame:
    required = {"station_id", "latitude", "longitude"}
    missing = required.difference(registry.columns)
    if missing:
        raise KeyError(sorted(missing))
    clean = registry.loc[:, ["station_id", "latitude", "longitude"]].dropna()
    rows = []
    for left, right in permutations(clean.to_dict("records"), 2):
        distance = haversine_km(
            float(left["latitude"]),
            float(left["longitude"]),
            float(right["latitude"]),
            float(right["longitude"]),
        )
        if distance <= radius_km:
            rows.append(
                {
                    "station_id": left["station_id"],
                    "neighbor_id": right["station_id"],
                    "distance_km": distance,
                }
            )
    return pd.DataFrame(rows, columns=["station_id", "neighbor_id", "distance_km"])


def isolated_stations(
    registry: pd.DataFrame,
    graph: pd.DataFrame,
    min_neighbors: int = SPATIAL_MIN_NEIGHBORS,
) -> list[str]:
    counts = graph.groupby("station_id")["neighbor_id"].nunique()
    station_ids = sorted(registry["station_id"].dropna().astype(str).unique())
    return [
        station_id
        for station_id in station_ids
        if int(counts.get(station_id, 0)) < min_neighbors
    ]


def _standardize_merged(merged: pd.DataFrame) -> pd.DataFrame:
    frame = merged.copy()
    if "time_utc" not in frame.columns:
        if "hour_utc" not in frame.columns:
            raise KeyError("hour_utc")
        frame = frame.rename(columns={"hour_utc": "time_utc"})
    frame["time_utc"] = pd.to_datetime(frame["time_utc"], utc=True)
    if frame.duplicated(["station_id", "time_utc"]).any():
        raise ValueError("duplicate station-hour keys")
    return frame


def _channel_column(channel: str) -> str:
    return SPATIAL_CHANNEL_COLUMNS.get(channel, channel)


def healthy_neighbors(neighbors: list[str]) -> list[str]:
    excluded = set(SPATIAL_EXCLUDED_NEIGHBORS)
    return [neighbor for neighbor in neighbors if neighbor not in excluded]


def _station_grid(
    merged: pd.DataFrame,
    station_ids: list[str],
    time_index: pd.DatetimeIndex,
) -> pd.DataFrame:
    frame = _standardize_merged(merged)
    columns = ["station_id", "time_utc"] + list(SPATIAL_CHANNEL_COLUMNS.values())
    frame = frame.loc[:, [column for column in columns if column in frame.columns]]
    grid = pd.MultiIndex.from_product(
        [station_ids, time_index],
        names=["station_id", "time_utc"],
    ).to_frame(index=False)
    return grid.merge(frame, on=["station_id", "time_utc"], how="left")


def neighbor_median(
    merged: pd.DataFrame,
    station_id: str,
    channel: str,
    graph: pd.DataFrame,
    time_index: pd.DatetimeIndex | None = None,
    min_neighbors_present: int = SPATIAL_MIN_NEIGHBORS_PRESENT,
    min_neighbors: int = SPATIAL_MIN_NEIGHBORS,
) -> pd.Series:
    frame = _standardize_merged(merged)
    if time_index is None:
        time_index = pd.DatetimeIndex(sorted(frame["time_utc"].dropna().unique()))
    station_ids = sorted(
        set(frame["station_id"].astype(str)).union({str(station_id)})
    )
    grid = _station_grid(frame, station_ids, time_index)
    column = _channel_column(channel)
    if column not in grid.columns:
        raise KeyError(column)
    neighbors = healthy_neighbors(
        graph.loc[graph["station_id"].eq(station_id), "neighbor_id"].astype(str).tolist()
    )
    if len(set(neighbors)) < min_neighbors:
        return pd.Series(np.nan, index=time_index, dtype=float)
    pivot = grid.pivot(index="time_utc", columns="station_id", values=column)
    available = [neighbor for neighbor in neighbors if neighbor in pivot.columns]
    if len(available) < min_neighbors:
        return pd.Series(np.nan, index=time_index, dtype=float)
    values = pivot.loc[:, available]
    medians = values.median(axis=1, skipna=True)
    medians = medians.where(values.notna().sum(axis=1).ge(min_neighbors_present))
    return medians.reindex(time_index)


def _add_channel_residuals(
    result: pd.DataFrame,
    grid: pd.DataFrame,
    graph: pd.DataFrame,
    station_ids: list[str],
    time_index: pd.DatetimeIndex,
    channel: str,
    baseline_window_hours: int,
    baseline_min_hours: int,
) -> pd.DataFrame:
    column = SPATIAL_CHANNEL_COLUMNS[channel]
    values = grid.pivot(index="time_utc", columns="station_id", values=column)
    residual = pd.DataFrame(index=time_index)
    for station_id in station_ids:
        med = neighbor_median(grid, station_id, channel, graph, time_index)
        station_values = values[station_id] if station_id in values.columns else np.nan
        residual[station_id] = pd.to_numeric(station_values, errors="coerce") - med
    long = (
        residual.reset_index()
        .melt(id_vars="index", var_name="station_id", value_name=f"r_spatial_{channel}")
        .rename(columns={"index": "time_utc"})
    )
    result = result.merge(long, on=["station_id", "time_utc"], how="left")
    base_values = []
    bmad_values = []
    for _, station in result.groupby("station_id", sort=False):
        base, bmad = rolling_median_mad(
            station[f"r_spatial_{channel}"],
            baseline_window_hours,
            baseline_min_hours,
        )
        base_values.append(base)
        bmad_values.append(bmad)
    result[f"base_spatial_{channel}"] = pd.concat(base_values).sort_index()
    result[f"bmad_spatial_{channel}"] = pd.concat(bmad_values).sort_index()
    spread = result[f"bmad_spatial_{channel}"].clip(lower=SPATIAL_MAD_FLOORS[channel])
    result[f"z_spatial_{channel}"] = (
        result[f"r_spatial_{channel}"] - result[f"base_spatial_{channel}"]
    ) / spread
    return result


def build_all(
    merged: pd.DataFrame | None = None,
    registry: pd.DataFrame | None = None,
    graph: pd.DataFrame | None = None,
    time_index: pd.DatetimeIndex | None = None,
    baseline_window_hours: int = SPATIAL_BASELINE_WINDOW_HOURS,
    baseline_min_hours: int = SPATIAL_BASELINE_MIN_HOURS,
) -> pd.DataFrame:
    merged_frame = load_merged() if merged is None else _standardize_merged(merged)
    registry_frame = load_registry() if registry is None else registry.copy()
    station_ids = sorted(registry_frame["station_id"].dropna().astype(str).unique())
    times = full_time_index() if time_index is None else pd.DatetimeIndex(time_index)
    neighbor_graph = build_neighbor_graph(registry_frame) if graph is None else graph.copy()
    grid = _station_grid(merged_frame, station_ids, times)
    result = pd.MultiIndex.from_product(
        [station_ids, times],
        names=["station_id", "time_utc"],
    ).to_frame(index=False)
    for channel in SPATIAL_CHANNELS:
        result = _add_channel_residuals(
            result,
            grid,
            neighbor_graph,
            station_ids,
            times,
            channel,
            baseline_window_hours,
            baseline_min_hours,
        )
    return result


def write_neighbor_graph(
    graph: pd.DataFrame,
    path: Path = NEIGHBORS_PATH,
) -> pd.DataFrame:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    graph.to_csv(path, index=False)
    return graph


def write_spatial_residuals(
    frame: pd.DataFrame,
    path: Path = RESIDUALS_PATH,
) -> pd.DataFrame:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    return frame
