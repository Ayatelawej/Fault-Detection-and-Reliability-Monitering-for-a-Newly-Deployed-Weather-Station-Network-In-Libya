from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.rules.config import (
    EXTERNAL_ARRAY_CHANNELS,
    EXTERNAL_FEATURES_PATH,
    EXTERNAL_OFFSET_MIN_FLEET_STATIONS,
    EXTERNAL_RATIO_MIN_HOURS_PER_DAY,
    EXTERNAL_RESIDUALS_PATH,
    EXTERNAL_SYSTEMIC_DISAGREE_Z,
    EXTERNAL_SYSTEMIC_EVIDENCE_PATH,
    EXTERNAL_TEMP_OFFSET_MIN_HOURS,
    EXTERNAL_TEMP_OFFSET_WINDOW_HOURS,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESIDUALS_PATH = PROJECT_ROOT / EXTERNAL_RESIDUALS_PATH
FEATURES_PATH = PROJECT_ROOT / EXTERNAL_FEATURES_PATH
SYSTEMIC_EVIDENCE_PATH = PROJECT_ROOT / EXTERNAL_SYSTEMIC_EVIDENCE_PATH
FEATURE_COLUMNS = [
    "station_id",
    "time_utc",
    "z_pressure",
    "z_temp",
    "z_dewpoint",
    "z_wind",
    "z_solar",
    "r_pressure",
    "offset_level_pressure",
    "offset_level_dewpoint",
    "offset_level_temp",
    "clear_sky_ratio",
    "rel_ratio_solar",
    "ext_abs_z_array_mean",
    "ext_n_valid_array",
]
LABEL_LEAK_TERMS = ["label", "verdict", "tier", "role", "episode", "cluster"]


def load_residuals(path: Path = RESIDUALS_PATH) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    frame["time_utc"] = pd.to_datetime(frame["time_utc"], utc=True)
    return frame.sort_values(["station_id", "time_utc"]).reset_index(drop=True)


def temp_offset_level(
    station: pd.DataFrame,
    window_hours: int = EXTERNAL_TEMP_OFFSET_WINDOW_HOURS,
    min_hours: int = EXTERNAL_TEMP_OFFSET_MIN_HOURS,
) -> pd.Series:
    values = pd.to_numeric(station["r_temp"], errors="coerce")
    return values.rolling(window_hours, min_periods=min_hours).median()


def solar_daily_relative_ratio(
    residuals: pd.DataFrame,
    min_hours_per_day: int = EXTERNAL_RATIO_MIN_HOURS_PER_DAY,
    min_fleet_stations: int = EXTERNAL_OFFSET_MIN_FLEET_STATIONS,
) -> pd.DataFrame:
    frame = residuals.loc[:, ["station_id", "time_utc", "clear_sky_ratio"]].copy()
    frame["time_utc"] = pd.to_datetime(frame["time_utc"], utc=True)
    frame["date"] = frame["time_utc"].dt.floor("D")
    daily = (
        frame.groupby(["station_id", "date"])["clear_sky_ratio"]
        .agg(ratio_daily_median="median", qualifying_hours="count")
        .reset_index()
    )
    daily.loc[
        daily["qualifying_hours"].lt(min_hours_per_day),
        "ratio_daily_median",
    ] = np.nan
    fleet = daily.groupby("date")["ratio_daily_median"].agg(
        fleet_daily_ratio="median",
        fleet_valid_stations="count",
    )
    fleet.loc[
        fleet["fleet_valid_stations"].lt(min_fleet_stations),
        "fleet_daily_ratio",
    ] = np.nan
    result = daily.merge(fleet.reset_index(), on="date", how="left")
    result["rel_ratio_solar"] = (
        result["ratio_daily_median"] / result["fleet_daily_ratio"]
    )
    return result.loc[:, ["station_id", "date", "rel_ratio_solar"]]


def build_external_features(residuals: pd.DataFrame) -> pd.DataFrame:
    frame = residuals.copy()
    frame["time_utc"] = pd.to_datetime(frame["time_utc"], utc=True)
    frame = frame.sort_values(["station_id", "time_utc"]).reset_index(drop=True)
    features = frame.loc[
        :,
        [
            "station_id",
            "time_utc",
            "z_pressure",
            "z_temp",
            "z_dewpoint",
            "z_wind",
            "z_solar",
            "r_pressure",
            "base_pressure",
            "base_dewpoint",
            "clear_sky_ratio",
        ],
    ].copy()
    features = features.rename(
        columns={
            "base_pressure": "offset_level_pressure",
            "base_dewpoint": "offset_level_dewpoint",
        }
    )
    features["offset_level_temp"] = frame.groupby("station_id")["r_temp"].transform(
        lambda values: pd.to_numeric(values, errors="coerce").rolling(
            EXTERNAL_TEMP_OFFSET_WINDOW_HOURS,
            min_periods=EXTERNAL_TEMP_OFFSET_MIN_HOURS,
        ).median()
    )
    daily_ratio = solar_daily_relative_ratio(frame)
    features["date"] = features["time_utc"].dt.floor("D")
    features = features.merge(daily_ratio, on=["station_id", "date"], how="left")
    z_columns = [f"z_{channel}" for channel in EXTERNAL_ARRAY_CHANNELS]
    array_z = features.loc[:, z_columns].abs()
    features["ext_abs_z_array_mean"] = array_z.mean(axis=1, skipna=True)
    features["ext_n_valid_array"] = array_z.notna().sum(axis=1)
    features = features.drop(columns=["date"])
    return features.loc[:, FEATURE_COLUMNS]


def feature_has_label_leak(frame: pd.DataFrame) -> bool:
    columns = [str(column).lower() for column in frame.columns]
    return any(term in column for column in columns for term in LABEL_LEAK_TERMS)


def select_systemic_medium_episodes(
    labeled_episodes: pd.DataFrame,
    expected_count_range: tuple[int, int] | None = (100, 160),
) -> tuple[pd.DataFrame, dict[str, object]]:
    required = {"label", "tier", "station_id", "start_hour", "end_hour"}
    missing = required.difference(labeled_episodes.columns)
    if missing:
        raise KeyError(sorted(missing))
    labels_used = ["systemic_array"]
    tiers_used = ["B"]
    selected = labeled_episodes.loc[
        labeled_episodes["label"].isin(labels_used)
        & labeled_episodes["tier"].isin(tiers_used)
    ].copy()
    count = len(selected)
    if expected_count_range is not None:
        low, high = expected_count_range
        if count < low or count > high:
            raise ValueError(
                f"systemic episode count {count} outside sane range {low}-{high}"
            )
    info = {
        "label_column": "label",
        "tier_column": "tier",
        "labels_used": labels_used,
        "tiers_used": tiers_used,
        "count": count,
    }
    return selected, info


def _episode_channel_medians(
    station: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> dict[str, float]:
    window = station.loc[station["time_utc"].between(start, end, inclusive="both")]
    values: dict[str, float] = {}
    for channel in EXTERNAL_ARRAY_CHANNELS:
        column = f"z_{channel}"
        clean = pd.to_numeric(window[column], errors="coerce").abs().dropna()
        med = clean.median() if not clean.empty else np.nan
        values[f"median_abs_z_{channel}"] = float(med) if pd.notna(med) else np.nan
    return values


def _support_tier(channel_medians: dict[str, float]) -> str:
    values = pd.Series(channel_medians, dtype=float).dropna()
    if values.empty:
        return "inconclusive"
    n_disagree = int(values.ge(EXTERNAL_SYSTEMIC_DISAGREE_Z).sum())
    if n_disagree >= 2:
        return "supported"
    if n_disagree == 1:
        return "single_channel"
    return "benign"


def systemic_external_evidence(
    residuals: pd.DataFrame,
    labeled_episodes: pd.DataFrame,
    expected_count_range: tuple[int, int] | None = (100, 160),
) -> pd.DataFrame:
    frame = residuals.copy()
    frame["time_utc"] = pd.to_datetime(frame["time_utc"], utc=True)
    by_station = {
        station_id: station.sort_values("time_utc")
        for station_id, station in frame.groupby("station_id", sort=True)
    }
    episodes, _ = select_systemic_medium_episodes(
        labeled_episodes,
        expected_count_range,
    )
    rows = []
    for _, episode in episodes.iterrows():
        station_id = str(episode["station_id"])
        start = pd.to_datetime(episode["start_hour"], utc=True)
        end = pd.to_datetime(episode["end_hour"], utc=True)
        station = by_station.get(station_id)
        if station is None:
            medians = {f"median_abs_z_{channel}": np.nan for channel in EXTERNAL_ARRAY_CHANNELS}
        else:
            medians = _episode_channel_medians(station, start, end)
        values = pd.Series(medians, dtype=float).dropna()
        n_disagree = int(values.ge(EXTERNAL_SYSTEMIC_DISAGREE_Z).sum())
        rows.append(
            {
                "station_id": station_id,
                "start": start,
                "end": end,
                "duration_hours": episode.get("duration_hours", np.nan),
                "sensor_groups": episode.get("affected_sensor_groups", ""),
                "n_channels_disagree": n_disagree,
                **medians,
                "mean_abs_z": float(values.mean()) if not values.empty else np.nan,
                "max_abs_z": float(values.max()) if not values.empty else np.nan,
                "support_tier": _support_tier(medians),
            }
        )
    columns = [
        "station_id",
        "start",
        "end",
        "duration_hours",
        "sensor_groups",
        "n_channels_disagree",
    ]
    columns.extend(f"median_abs_z_{channel}" for channel in EXTERNAL_ARRAY_CHANNELS)
    columns.extend(["mean_abs_z", "max_abs_z", "support_tier"])
    return pd.DataFrame(rows, columns=columns)


def write_external_features(
    residuals: pd.DataFrame,
    path: Path = FEATURES_PATH,
) -> pd.DataFrame:
    features = build_external_features(residuals)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(path, index=False)
    return features


def write_systemic_evidence(
    residuals: pd.DataFrame,
    labeled_episodes: pd.DataFrame,
    path: Path = SYSTEMIC_EVIDENCE_PATH,
) -> pd.DataFrame:
    evidence = systemic_external_evidence(residuals, labeled_episodes)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    evidence.to_csv(path, index=False)
    return evidence
