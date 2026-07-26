from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.rules.config import (
    EXTERNAL_BASELINE_MIN_HOURS,
    EXTERNAL_BASELINE_WINDOW_HOURS,
    EXTERNAL_INSUFFICIENT_MIN_HOURS,
    EXTERNAL_OFFSET_CHANNELS,
    EXTERNAL_OFFSET_GAP_HOURS,
    EXTERNAL_OFFSET_MED_STABILITY_MAX,
    EXTERNAL_OFFSET_MIN_DAYS,
    EXTERNAL_OFFSET_MIN_DENSITY,
    EXTERNAL_OFFSET_MIN_FLEET_STATIONS,
    EXTERNAL_OFFSET_PHYSICAL_FLOORS,
    EXTERNAL_OFFSET_QUEUE_PATH,
    EXTERNAL_OFFSET_SCORE_HIGH,
    EXTERNAL_OFFSET_SCORE_MED,
    EXTERNAL_OFFSET_SPREAD_FLOORS,
    EXTERNAL_OFFSET_STABILITY_MAX,
    EXTERNAL_RATIO_MIN_DAYS,
    EXTERNAL_RATIO_MIN_HOURS_PER_DAY,
    EXTERNAL_RATIO_WINDOW_DAYS,
    EXTERNAL_REL_RATIO_MAX,
    EXTERNAL_REL_RATIO_MIN,
    EXTERNAL_RESIDUALS_PATH,
    EXTERNAL_SOLAR_QUEUE_ENABLED,
)
from src.rules.review_queue import OUTPUT_COLUMNS

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESIDUALS_PATH = PROJECT_ROOT / EXTERNAL_RESIDUALS_PATH
OFFSET_QUEUE_PATH = PROJECT_ROOT / EXTERNAL_OFFSET_QUEUE_PATH


def load_residuals(path: Path = RESIDUALS_PATH) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    frame["time_utc"] = pd.to_datetime(frame["time_utc"], utc=True)
    return frame.sort_values(["station_id", "time_utc"]).reset_index(drop=True)


def _mad(values: pd.Series) -> float:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if clean.empty:
        return np.nan
    median = float(clean.median())
    return float((clean - median).abs().median())


def _rolling_mad_raw(values: np.ndarray) -> float:
    clean = values[~np.isnan(values)]
    if len(clean) == 0:
        return np.nan
    median = float(np.median(clean))
    return float(np.median(np.abs(clean - median)))


def fleet_frame(
    frame: pd.DataFrame,
    channel: str,
    min_stations: int = EXTERNAL_OFFSET_MIN_FLEET_STATIONS,
) -> pd.DataFrame:
    base_column = f"base_{channel}"
    grouped = frame.groupby("time_utc")[base_column]
    result = grouped.agg(
        fleet_median="median",
        fleet_mad=_mad,
        fleet_valid_stations="count",
    ).reset_index()
    low_support = result["fleet_valid_stations"].lt(min_stations)
    result.loc[low_support, ["fleet_median", "fleet_mad"]] = np.nan
    result["channel"] = channel
    return result.loc[
        :,
        [
            "time_utc",
            "channel",
            "fleet_median",
            "fleet_mad",
            "fleet_valid_stations",
        ],
    ]


def offset_score(
    station: pd.DataFrame,
    fleet: pd.DataFrame,
    channel: str,
    spread_floor: float | None = None,
) -> pd.DataFrame:
    floor = EXTERNAL_OFFSET_SPREAD_FLOORS[channel] if spread_floor is None else spread_floor
    base_column = f"base_{channel}"
    bmad_column = f"bmad_{channel}"
    joined = station.merge(
        fleet.loc[:, ["time_utc", "fleet_median", "fleet_mad"]],
        on="time_utc",
        how="left",
    )
    spread = joined["fleet_mad"].clip(lower=floor)
    joined["offset_score"] = (joined[base_column] - joined["fleet_median"]) / spread
    joined.loc[
        joined[[base_column, "fleet_median", "fleet_mad"]].isna().any(axis=1),
        "offset_score",
    ] = np.nan
    joined["offset_abs_score"] = joined["offset_score"].abs()
    if channel == "temp":
        r_values = pd.to_numeric(joined["r_temp"], errors="coerce")
        joined["offset_level"] = r_values.rolling(
            window=EXTERNAL_BASELINE_WINDOW_HOURS,
            min_periods=EXTERNAL_BASELINE_MIN_HOURS,
        ).median()
        joined["offset_bmad"] = r_values.rolling(
            window=EXTERNAL_BASELINE_WINDOW_HOURS,
            min_periods=EXTERNAL_BASELINE_MIN_HOURS,
        ).apply(_rolling_mad_raw, raw=True)
    else:
        joined["offset_level"] = joined[base_column]
        joined["offset_bmad"] = joined[bmad_column]
    return joined


def add_hour_flags(
    scored: pd.DataFrame,
    channel: str,
    high_score: float = EXTERNAL_OFFSET_SCORE_HIGH,
    med_score: float = EXTERNAL_OFFSET_SCORE_MED,
) -> pd.DataFrame:
    result = scored.copy()
    abs_score = result["offset_abs_score"]
    physical_floor = EXTERNAL_OFFSET_PHYSICAL_FLOORS[channel]
    level_ok = result["offset_level"].abs().ge(physical_floor).fillna(False)
    high = (
        abs_score.ge(high_score)
        & result["offset_bmad"].le(EXTERNAL_OFFSET_STABILITY_MAX[channel])
        & level_ok
    )
    med = (
        abs_score.ge(med_score)
        & abs_score.lt(high_score)
        & result["offset_bmad"].le(EXTERNAL_OFFSET_MED_STABILITY_MAX[channel])
        & level_ok
    )
    result["offset_flag_high"] = high.fillna(False)
    result["offset_flag_med"] = med.fillna(False)
    result["offset_flag"] = result["offset_flag_high"] | result["offset_flag_med"]
    return result


def _merge_flagged_runs(
    frame: pd.DataFrame,
    time_column: str,
    flag_column: str,
    gap_hours: int,
    step_hours: int,
) -> list[pd.DataFrame]:
    flagged = frame.loc[frame[flag_column].fillna(False)].sort_values(time_column)
    if flagged.empty:
        return []

    groups: list[pd.DataFrame] = []
    start = 0
    times = pd.to_datetime(flagged[time_column], utc=True).reset_index(drop=True)
    max_delta = pd.Timedelta(hours=gap_hours + step_hours)

    for position in range(1, len(flagged)):
        if times.iloc[position] - times.iloc[position - 1] > max_delta:
            groups.append(flagged.iloc[start:position].copy())
            start = position

    groups.append(flagged.iloc[start:].copy())
    return groups


def _duration_hours(start: pd.Timestamp, end: pd.Timestamp, step_hours: int) -> float:
    return float((end - start) / pd.Timedelta(hours=1) + step_hours)


def _offset_episode_row(group: pd.DataFrame, station_id: str, channel: str) -> dict[str, object]:
    start = pd.to_datetime(group["time_utc"].min(), utc=True)
    end = pd.to_datetime(group["time_utc"].max(), utc=True)
    tier = "HIGH" if bool(group["offset_flag_high"].any()) else "MED"
    levels = pd.to_numeric(group["offset_level"], errors="coerce")
    scores = pd.to_numeric(group["offset_score"], errors="coerce")
    bmads = pd.to_numeric(group["offset_bmad"], errors="coerce")
    return {
        "station_id": station_id,
        "channel": channel,
        "start_hour": start,
        "end_hour": end,
        "duration_hours": _duration_hours(start, end, 1),
        "n_flagged_hours": int(len(group)),
        "tier": tier,
        "level_p10": float(levels.quantile(0.10)),
        "level_p50": float(levels.quantile(0.50)),
        "level_p90": float(levels.quantile(0.90)),
        "mean_score": float(scores.mean()),
        "peak_abs_score": float(scores.abs().max()),
        "median_bmad": float(bmads.median()),
        "episode_type": "offset",
    }


def offset_episodes_for_station(
    scored: pd.DataFrame,
    station_id: str,
    channel: str,
    gap_hours: int | None = None,
    min_days: int = EXTERNAL_OFFSET_MIN_DAYS,
    min_density: float = EXTERNAL_OFFSET_MIN_DENSITY,
) -> pd.DataFrame:
    effective_gap = EXTERNAL_OFFSET_GAP_HOURS[channel] if gap_hours is None else gap_hours
    rows = []
    for group in _merge_flagged_runs(scored, "time_utc", "offset_flag", effective_gap, 1):
        row = _offset_episode_row(group, station_id, channel)
        if row["duration_hours"] < min_days * 24:
            continue
        density = (
            row["n_flagged_hours"] / row["duration_hours"]
            if row["duration_hours"] > 0
            else 0.0
        )
        if density < min_density:
            continue
        rows.append(row)
    return pd.DataFrame(rows)


def offset_status(
    scored: pd.DataFrame,
    episodes: pd.DataFrame,
    channel: str,
    min_hours: int = EXTERNAL_INSUFFICIENT_MIN_HOURS,
) -> str:
    valid = scored[f"base_{channel}"].notna() & scored["fleet_median"].notna()
    if int(valid.sum()) < min_hours:
        return "insufficient"
    if not episodes.empty:
        return "offset"
    return "clean"


def detect_offset_channel(
    frame: pd.DataFrame,
    channel: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    fleet = fleet_frame(frame, channel)
    episode_frames = []
    status_rows = []

    for station_id, station in frame.groupby("station_id", sort=True):
        scored = add_hour_flags(offset_score(station, fleet, channel), channel)
        if offset_status(scored, pd.DataFrame(), channel) == "insufficient":
            episodes = pd.DataFrame()
            status = "insufficient"
        else:
            episodes = offset_episodes_for_station(scored, str(station_id), channel)
            status = offset_status(scored, episodes, channel)
        status_rows.append(
            {
                "station_id": station_id,
                "channel": channel,
                "status": status,
                "peak_metric": float(scored["offset_abs_score"].max())
                if scored["offset_abs_score"].notna().any()
                else np.nan,
            }
        )
        if not episodes.empty:
            episode_frames.append(episodes)

    episodes_all = (
        pd.concat(episode_frames, ignore_index=True)
        if episode_frames
        else pd.DataFrame()
    )
    return pd.DataFrame(status_rows), episodes_all


def daily_ratio_frame(
    station: pd.DataFrame,
    min_hours_per_day: int = EXTERNAL_RATIO_MIN_HOURS_PER_DAY,
    window_days: int = EXTERNAL_RATIO_WINDOW_DAYS,
    min_days: int = EXTERNAL_RATIO_MIN_DAYS,
) -> pd.DataFrame:
    data = station.loc[:, ["time_utc", "clear_sky_ratio"]].copy()
    data["date"] = pd.to_datetime(data["time_utc"], utc=True).dt.floor("D")
    grouped = data.groupby("date")["clear_sky_ratio"].agg(
        ratio_daily_median="median",
        qualifying_hours="count",
    )
    grouped.loc[grouped["qualifying_hours"].lt(min_hours_per_day), "ratio_daily_median"] = np.nan
    full_index = pd.date_range(
        data["date"].min(),
        data["date"].max(),
        freq="D",
        tz="UTC",
    )
    result = grouped.reindex(full_index)
    result.index.name = "time_utc"
    result["ratio_rolling_median"] = result["ratio_daily_median"].rolling(
        window_days,
        min_periods=min_days,
    ).median()
    return result.reset_index()


def ratio_episodes_for_station(
    ratio: pd.DataFrame,
    station_id: str,
    gap_hours: int = 72,
    min_days: int = EXTERNAL_RATIO_MIN_DAYS,
) -> pd.DataFrame:
    data = ratio.copy()
    rel_column = "rel_ratio_rolling_median"
    if rel_column not in data.columns:
        data[rel_column] = data.get("rel_ratio", np.nan)
    data["ratio_over"] = data[rel_column].ge(EXTERNAL_REL_RATIO_MAX)
    data["ratio_under"] = data[rel_column].le(EXTERNAL_REL_RATIO_MIN)
    rows = []

    for direction, column in [("over", "ratio_over"), ("under", "ratio_under")]:
        for group in _merge_flagged_runs(data, "time_utc", column, gap_hours, 24):
            start = pd.to_datetime(group["time_utc"].min(), utc=True)
            end = pd.to_datetime(group["time_utc"].max(), utc=True)
            duration_hours = _duration_hours(start, end, 24)
            if duration_hours < min_days * 24:
                continue
            ratios = pd.to_numeric(group["ratio_rolling_median"], errors="coerce")
            rels = pd.to_numeric(group[rel_column], errors="coerce")
            rows.append(
                {
                    "station_id": station_id,
                    "channel": "solar",
                    "start_hour": start,
                    "end_hour": end,
                    "duration_hours": duration_hours,
                    "n_flagged_hours": int(len(group) * 24),
                    "direction": direction,
                    "ratio_p10": float(ratios.quantile(0.10)),
                    "ratio_p50": float(ratios.quantile(0.50)),
                    "ratio_p90": float(ratios.quantile(0.90)),
                    "rel_p10": float(rels.quantile(0.10)),
                    "rel_p50": float(rels.quantile(0.50)),
                    "rel_p90": float(rels.quantile(0.90)),
                    "peak_ratio": float(ratios.max() if direction == "over" else ratios.min()),
                    "peak_rel_ratio": float(rels.max() if direction == "over" else rels.min()),
                    "episode_type": "ratio",
                }
            )

    return pd.DataFrame(rows)


def ratio_status(
    ratio: pd.DataFrame,
    episodes: pd.DataFrame,
    min_days: int = EXTERNAL_RATIO_MIN_DAYS,
) -> str:
    valid_column = "rel_ratio" if "rel_ratio" in ratio.columns else "ratio_daily_median"
    if int(ratio[valid_column].notna().sum()) < min_days:
        return "insufficient"
    if not episodes.empty:
        return "offset"
    return "clean"


def detect_ratio(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    station_ratio_frames: dict[str, pd.DataFrame] = {}
    for station_id, station in frame.groupby("station_id", sort=True):
        station_ratio_frames[str(station_id)] = daily_ratio_frame(station)

    daily_medians = {
        sid: ratio.set_index("time_utc")["ratio_daily_median"]
        for sid, ratio in station_ratio_frames.items()
    }
    if daily_medians:
        combined = pd.DataFrame(daily_medians)
        valid_counts = combined.notna().sum(axis=1)
        fleet_daily = combined.median(axis=1)
        fleet_daily.loc[valid_counts.lt(EXTERNAL_OFFSET_MIN_FLEET_STATIONS)] = np.nan
    else:
        fleet_daily = pd.Series(dtype=float)

    status_rows = []
    episode_frames = []
    all_ratio_frames: list[pd.DataFrame] = []

    for station_id, ratio in station_ratio_frames.items():
        ratio_with_rel = ratio.copy()
        ratio_with_rel["station_id"] = station_id
        fleet_vals = fleet_daily.reindex(
            pd.DatetimeIndex(ratio_with_rel["time_utc"])
        )
        ratio_with_rel["fleet_daily_ratio"] = fleet_vals.values
        ratio_with_rel["rel_ratio"] = (
            ratio_with_rel["ratio_daily_median"] / ratio_with_rel["fleet_daily_ratio"]
        )
        ratio_with_rel["rel_ratio_rolling_median"] = ratio_with_rel["rel_ratio"].rolling(
            EXTERNAL_RATIO_WINDOW_DAYS,
            min_periods=EXTERNAL_RATIO_MIN_DAYS,
        ).median()
        all_ratio_frames.append(ratio_with_rel)

        episodes = ratio_episodes_for_station(ratio_with_rel, station_id)
        rolling = pd.to_numeric(
            ratio_with_rel["rel_ratio_rolling_median"],
            errors="coerce",
        )
        if rolling.notna().any():
            high = float(rolling.max())
            low = float(rolling.min())
            peak = high if abs(high - 1.0) >= abs(low - 1.0) else low
        else:
            peak = np.nan
        status_rows.append(
            {
                "station_id": station_id,
                "channel": "solar",
                "status": ratio_status(ratio_with_rel, episodes),
                "peak_metric": peak,
            }
        )
        if not episodes.empty:
            episode_frames.append(episodes)

    episodes_all = (
        pd.concat(episode_frames, ignore_index=True)
        if episode_frames
        else pd.DataFrame()
    )
    daily_ratios_all = (
        pd.concat(all_ratio_frames, ignore_index=True)
        if all_ratio_frames
        else pd.DataFrame()
    )
    return pd.DataFrame(status_rows), episodes_all, daily_ratios_all


def detect_external_offsets(
    frame: pd.DataFrame,
    channels: list[str] = EXTERNAL_OFFSET_CHANNELS,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    status_frames = []
    episode_frames = []

    for channel in channels:
        status, episodes = detect_offset_channel(frame, channel)
        status_frames.append(status)
        if not episodes.empty:
            episode_frames.append(episodes)

    ratio_statuses, ratio_episodes, daily_ratios = detect_ratio(frame)
    status_frames.append(ratio_statuses)
    if not ratio_episodes.empty:
        episode_frames.append(ratio_episodes)

    statuses = pd.concat(status_frames, ignore_index=True)
    episodes = (
        pd.concat(episode_frames, ignore_index=True)
        if episode_frames
        else pd.DataFrame()
    )
    return statuses, episodes, daily_ratios


def _sensor_group(channel: str) -> str:
    if channel == "pressure":
        return "barometer"
    if channel in {"temp", "dewpoint"}:
        return "thermo_hygrometer"
    if channel == "solar":
        return "light_uv"
    return channel


def _reason(row: pd.Series) -> str:
    if row["episode_type"] == "ratio":
        return (
            "external_ratio"
            f"|channel={row['channel']}"
            f"|direction={row['direction']}"
            f"|rel_p50={float(row['rel_p50']):.3f}"
            f"|ratio_p50={float(row['ratio_p50']):.3f}"
            f"|peak_rel_ratio={float(row['peak_rel_ratio']):.3f}"
        )
    return (
        "external_offset"
        f"|channel={row['channel']}"
        f"|tier={row['tier']}"
        f"|level_p50={float(row['level_p50']):.3f}"
        f"|mean_score={float(row['mean_score']):.3f}"
    )


def build_offset_queue(episodes: pd.DataFrame) -> pd.DataFrame:
    if episodes.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    filtered = episodes.copy()
    if not EXTERNAL_SOLAR_QUEUE_ENABLED:
        filtered = filtered.loc[filtered["episode_type"] != "ratio"]

    if filtered.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    rows = []
    for _, row in filtered.sort_values(["start_hour", "station_id", "channel"]).iterrows():
        is_ratio = row["episode_type"] == "ratio"
        score = (
            abs(float(row["peak_rel_ratio"]) - 1.0)
            if is_ratio
            else float(row["peak_abs_score"])
        )
        role = (
            f"external_ratio_{row['direction']}"
            if is_ratio
            else f"external_offset_{str(row['tier']).lower()}"
        )
        rows.append(
            {
                "review_id": len(rows) + 1,
                "cluster_label": -9000 - len(rows),
                "cluster_size": 1,
                "role": role,
                "needs_5min_confirmation": False,
                "station_id": row["station_id"],
                "start_hour": row["start_hour"],
                "end_hour": row["end_hour"],
                "duration_hours": row["duration_hours"],
                "affected_sensor_groups": _sensor_group(str(row["channel"])),
                "n_sensor_groups": 1,
                "dominant_detector": "external_ratio" if is_ratio else "external_offset",
                "detector_concordance": 1,
                "max_abs_zscore": score,
                "max_iforest_score": np.nan,
                "min_rolling_variance": np.nan,
                "reasons": _reason(row),
                "cluster_probability": 1.0,
                "label": "calibration_offset",
            }
        )

    return pd.DataFrame(rows, columns=OUTPUT_COLUMNS)


def write_offset_queue(
    episodes: pd.DataFrame,
    path: Path = OFFSET_QUEUE_PATH,
) -> pd.DataFrame:
    queue = build_offset_queue(episodes)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    queue.to_csv(path, index=False)
    return queue
