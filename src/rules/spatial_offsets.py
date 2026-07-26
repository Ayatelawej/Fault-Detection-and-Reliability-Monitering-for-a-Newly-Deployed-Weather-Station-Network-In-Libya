from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.config.paths import MERGED_DATASET_PATH
from src.rules.config import (
    EXTERNAL_FEATURES_PATH,
    SPATIAL_ANOMALY_QUEUE_PATH,
    SPATIAL_EXCLUDED_NEIGHBORS,
    SPATIAL_FEATURES_PATH,
    SPATIAL_INSUFFICIENT_MIN_HOURS,
    SPATIAL_MIN_NEIGHBORS_PRESENT,
    SPATIAL_NEIGHBORS_PATH,
    SPATIAL_OFFSET_CHANNEL,
    SPATIAL_OFFSET_EXTERNAL_MARGIN_HPA,
    SPATIAL_OFFSET_GAP_HOURS,
    SPATIAL_OFFSET_MIN_DAYS,
    SPATIAL_OFFSET_MIN_DENSITY,
    SPATIAL_OFFSET_PHYSICAL_FLOOR_HPA,
    SPATIAL_OFFSET_REQUIRE_EXTERNAL_AGREEMENT,
    SPATIAL_OFFSET_SCORE_HIGH_HPA,
    SPATIAL_OFFSET_STABILITY_MAX_HPA,
    SPATIAL_RESIDUALS_PATH,
)
from src.rules.review_queue import OUTPUT_COLUMNS

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESIDUALS_PATH = PROJECT_ROOT / SPATIAL_RESIDUALS_PATH
EXTERNAL_FEATURES_FILE = PROJECT_ROOT / EXTERNAL_FEATURES_PATH
FEATURES_PATH = PROJECT_ROOT / SPATIAL_FEATURES_PATH
ANOMALY_QUEUE_PATH = PROJECT_ROOT / SPATIAL_ANOMALY_QUEUE_PATH
NEIGHBORS_PATH = PROJECT_ROOT / SPATIAL_NEIGHBORS_PATH
FEATURE_COLUMNS = [
    "station_id",
    "time_utc",
    "z_spatial_pressure",
    "z_spatial_temp",
    "z_spatial_dewpoint",
    "spatial_offset_level_pressure",
    "n_neighbors_present_pressure",
    "spatial_isolated",
]


def load_residuals(path: Path = RESIDUALS_PATH) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    frame["time_utc"] = pd.to_datetime(frame["time_utc"], utc=True)
    return frame.sort_values(["station_id", "time_utc"]).reset_index(drop=True)


def load_external_features(path: Path = EXTERNAL_FEATURES_FILE) -> pd.DataFrame:
    frame = pd.read_parquet(path, columns=["station_id", "time_utc", "r_pressure"])
    frame["time_utc"] = pd.to_datetime(frame["time_utc"], utc=True)
    return frame.sort_values(["station_id", "time_utc"]).reset_index(drop=True)


def add_hour_flags(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    level = pd.to_numeric(result["base_spatial_pressure"], errors="coerce")
    bmad = pd.to_numeric(result["bmad_spatial_pressure"], errors="coerce")
    large = level.abs().ge(SPATIAL_OFFSET_PHYSICAL_FLOOR_HPA)
    stable = bmad.le(SPATIAL_OFFSET_STABILITY_MAX_HPA)
    result["spatial_offset_flag"] = (large & stable).fillna(False)
    result["spatial_offset_high"] = (
        result["spatial_offset_flag"]
        & level.abs().ge(SPATIAL_OFFSET_SCORE_HIGH_HPA)
    )
    result["spatial_offset_med"] = (
        result["spatial_offset_flag"]
        & ~result["spatial_offset_high"]
    )
    return result


def _merge_flagged_runs(
    frame: pd.DataFrame,
    flag_column: str,
    gap_hours: int = SPATIAL_OFFSET_GAP_HOURS,
) -> list[pd.DataFrame]:
    flagged = frame.loc[frame[flag_column].fillna(False)].sort_values("time_utc")
    if flagged.empty:
        return []
    groups: list[pd.DataFrame] = []
    start = 0
    times = pd.to_datetime(flagged["time_utc"], utc=True).reset_index(drop=True)
    max_delta = pd.Timedelta(hours=gap_hours + 1)
    for position in range(1, len(flagged)):
        if times.iloc[position] - times.iloc[position - 1] > max_delta:
            groups.append(flagged.iloc[start:position].copy())
            start = position
    groups.append(flagged.iloc[start:].copy())
    return groups


def _duration_hours(start: pd.Timestamp, end: pd.Timestamp) -> float:
    return float((end - start) / pd.Timedelta(hours=1) + 1)


def _episode_row(group: pd.DataFrame, station_id: str) -> dict[str, object]:
    start = pd.to_datetime(group["time_utc"].min(), utc=True)
    end = pd.to_datetime(group["time_utc"].max(), utc=True)
    levels = pd.to_numeric(group["base_spatial_pressure"], errors="coerce")
    bmads = pd.to_numeric(group["bmad_spatial_pressure"], errors="coerce")
    return {
        "station_id": station_id,
        "channel": SPATIAL_OFFSET_CHANNEL,
        "start_hour": start,
        "end_hour": end,
        "duration_hours": _duration_hours(start, end),
        "n_flagged_hours": int(len(group)),
        "tier": "HIGH" if bool(group["spatial_offset_high"].any()) else "MED",
        "level_p10": float(levels.quantile(0.10)),
        "level_p50": float(levels.quantile(0.50)),
        "level_p90": float(levels.quantile(0.90)),
        "peak_abs_level": float(levels.abs().max()),
        "median_bmad": float(bmads.median()),
    }


def offset_episodes_for_station(
    scored: pd.DataFrame,
    station_id: str,
    min_days: int = SPATIAL_OFFSET_MIN_DAYS,
    min_density: float = SPATIAL_OFFSET_MIN_DENSITY,
) -> pd.DataFrame:
    rows = []
    for group in _merge_flagged_runs(scored, "spatial_offset_flag"):
        row = _episode_row(group, station_id)
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


def _external_median_for_episode(
    external: pd.DataFrame,
    row: pd.Series,
) -> float:
    station = external.loc[external["station_id"].eq(row["station_id"])]
    window = station.loc[
        pd.to_datetime(station["time_utc"], utc=True).between(
            pd.to_datetime(row["start_hour"], utc=True),
            pd.to_datetime(row["end_hour"], utc=True),
            inclusive="both",
        )
    ]
    values = pd.to_numeric(window["r_pressure"], errors="coerce").dropna()
    return float(values.median()) if not values.empty else np.nan


def apply_external_agreement_gate(
    episodes: pd.DataFrame,
    external: pd.DataFrame | None,
    require_agreement: bool = SPATIAL_OFFSET_REQUIRE_EXTERNAL_AGREEMENT,
    margin_hpa: float = SPATIAL_OFFSET_EXTERNAL_MARGIN_HPA,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if episodes.empty or not require_agreement:
        kept = episodes.copy()
        kept["external_r_pressure_median"] = np.nan
        return kept, pd.DataFrame()
    if external is None:
        raise ValueError("external pressure features are required")
    kept_rows = []
    gated_rows = []
    for _, row in episodes.iterrows():
        median = _external_median_for_episode(external, row)
        level = float(row["level_p50"])
        if level < 0:
            keep = pd.notna(median) and median <= -margin_hpa
            reason = "negative-not-confirmed"
        else:
            keep = pd.notna(median) and median >= margin_hpa
            reason = "positive-not-confirmed"
        out = row.to_dict()
        out["external_r_pressure_median"] = median
        if keep:
            kept_rows.append(out)
        else:
            out["gate_reason"] = reason
            gated_rows.append(out)
    kept = pd.DataFrame(kept_rows)
    gated = pd.DataFrame(gated_rows)
    return kept, gated


def spatial_status(
    station: pd.DataFrame,
    episodes: pd.DataFrame,
    min_hours: int = SPATIAL_INSUFFICIENT_MIN_HOURS,
) -> str:
    residual = pd.to_numeric(station["r_spatial_pressure"], errors="coerce")
    base = pd.to_numeric(station["base_spatial_pressure"], errors="coerce")
    bmad = pd.to_numeric(station["bmad_spatial_pressure"], errors="coerce")
    if residual.isna().all():
        return "isolated"
    if int((base.notna() & bmad.notna()).sum()) < min_hours:
        return "insufficient"
    if not episodes.empty:
        return "offset"
    return "clean"


def detect_spatial_offsets(
    frame: pd.DataFrame,
    external: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    statuses = []
    episode_frames = []
    gated_frames = []
    for station_id, station in frame.groupby("station_id", sort=True):
        scored = add_hour_flags(station)
        empty = pd.DataFrame()
        initial_status = spatial_status(scored, empty)
        if initial_status in {"isolated", "insufficient"}:
            episodes = empty
            gated = empty
            status = initial_status
        else:
            candidates = offset_episodes_for_station(scored, str(station_id))
            episodes, gated = apply_external_agreement_gate(candidates, external)
            status = spatial_status(scored, episodes)
        statuses.append(
            {
                "station_id": station_id,
                "status": status,
                "peak_abs_level": float(
                    pd.to_numeric(scored["base_spatial_pressure"], errors="coerce")
                    .abs()
                    .max()
                ),
            }
        )
        if not episodes.empty:
            episode_frames.append(episodes)
        if not gated.empty:
            gated_frames.append(gated)
    episodes_all = (
        pd.concat(episode_frames, ignore_index=True)
        if episode_frames
        else pd.DataFrame()
    )
    gated_all = (
        pd.concat(gated_frames, ignore_index=True)
        if gated_frames
        else pd.DataFrame()
    )
    return pd.DataFrame(statuses), episodes_all, gated_all


def _reason(row: pd.Series) -> str:
    return (
        "spatial_offset"
        f"|channel={row['channel']}"
        f"|tier={row['tier']}"
        f"|level_p50={float(row['level_p50']):.3f}"
        f"|median_bmad={float(row['median_bmad']):.3f}"
    )


def build_spatial_queue(episodes: pd.DataFrame) -> pd.DataFrame:
    if episodes.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    rows = []
    for _, row in episodes.sort_values(["start_hour", "station_id"]).iterrows():
        rows.append(
            {
                "review_id": len(rows) + 1,
                "cluster_label": -9100 - len(rows),
                "cluster_size": 1,
                "role": f"spatial_offset_{str(row['tier']).lower()}",
                "needs_5min_confirmation": False,
                "station_id": row["station_id"],
                "start_hour": row["start_hour"],
                "end_hour": row["end_hour"],
                "duration_hours": row["duration_hours"],
                "affected_sensor_groups": "barometer",
                "n_sensor_groups": 1,
                "dominant_detector": "spatial_offset",
                "detector_concordance": 1,
                "max_abs_zscore": row["peak_abs_level"],
                "max_iforest_score": np.nan,
                "min_rolling_variance": np.nan,
                "reasons": _reason(row),
                "cluster_probability": 1.0,
                "label": "spatial_anomaly",
            }
        )
    return pd.DataFrame(rows, columns=OUTPUT_COLUMNS)


def write_spatial_queue(
    episodes: pd.DataFrame,
    path: Path = ANOMALY_QUEUE_PATH,
) -> pd.DataFrame:
    queue = build_spatial_queue(episodes)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    queue.to_csv(path, index=False)
    return queue


def feature_has_label_leak(frame: pd.DataFrame) -> bool:
    columns = [str(column).lower() for column in frame.columns]
    return any("label" in column or "verdict" in column for column in columns)


def neighbor_present_counts(
    residuals: pd.DataFrame,
    merged: pd.DataFrame | None = None,
    graph: pd.DataFrame | None = None,
) -> pd.Series:
    frame = residuals.loc[:, ["station_id", "time_utc"]].copy()
    frame["time_utc"] = pd.to_datetime(frame["time_utc"], utc=True)
    if merged is None or graph is None:
        return pd.Series(
            np.where(
                residuals["r_spatial_pressure"].notna(),
                SPATIAL_MIN_NEIGHBORS_PRESENT,
                0,
            ),
            index=residuals.index,
            dtype=int,
        )
    merged_frame = merged.copy()
    if "time_utc" not in merged_frame.columns:
        merged_frame = merged_frame.rename(columns={"hour_utc": "time_utc"})
    merged_frame["time_utc"] = pd.to_datetime(merged_frame["time_utc"], utc=True)
    pivot = merged_frame.pivot_table(
        index="time_utc",
        columns="station_id",
        values="pressure_max_hpa",
        aggfunc="first",
    )
    values = []
    for station_id, station_rows in frame.groupby("station_id", sort=False):
        neighbors = graph.loc[
            graph["station_id"].eq(station_id),
            "neighbor_id",
        ].astype(str).tolist()
        neighbors = [
            neighbor
            for neighbor in neighbors
            if neighbor not in set(SPATIAL_EXCLUDED_NEIGHBORS)
        ]
        available = [neighbor for neighbor in neighbors if neighbor in pivot.columns]
        if not available:
            counts = pd.Series(0, index=station_rows.index, dtype=int)
        else:
            neighbor_values = pivot.reindex(station_rows["time_utc"]).loc[:, available]
            counts = pd.Series(
                neighbor_values.notna().sum(axis=1).to_numpy(dtype=int),
                index=station_rows.index,
            )
        values.append(counts)
    return pd.concat(values).sort_index()


def build_spatial_features(
    residuals: pd.DataFrame,
    neighbor_counts: pd.Series | None = None,
) -> pd.DataFrame:
    frame = residuals.copy()
    frame["time_utc"] = pd.to_datetime(frame["time_utc"], utc=True)
    features = frame.loc[
        :,
        [
            "station_id",
            "time_utc",
            "z_spatial_pressure",
            "z_spatial_temp",
            "z_spatial_dewpoint",
            "base_spatial_pressure",
        ],
    ].rename(columns={"base_spatial_pressure": "spatial_offset_level_pressure"})
    if neighbor_counts is None:
        neighbor_counts = neighbor_present_counts(frame)
    features["n_neighbors_present_pressure"] = neighbor_counts.reindex(frame.index).fillna(0).astype(int)
    isolated = frame.groupby("station_id")["r_spatial_pressure"].transform(
        lambda values: values.isna().all()
    )
    features["spatial_isolated"] = isolated.astype(bool)
    return features.loc[:, FEATURE_COLUMNS]


def write_spatial_features(
    residuals: pd.DataFrame,
    neighbor_counts: pd.Series | None = None,
    path: Path = FEATURES_PATH,
) -> pd.DataFrame:
    features = build_spatial_features(residuals, neighbor_counts)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(path, index=False)
    return features
