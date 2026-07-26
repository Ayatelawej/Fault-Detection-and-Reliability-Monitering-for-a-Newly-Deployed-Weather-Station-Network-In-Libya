from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.config.paths import HOURLY_ROW_STATES_PATH
from src.features.row_state import (
    ROW_STATE_COMPLETE,
    ROW_STATE_PARTIAL,
    ROW_STATE_TRUE_OUTAGE,
)

HORIZONS = (6, 12, 24)
PRESENT_STATES = (ROW_STATE_COMPLETE, ROW_STATE_PARTIAL)
MODELED_STATES = PRESENT_STATES + (ROW_STATE_TRUE_OUTAGE,)
FEATURE_COLUMNS = [
    "trailing_missing_frac_6h",
    "trailing_missing_frac_24h",
    "trailing_missing_frac_72h",
    "n_gap_starts_72h",
    "hours_since_last_gap",
    "current_up_run_hours",
    "expanding_uptime_frac",
    "expanding_outage_event_count",
    "network_frac_stations_absent_now",
    "network_frac_stations_with_gap_24h",
]


@dataclass(frozen=True)
class RiskDataset:
    frame: pd.DataFrame
    horizons: tuple[int, ...] = HORIZONS

    def for_horizon(self, horizon: int) -> pd.DataFrame:
        key = f"eligible_{int(horizon)}h"
        label = f"y_{int(horizon)}h"
        if key not in self.frame.columns or label not in self.frame.columns:
            raise KeyError(horizon)
        columns = ["station_id", "hour_utc", label, *FEATURE_COLUMNS]
        return self.frame.loc[self.frame[key], columns].rename(columns={label: "y"}).reset_index(drop=True)


def load_hourly_row_states(path=HOURLY_ROW_STATES_PATH) -> pd.DataFrame:
    return pd.read_parquet(path)


def _require_columns(frame: pd.DataFrame, required: list[str]) -> None:
    missing = sorted(set(required).difference(frame.columns))
    if missing:
        raise KeyError(missing)


def prepare_states(frame: pd.DataFrame) -> pd.DataFrame:
    _require_columns(frame, ["station_id", "hour_utc", "row_state"])
    out = frame.loc[:, ["station_id", "hour_utc", "row_state"]].copy()
    out["station_id"] = out["station_id"].astype(str)
    out["hour_utc"] = pd.to_datetime(out["hour_utc"], utc=True, errors="coerce")
    out["row_state"] = out["row_state"].astype(str)
    out = out.loc[out["station_id"].notna() & out["hour_utc"].notna()].copy()
    out = out.loc[out["row_state"].isin(MODELED_STATES)].copy()
    out = out.sort_values(["station_id", "hour_utc"], kind="mergesort").reset_index(drop=True)
    if out.duplicated(["station_id", "hour_utc"]).any():
        raise ValueError("duplicate station-hour rows")
    out["is_present"] = out["row_state"].isin(PRESENT_STATES).astype(int)
    out["is_outage"] = out["row_state"].eq(ROW_STATE_TRUE_OUTAGE).astype(int)
    return out


def _station_features(group: pd.DataFrame, horizons: tuple[int, ...]) -> pd.DataFrame:
    out = group.sort_values("hour_utc", kind="mergesort").copy()
    outage = out["is_outage"].astype(int)
    present = out["is_present"].astype(int)
    gap_start = outage.eq(1) & outage.shift(fill_value=0).eq(0)
    for window in [6, 24, 72]:
        out[f"trailing_missing_frac_{window}h"] = outage.rolling(window, min_periods=1).mean().to_numpy(dtype=float)
    out["n_gap_starts_72h"] = gap_start.astype(int).rolling(72, min_periods=1).sum().to_numpy(dtype=float)
    gap_hours = out["hour_utc"].where(gap_start)
    last_gap = gap_hours.ffill()
    since = (out["hour_utc"] - last_gap) / pd.Timedelta(hours=1)
    out["hours_since_last_gap"] = np.log1p(since.fillna(720).clip(upper=720).astype(float))
    reset = outage.eq(1)
    run_group = reset.cumsum()
    up_run = present.groupby(run_group).cumsum().where(present.eq(1), 0)
    out["current_up_run_hours"] = np.log1p(up_run.astype(float))
    row_number = np.arange(1, len(out) + 1, dtype=float)
    out["expanding_uptime_frac"] = present.cumsum().to_numpy(dtype=float) / row_number
    out["expanding_outage_event_count"] = np.log1p(gap_start.astype(int).cumsum().to_numpy(dtype=float))
    outage_values = outage.to_numpy(dtype=int)
    n = len(out)
    for horizon in horizons:
        labels = np.zeros(n, dtype=int)
        eligible = np.zeros(n, dtype=bool)
        for index in range(n):
            end = index + horizon
            if end < n:
                eligible[index] = bool(present.iloc[index] == 1)
                labels[index] = int(outage_values[index + 1:end + 1].sum() > 0)
        out[f"eligible_{horizon}h"] = eligible
        out[f"y_{horizon}h"] = labels
    return out


def add_risk_features(states: pd.DataFrame, horizons: tuple[int, ...] = HORIZONS) -> pd.DataFrame:
    prepared = prepare_states(states)
    station_frames = [
        _station_features(group, horizons)
        for _, group in prepared.groupby("station_id", sort=False)
    ]
    out = pd.concat(station_frames, ignore_index=True)
    per_hour_absent = (
        out.groupby("hour_utc")["is_outage"]
        .mean()
        .rename("network_frac_stations_absent_now")
    )
    with_gap = (
        out.assign(station_gap_24h=out.groupby("station_id")["is_outage"].transform(
            lambda series: (series.eq(1) & series.shift(fill_value=0).eq(0)).rolling(24, min_periods=1).max()
        ))
        .groupby("hour_utc")["station_gap_24h"]
        .mean()
        .rename("network_frac_stations_with_gap_24h")
    )
    out = out.drop(columns=[column for column in ["network_frac_stations_absent_now", "network_frac_stations_with_gap_24h"] if column in out.columns])
    out = out.merge(per_hour_absent, on="hour_utc", how="left")
    out = out.merge(with_gap, on="hour_utc", how="left")
    out[FEATURE_COLUMNS] = out[FEATURE_COLUMNS].astype(float)
    return out.sort_values(["station_id", "hour_utc"], kind="mergesort").reset_index(drop=True)


def build_risk_dataset(
    hourly_row_states: pd.DataFrame | None = None,
    horizons: tuple[int, ...] = HORIZONS,
) -> RiskDataset:
    states = load_hourly_row_states() if hourly_row_states is None else hourly_row_states
    return RiskDataset(add_risk_features(states, horizons), horizons)
