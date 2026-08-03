from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.config.paths import (
    AVAILABILITY_CLASSIFICATION_PATH,
    HOURLY_ROW_STATES_PATH,
    MEASUREMENT_COLUMNS,
    MERGED_DATASET_PATH,
    NETWORK_OUTAGE_WINDOWS_PATH,
    STATION_REGISTRY_PATH,
)
from src.availability.build_availability_events import SENSOR_GROUP_ORDER
from src.features.row_state import (
    ROW_STATE_COMPLETE,
    ROW_STATE_PARTIAL,
    ROW_STATE_TRUE_OUTAGE,
    ROW_STATE_WARMUP,
)
from src.rules.channel_handlers import sensor_group_for_channel
from src.rules.config import (
    EXTERNAL_CACHE_DIR,
    EXTERNAL_CLEARSKY_REF_MIN,
    SPATIAL_MIN_NEIGHBORS_PRESENT,
    SPATIAL_NEIGHBOR_RADIUS_KM,
)
from src.rules.physical_limits import physical_limit_flags
from src.model.hourly_detection import (
    FEATURE_PATH as HOURLY_FEATURE_PATH,
    LABEL_PATH as EPISODE_LABEL_PATH,
    SOURCE_PATH as HOURLY_SOURCE_PATH,
    build_hourly_labels,
    load_hourly_frame,
    load_labelled_episodes,
)

HORIZONS = (6, 12, 24)
EVENT_HISTORY_SOURCES = {
    "fault": "fault_target",
    "outage": "is_outage",
}
EVENT_HISTORY_KNOWN_COLUMNS = {
    "fault": "fault_scoreable",
    "outage": None,
}
EVENT_HISTORY_FIXED_WINDOWS = (24, 24 * 7)
EVENT_HISTORY_LONG_WINDOWS = (24 * 7, 24 * 30)
INCIDENT_TARGETS = (
    "fault",
    "fault_statistical_anomaly",
    "fault_stuck_flatline",
    "outage",
)
INCIDENT_FAULT_MECHANISM_BY_TARGET = {
    "fault_statistical_anomaly": "statistical_anomaly",
    "fault_stuck_flatline": "stuck_flatline",
}
FAULT_INCIDENT_MIN_DURATION_HOURS = 3
OUTAGE_INCIDENT_MIN_DURATION_HOURS = 6
ONSET_RECOVERY_EXCLUSION_HOURS = 24
INCIDENT_FAULT_WINDOW_POLICY = "censor_future_unobservable_hours"
INCIDENT_OUTAGE_NETWORK_POLICY = "censor_network_associated_outages"
PRESENT_STATES = (ROW_STATE_COMPLETE, ROW_STATE_PARTIAL)
TRANSMITTING_STATES = PRESENT_STATES + (ROW_STATE_WARMUP,)
MODELED_STATES = PRESENT_STATES + (ROW_STATE_TRUE_OUTAGE,)
AVAILABILITY_CLASS_FULL_OUTAGE = "full_outage"
AVAILABILITY_CLASS_PARTIAL_OUTAGE = "partial_outage"
AVAILABILITY_CLASS_ONLINE = "online"
AVAILABILITY_CLASS_EXCLUDED = "excluded"
SOURCE_KIND_OBSERVED = "observed_row"
SOURCE_KIND_GRID_MATERIALIZED = "materialized_risk_grid_gap"
FAULT_WINDOW_POLICY = "excluded_hours_neutral"
FAULT_WINDOW_POLICY_DESCRIPTION = (
    "Excluded hours remain on the continuous clock but are neutral in a future "
    "window: they are not counted as faults and do not invalidate the target. "
    "The target therefore means that a confirmed labelled fault occurs in the "
    "next H clock-hours."
)
FAULT_LABEL_REQUIRED_COLUMNS = [
    "station_id",
    "hour",
    "fault_hour",
    "display_state",
    "training_eligible",
]
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
FORECAST_HISTORY_WINDOWS = (6, 12, 24, 48)
FORECAST_HISTORY_LAGS = (1, 6, 12, 24, 48)
FORECAST_RATE_LAGS = (1, 6, 24)
FORECAST_HISTORY_STATISTICS = ("mean", "std", "min", "max")
FORECAST_EXTERNAL_CHANNELS = {
    "pressure": ("pressure_max_hpa", "pressure_msl", 1.0),
    "temp": ("temp_avg_c", "temperature_2m", 1.0),
    "dewpoint": ("dewpoint_avg_c", "dew_point_2m", 1.0),
    "wind": ("windspeed_avg_kmh", "wind_speed_10m", 1.0 / 3.6),
    "solar": (
        "solar_radiation_high_wm2",
        "shortwave_radiation",
        1.0,
    ),
}
FORECAST_SPATIAL_CHANNELS = {
    "pressure": "pressure_max_hpa",
    "temp": "temp_avg_c",
    "dewpoint": "dewpoint_avg_c",
}
FORECAST_DETECTOR_KINDS = ("physical", "stuck", "deviation")
FORECAST_DETECTOR_WINDOWS = (6, 24, 48)
FORECAST_STUCK_WINDOW_HOURS = 24
FORECAST_STUCK_VARIANCE_THRESHOLD = 1e-6
FORECAST_DEVIATION_WINDOW_HOURS = 48
FORECAST_DEVIATION_MIN_PERIODS = 12
FORECAST_DEVIATION_Z_THRESHOLD = 3.5
INCIDENT_HAZARD_RAW_CHANNELS = (
    "temp_avg_c",
    "humidity_avg_pct",
    "pressure_max_hpa",
    "pressure_min_hpa",
    "pressure_trend_hpa",
    "windspeed_avg_kmh",
    "windgust_high_kmh",
    "solar_radiation_high_wm2",
    "uv_high",
    "precip_rate_mmh",
    "precip_total_mm",
    "dewpoint_avg_c",
)
INCIDENT_HAZARD_HISTORY_LAGS = (1, 6, 24, 48)
INCIDENT_HAZARD_HISTORY_WINDOWS = (6, 24, 48)
INCIDENT_HAZARD_Z_WINDOWS = (6, 24, 48)
INCIDENT_HAZARD_SEQUENCE_WINDOW_HOURS = 48

BASE_CAUSAL_FORECAST_FEATURE_COLUMNS = (
    "past_full_outage_frac_6h",
    "past_full_outage_frac_24h",
    "past_full_outage_frac_72h",
    "past_full_outage_start_count_72h",
    "hours_since_last_full_outage_log",
    "past_transmitting_run_hours_log",
    "past_partial_outage_frac_6h",
    "past_partial_outage_frac_24h",
    "past_n_raw_records_mean_6h",
    "past_n_raw_records_mean_24h",
    "past_network_full_outage_frac_1h",
    "past_network_full_outage_start_frac_24h",
    "past_network_partial_outage_frac_1h",
    *(
        f"past_sensor_group_absent_frac_24h_{group}"
        for group in SENSOR_GROUP_ORDER
    ),
    "past_physical_limit_any_frac_6h",
    "past_physical_limit_any_frac_24h",
    *(f"past_physical_limit_frac_24h_{group}" for group in SENSOR_GROUP_ORDER),
    "ctx_elevation",
    "ctx_n_neighbors",
    "hour_of_day_sin",
    "hour_of_day_cos",
    "day_of_year_sin",
    "day_of_year_cos",
)
RAW_CAUSAL_FORECAST_FEATURE_COLUMNS = (
    "raw_now_n_raw_records",
    *(f"raw_now_{channel}" for channel in MEASUREMENT_COLUMNS),
    "raw_now_winddir_sin",
    "raw_now_winddir_cos",
    *(
        f"raw_lag_{lag}h_{channel}"
        for channel in MEASUREMENT_COLUMNS
        for lag in FORECAST_HISTORY_LAGS
    ),
    *(
        f"raw_prior_{statistic}_{window}h_{channel}"
        for channel in MEASUREMENT_COLUMNS
        for window in FORECAST_HISTORY_WINDOWS
        for statistic in FORECAST_HISTORY_STATISTICS
    ),
    *(
        f"raw_delta_{lag}h_{channel}"
        for channel in MEASUREMENT_COLUMNS
        for lag in FORECAST_RATE_LAGS
    ),
)
EXTERNAL_CAUSAL_FORECAST_FEATURE_COLUMNS = (
    *(f"era5_now_{short}" for short in FORECAST_EXTERNAL_CHANNELS),
    *(f"external_residual_now_{short}" for short in FORECAST_EXTERNAL_CHANNELS),
    "external_clear_sky_ratio_now",
    *(
        f"external_residual_prior_{statistic}_{window}h_{short}"
        for short in FORECAST_EXTERNAL_CHANNELS
        for window in FORECAST_HISTORY_WINDOWS
        for statistic in FORECAST_HISTORY_STATISTICS
    ),
    *(
        f"external_residual_delta_{lag}h_{short}"
        for short in FORECAST_EXTERNAL_CHANNELS
        for lag in FORECAST_RATE_LAGS
    ),
)
SPATIAL_CAUSAL_FORECAST_FEATURE_COLUMNS = (
    *(f"spatial_residual_now_{short}" for short in FORECAST_SPATIAL_CHANNELS),
    *(f"spatial_neighbor_count_now_{short}" for short in FORECAST_SPATIAL_CHANNELS),
    *(
        f"spatial_residual_prior_{statistic}_{window}h_{short}"
        for short in FORECAST_SPATIAL_CHANNELS
        for window in FORECAST_HISTORY_WINDOWS
        for statistic in FORECAST_HISTORY_STATISTICS
    ),
    *(
        f"spatial_residual_delta_{lag}h_{short}"
        for short in FORECAST_SPATIAL_CHANNELS
        for lag in FORECAST_RATE_LAGS
    ),
)
DETECTOR_CAUSAL_FORECAST_FEATURE_COLUMNS = (
    *(
        f"causal_detector_{kind}_now_{group}"
        for kind in FORECAST_DETECTOR_KINDS
        for group in SENSOR_GROUP_ORDER
    ),
    *(
        f"causal_detector_{kind}_count_{window}h_{group}"
        for kind in FORECAST_DETECTOR_KINDS
        for window in FORECAST_DETECTOR_WINDOWS
        for group in SENSOR_GROUP_ORDER
    ),
    *(
        f"causal_detector_{kind}_any_now"
        for kind in FORECAST_DETECTOR_KINDS
    ),
    *(
        f"causal_detector_{kind}_any_count_{window}h"
        for kind in FORECAST_DETECTOR_KINDS
        for window in FORECAST_DETECTOR_WINDOWS
    ),
)
CAUSAL_FORECAST_FEATURE_COLUMNS = (
    *BASE_CAUSAL_FORECAST_FEATURE_COLUMNS,
    *RAW_CAUSAL_FORECAST_FEATURE_COLUMNS,
    *EXTERNAL_CAUSAL_FORECAST_FEATURE_COLUMNS,
    *SPATIAL_CAUSAL_FORECAST_FEATURE_COLUMNS,
    *DETECTOR_CAUSAL_FORECAST_FEATURE_COLUMNS,
)
CAUSAL_FORECAST_FEATURE_COUNT = len(CAUSAL_FORECAST_FEATURE_COLUMNS)

INCIDENT_HAZARD_TEMPORAL_FEATURE_COLUMNS = (
    "raw_now_n_raw_records",
    *(f"raw_now_{channel}" for channel in INCIDENT_HAZARD_RAW_CHANNELS),
    *(f"external_residual_now_{short}" for short in FORECAST_EXTERNAL_CHANNELS),
    *(f"spatial_residual_now_{short}" for short in FORECAST_SPATIAL_CHANNELS),
    *(f"causal_detector_{kind}_any_now" for kind in FORECAST_DETECTOR_KINDS),
)
INCIDENT_HAZARD_CONTEXT_FEATURE_COLUMNS = (
    "incident_station_age_hours",
    "ctx_elevation",
    "ctx_n_neighbors",
    "hour_of_day_sin",
    "hour_of_day_cos",
    "day_of_year_sin",
    "day_of_year_cos",
)
INCIDENT_HAZARD_EVIDENCE_FEATURE_COLUMNS = tuple(
    dict.fromkeys(
        [
            *(
                column
                for column in BASE_CAUSAL_FORECAST_FEATURE_COLUMNS
                if column not in INCIDENT_HAZARD_CONTEXT_FEATURE_COLUMNS
            ),
            *(f"external_residual_now_{short}" for short in FORECAST_EXTERNAL_CHANNELS),
            *(f"spatial_residual_now_{short}" for short in FORECAST_SPATIAL_CHANNELS),
            *(
                f"incident_z_48h_{channel}"
                for channel in INCIDENT_HAZARD_RAW_CHANNELS
            ),
            *(
                f"incident_abs_z_ewma_24h_{channel}"
                for channel in INCIDENT_HAZARD_RAW_CHANNELS
            ),
            *(
                f"raw_delta_{lag}h_{channel}"
                for channel in INCIDENT_HAZARD_RAW_CHANNELS
                for lag in (1, 6)
            ),
            *DETECTOR_CAUSAL_FORECAST_FEATURE_COLUMNS,
        ]
    )
)


@dataclass(frozen=True)
class CausalForecastFeatureBundle:
    frame: pd.DataFrame
    feature_columns: tuple[str, ...]
    causality_audit: pd.DataFrame


@dataclass(frozen=True)
class IncidentHazardFeatureBundle:
    frame: pd.DataFrame
    numeric_feature_columns: tuple[str, ...]
    categorical_feature_columns: tuple[str, ...]
    causality_audit: pd.DataFrame


@dataclass(frozen=True)
class DiscreteHazardFeatureBundle:
    frame: pd.DataFrame
    numeric_feature_columns: tuple[str, ...]
    station_indicator_columns: tuple[str, ...]
    model_feature_columns: tuple[str, ...]
    causality_audit: pd.DataFrame


def _for_horizon_frame(
    frame: pd.DataFrame,
    horizon: int,
    feature_columns: tuple[str, ...] = (),
) -> pd.DataFrame:
    key = f"eligible_{int(horizon)}h"
    label = f"y_{int(horizon)}h"
    label_end = f"label_end_{int(horizon)}h"
    if key not in frame.columns or label not in frame.columns:
        raise KeyError(horizon)
    columns = ["station_id", "hour_utc", label_end, label, *feature_columns]
    return (
        frame.loc[frame[key], columns]
        .rename(columns={label_end: "label_end_utc", label: "y"})
        .reset_index(drop=True)
    )


@dataclass(frozen=True)
class RiskDataset:
    frame: pd.DataFrame
    horizons: tuple[int, ...] = HORIZONS

    def for_horizon(self, horizon: int) -> pd.DataFrame:
        return _for_horizon_frame(self.frame, horizon, tuple(FEATURE_COLUMNS))

    def label_change_summary(self) -> pd.DataFrame:
        return summarize_label_changes(self)


@dataclass(frozen=True)
class FaultRiskDataset:
    frame: pd.DataFrame
    horizons: tuple[int, ...] = HORIZONS
    window_policy: str = FAULT_WINDOW_POLICY

    def for_horizon(self, horizon: int) -> pd.DataFrame:
        return _for_horizon_frame(self.frame, horizon)

    def construction_summary(self) -> pd.DataFrame:
        return summarize_fault_risk_construction(self)


@dataclass(frozen=True)
class IncidentHazardDataset:
    frame: pd.DataFrame
    target: str
    minimum_duration_hours: int
    recovery_exclusion_hours: int = 0
    horizons: tuple[int, ...] = HORIZONS

    def for_horizon(self, horizon: int) -> pd.DataFrame:
        horizon = int(horizon)
        if horizon not in self.horizons:
            raise KeyError(horizon)
        eligible_column = f"incident_eligible_{horizon}h"
        label_column = f"incident_y_{horizon}h"
        label_end_column = f"incident_label_end_{horizon}h"
        event_column = f"incident_future_event_id_{horizon}h"
        required = [
            "station_id",
            "hour_utc",
            eligible_column,
            label_column,
            label_end_column,
            event_column,
        ]
        _require_columns(self.frame, required)
        result = self.frame.loc[
            self.frame[eligible_column].astype(bool), required
        ].copy()
        result = result.rename(
            columns={
                label_column: "y",
                label_end_column: "label_end_utc",
                event_column: "future_event_id",
            }
        )
        result["target"] = self.target
        result["prediction_horizon_h"] = horizon
        result["minimum_duration_hours"] = int(self.minimum_duration_hours)
        result["recovery_exclusion_hours"] = int(self.recovery_exclusion_hours)
        result["label_span_hours"] = int(horizon + self.minimum_duration_hours - 1)
        return result.reset_index(drop=True)

    def construction_summary(self) -> pd.DataFrame:
        return summarize_incident_hazard_construction(self)


def load_hourly_row_states(path=HOURLY_ROW_STATES_PATH) -> pd.DataFrame:
    return pd.read_parquet(path)


def load_operational_availability(path=AVAILABILITY_CLASSIFICATION_PATH) -> pd.DataFrame:
    return pd.read_parquet(path)


def _require_columns(frame: pd.DataFrame, required: list[str]) -> None:
    missing = sorted(set(required).difference(frame.columns))
    if missing:
        raise KeyError(missing)


def _normalise_source(frame: pd.DataFrame) -> pd.DataFrame:
    _require_columns(frame, ["station_id", "hour_utc"])
    out = frame.copy(deep=True)
    out["station_id"] = out["station_id"].astype("string")
    out["hour_utc"] = pd.to_datetime(out["hour_utc"], utc=True, errors="coerce")
    out = out.loc[out["station_id"].notna() & out["hour_utc"].notna()].copy()
    if out.duplicated(["station_id", "hour_utc"]).any():
        raise ValueError("duplicate station-hour rows")

    if "availability_class" in out.columns:
        availability_class = out["availability_class"].astype("string")
        row_state = (
            out["row_state"].astype("string")
            if "row_state" in out.columns
            else pd.Series(pd.NA, index=out.index, dtype="string")
        )
        source_kind = (
            out["source_kind"].astype("string")
            if "source_kind" in out.columns
            else pd.Series(SOURCE_KIND_OBSERVED, index=out.index, dtype="string")
        )
        out["is_outage"] = availability_class.eq(AVAILABILITY_CLASS_FULL_OUTAGE)
        out["is_transmitting"] = availability_class.isin(
            [AVAILABILITY_CLASS_ONLINE, AVAILABILITY_CLASS_PARTIAL_OUTAGE]
        )
        out["is_scoreable"] = (
            out["is_transmitting"] & ~row_state.eq(ROW_STATE_WARMUP)
        )
        out["in_coverage"] = availability_class.ne(AVAILABILITY_CLASS_EXCLUDED)
        out["row_state"] = row_state
        out["source_kind"] = source_kind
    else:
        _require_columns(out, ["row_state"])
        row_state = out["row_state"].astype("string")
        out["row_state"] = row_state
        out["is_outage"] = row_state.eq(ROW_STATE_TRUE_OUTAGE)
        out["is_transmitting"] = row_state.isin(TRANSMITTING_STATES)
        out["is_scoreable"] = row_state.isin(PRESENT_STATES)
        out["in_coverage"] = out["is_outage"] | out["is_transmitting"]
        out["source_kind"] = SOURCE_KIND_OBSERVED

    out["station_id"] = out["station_id"].astype(str)
    for column in [
        "is_outage",
        "is_transmitting",
        "is_scoreable",
        "in_coverage",
    ]:
        out[column] = out[column].astype("boolean").fillna(False).astype(bool)
    out["is_materialized_gap"] = out["source_kind"].eq(
        "materialized_structural_gap"
    )
    return out.sort_values(["station_id", "hour_utc"], kind="mergesort").reset_index(
        drop=True
    )


def _legacy_row_offset_targets(
    source: pd.DataFrame,
    horizons: tuple[int, ...],
) -> pd.DataFrame:
    observed = source.loc[
        source["source_kind"].eq(SOURCE_KIND_OBSERVED)
        & source["row_state"].isin(MODELED_STATES),
        ["station_id", "hour_utc", "is_outage", "is_scoreable"],
    ].copy()
    if observed.empty:
        columns = ["station_id", "hour_utc"]
        for horizon in horizons:
            columns.extend(
                [f"legacy_eligible_{horizon}h", f"legacy_y_{horizon}h"]
            )
        return pd.DataFrame(columns=columns)

    frames = []
    for _, group in observed.groupby("station_id", sort=False):
        out = group.sort_values("hour_utc", kind="mergesort").copy()
        outage = out["is_outage"].astype(int).to_numpy(dtype=int)
        scoreable = out["is_scoreable"].to_numpy(dtype=bool)
        n_rows = len(out)
        for horizon in horizons:
            eligible = np.zeros(n_rows, dtype=bool)
            labels = np.zeros(n_rows, dtype=int)
            for index in range(n_rows):
                end = index + int(horizon)
                if end < n_rows and scoreable[index]:
                    eligible[index] = True
                    labels[index] = int(outage[index + 1:end + 1].any())
            out[f"legacy_eligible_{horizon}h"] = eligible
            out[f"legacy_y_{horizon}h"] = labels
        frames.append(out.drop(columns=["is_outage", "is_scoreable"]))
    return pd.concat(frames, ignore_index=True)


def prepare_states(frame: pd.DataFrame) -> pd.DataFrame:
    source = _normalise_source(frame)
    station_frames = []
    for station_id, group in source.loc[source["in_coverage"]].groupby(
        "station_id",
        sort=False,
    ):
        start = group["hour_utc"].min()
        end = group["hour_utc"].max()
        grid = pd.DataFrame(
            {
                "station_id": station_id,
                "hour_utc": pd.date_range(start, end, freq="h"),
            }
        )
        columns = [
            "hour_utc",
            "row_state",
            "is_outage",
            "is_transmitting",
            "is_scoreable",
            "source_kind",
            "is_materialized_gap",
        ]
        merged = grid.merge(group.loc[:, columns], on="hour_utc", how="left")
        missing_source_row = merged["source_kind"].isna()
        merged.loc[missing_source_row, "row_state"] = ROW_STATE_TRUE_OUTAGE
        merged.loc[missing_source_row, "is_outage"] = True
        merged.loc[missing_source_row, "is_transmitting"] = False
        merged.loc[missing_source_row, "is_scoreable"] = False
        merged.loc[missing_source_row, "source_kind"] = SOURCE_KIND_GRID_MATERIALIZED
        merged.loc[missing_source_row, "is_materialized_gap"] = True
        for column in [
            "is_outage",
            "is_transmitting",
            "is_scoreable",
            "is_materialized_gap",
        ]:
            merged[column] = merged[column].astype("boolean").fillna(False).astype(bool)
        station_frames.append(merged)
    if not station_frames:
        return pd.DataFrame(
            columns=[
                "station_id",
                "hour_utc",
                "row_state",
                "is_outage",
                "is_transmitting",
                "is_scoreable",
                "source_kind",
                "is_materialized_gap",
            ]
        )
    out = pd.concat(station_frames, ignore_index=True)
    if out.duplicated(["station_id", "hour_utc"]).any():
        raise ValueError("continuous risk grid contains duplicate station-hour rows")
    return out.sort_values(["station_id", "hour_utc"], kind="mergesort").reset_index(
        drop=True
    )


def _add_future_targets(
    frame: pd.DataFrame,
    *,
    target_column: str,
    scoreable_column: str,
    horizons: tuple[int, ...],
    excluded_column: str | None = None,
) -> pd.DataFrame:
    out = frame.copy()
    target = (
        out[target_column].astype("boolean").fillna(False).to_numpy(dtype=bool)
    )
    scoreable = (
        out[scoreable_column].astype("boolean").fillna(False).to_numpy(dtype=bool)
    )
    excluded = (
        out[excluded_column].astype("boolean").fillna(False).to_numpy(dtype=bool)
        if excluded_column is not None
        else None
    )
    n_rows = len(out)
    for horizon in horizons:
        horizon = int(horizon)
        labels = np.zeros(n_rows, dtype=int)
        eligible = np.zeros(n_rows, dtype=bool)
        future_excluded = np.zeros(n_rows, dtype=bool)
        for index in range(n_rows):
            end = index + horizon
            if end < n_rows and scoreable[index]:
                eligible[index] = True
                labels[index] = int(target[index + 1 : end + 1].any())
                if excluded is not None:
                    future_excluded[index] = bool(
                        excluded[index + 1 : end + 1].any()
                    )
        out[f"eligible_{horizon}h"] = eligible
        out[f"y_{horizon}h"] = labels
        out[f"label_end_{horizon}h"] = out["hour_utc"] + pd.Timedelta(
            hours=horizon
        )
        if excluded is not None:
            out[f"future_excluded_{horizon}h"] = future_excluded
    return out


def _station_features(group: pd.DataFrame, horizons: tuple[int, ...]) -> pd.DataFrame:
    out = group.sort_values("hour_utc", kind="mergesort").copy()
    outage = out["is_outage"].astype(int)
    transmitting = out["is_transmitting"].astype(int)
    gap_start = outage.eq(1) & outage.shift(fill_value=0).eq(0)
    for window in [6, 24, 72]:
        out[f"trailing_missing_frac_{window}h"] = (
            outage.rolling(window, min_periods=1).mean().to_numpy(dtype=float)
        )
    out["n_gap_starts_72h"] = (
        gap_start.astype(int).rolling(72, min_periods=1).sum().to_numpy(dtype=float)
    )
    gap_hours = out["hour_utc"].where(gap_start)
    last_gap = gap_hours.ffill()
    since = (out["hour_utc"] - last_gap) / pd.Timedelta(hours=1)
    out["hours_since_last_gap"] = np.log1p(
        since.fillna(720).clip(upper=720).astype(float)
    )
    outage_group = outage.eq(1).cumsum()
    up_run = transmitting.groupby(outage_group).cumsum().where(transmitting.eq(1), 0)
    out["current_up_run_hours"] = np.log1p(up_run.astype(float))
    row_number = np.arange(1, len(out) + 1, dtype=float)
    out["expanding_uptime_frac"] = transmitting.cumsum().to_numpy(dtype=float) / row_number
    out["expanding_outage_event_count"] = np.log1p(
        gap_start.astype(int).cumsum().to_numpy(dtype=float)
    )

    return _add_future_targets(
        out,
        target_column="is_outage",
        scoreable_column="is_scoreable",
        horizons=horizons,
    )


def add_risk_features(
    states: pd.DataFrame,
    horizons: tuple[int, ...] = HORIZONS,
) -> pd.DataFrame:
    source = _normalise_source(states)
    legacy = _legacy_row_offset_targets(source, horizons)
    prepared = prepare_states(source)
    station_frames = [
        _station_features(group, horizons)
        for _, group in prepared.groupby("station_id", sort=False)
    ]
    if not station_frames:
        return prepared
    out = pd.concat(station_frames, ignore_index=True)
    out = out.merge(legacy, on=["station_id", "hour_utc"], how="left")
    for horizon in horizons:
        out[f"legacy_eligible_{horizon}h"] = (
            out[f"legacy_eligible_{horizon}h"]
            .astype("boolean")
            .fillna(False)
            .astype(bool)
        )
        out[f"legacy_y_{horizon}h"] = (
            pd.to_numeric(out[f"legacy_y_{horizon}h"], errors="coerce")
            .fillna(0)
            .astype(int)
        )

    per_hour_absent = (
        out.groupby("hour_utc")["is_outage"]
        .mean()
        .rename("network_frac_stations_absent_now")
    )
    with_gap = (
        out.assign(
            station_gap_24h=out.groupby("station_id")["is_outage"].transform(
                lambda series: (
                    series.eq(1) & series.shift(fill_value=0).eq(0)
                ).rolling(24, min_periods=1).max()
            )
        )
        .groupby("hour_utc")["station_gap_24h"]
        .mean()
        .rename("network_frac_stations_with_gap_24h")
    )
    out = out.drop(
        columns=[
            column
            for column in [
                "network_frac_stations_absent_now",
                "network_frac_stations_with_gap_24h",
            ]
            if column in out.columns
        ]
    )
    out = out.merge(per_hour_absent, on="hour_utc", how="left")
    out = out.merge(with_gap, on="hour_utc", how="left")
    out[FEATURE_COLUMNS] = out[FEATURE_COLUMNS].astype(float)
    return out.sort_values(["station_id", "hour_utc"], kind="mergesort").reset_index(
        drop=True
    )


def summarize_label_changes(dataset: RiskDataset) -> pd.DataFrame:
    rows = []
    for horizon in dataset.horizons:
        current_eligible = dataset.frame[f"eligible_{horizon}h"]
        legacy_eligible = dataset.frame[f"legacy_eligible_{horizon}h"]
        common = current_eligible & legacy_eligible
        changed = common & (
            dataset.frame[f"y_{horizon}h"].ne(
                dataset.frame[f"legacy_y_{horizon}h"]
            )
        )
        rows.append(
            {
                "horizon_h": int(horizon),
                "continuous_eligible_rows": int(current_eligible.sum()),
                "row_offset_eligible_rows": int(legacy_eligible.sum()),
                "common_eligible_rows": int(common.sum()),
                "changed_label_rows": int(changed.sum()),
                "changed_label_rate": (
                    float(changed.sum() / common.sum()) if common.any() else np.nan
                ),
                "continuous_only_eligible_rows": int(
                    (current_eligible & ~legacy_eligible).sum()
                ),
                "row_offset_only_eligible_rows": int(
                    (legacy_eligible & ~current_eligible).sum()
                ),
                "continuous_positive_rows": int(
                    dataset.frame.loc[current_eligible, f"y_{horizon}h"].sum()
                ),
                "row_offset_positive_rows": int(
                    dataset.frame.loc[legacy_eligible, f"legacy_y_{horizon}h"].sum()
                ),
            }
        )
    return pd.DataFrame(rows)


def build_live_hourly_fault_labels(
    source_path=HOURLY_SOURCE_PATH,
    feature_path=HOURLY_FEATURE_PATH,
    episode_label_path=EPISODE_LABEL_PATH,
) -> pd.DataFrame:
    hourly = load_hourly_frame(source_path, feature_path)
    episodes = load_labelled_episodes(episode_label_path)
    return build_hourly_labels(hourly, episodes)


def _normalise_fault_labels(frame: pd.DataFrame) -> pd.DataFrame:
    _require_columns(frame, FAULT_LABEL_REQUIRED_COLUMNS)
    out = frame.loc[:, FAULT_LABEL_REQUIRED_COLUMNS].copy()
    out["mechanisms"] = (
        frame["mechanisms"].fillna("").astype(str)
        if "mechanisms" in frame.columns
        else ""
    )
    out["station_id"] = out["station_id"].astype(str)
    out["hour"] = pd.to_datetime(out["hour"], utc=True, errors="coerce")
    if out["hour"].isna().any():
        raise ValueError("fault labels contain invalid station-hour timestamps")
    if out.duplicated(["station_id", "hour"]).any():
        raise ValueError("fault labels contain duplicate station-hour rows")
    out["display_state"] = out["display_state"].astype("string")
    states = {"fault", "benign", "clean", "excluded"}
    unknown_states = sorted(set(out["display_state"].dropna()).difference(states))
    if unknown_states:
        raise ValueError(f"fault labels contain unknown display states: {unknown_states}")
    out["training_eligible"] = (
        out["training_eligible"].astype("boolean").fillna(False).astype(bool)
    )
    numeric_fault = pd.to_numeric(out["fault_hour"], errors="coerce")
    if not numeric_fault.dropna().isin([0, 1]).all():
        raise ValueError("fault_hour must contain only zero, one, or missing")
    out["fault_hour"] = numeric_fault.astype("Float64")
    if not out.loc[out["training_eligible"], "fault_hour"].notna().all():
        raise ValueError("training-eligible fault labels must carry fault_hour")
    if not out.loc[out["display_state"].eq("fault"), "fault_hour"].eq(1).all():
        raise ValueError("fault display state must carry fault_hour equal to one")
    if not out.loc[out["display_state"].ne("fault") & out["fault_hour"].notna(), "fault_hour"].eq(0).all():
        raise ValueError("non-fault labelled hours must carry fault_hour equal to zero")
    return out.rename(columns={"hour": "hour_utc"}).sort_values(
        ["station_id", "hour_utc"],
        kind="mergesort",
    ).reset_index(drop=True)


def prepare_fault_risk_states(
    hourly_labels: pd.DataFrame,
    availability: pd.DataFrame,
    *,
    mechanism: str | None = None,
) -> pd.DataFrame:
    labels = _normalise_fault_labels(hourly_labels)
    availability_source = _normalise_source(availability)
    all_label_keys = labels.merge(
        availability_source.loc[:, ["station_id", "hour_utc"]],
        on=["station_id", "hour_utc"],
        how="left",
        indicator=True,
        validate="one_to_one",
    )
    if all_label_keys["_merge"].ne("both").any():
        raise ValueError("fault labels contain station-hours absent from availability")
    grid = prepare_states(availability_source)
    out = grid.merge(
        labels,
        on=["station_id", "hour_utc"],
        how="left",
        validate="one_to_one",
    )
    out["any_fault_target"] = (
        pd.to_numeric(out["fault_hour"], errors="coerce")
        .eq(1)
        .fillna(False)
        .astype(bool)
    )
    if mechanism is None:
        out["fault_target"] = out["any_fault_target"]
    else:
        mechanism = str(mechanism)
        mechanism_member = out["mechanisms"].fillna("").astype(str).map(
            lambda value: mechanism in {token for token in value.split("|") if token}
        )
        out["fault_target"] = out["any_fault_target"] & mechanism_member.astype(bool)
    training_eligible = (
        out["training_eligible"].astype("boolean").fillna(False).astype(bool)
    )
    warmup = out["row_state"].astype("string").eq(ROW_STATE_WARMUP)
    out["fault_scoreable"] = training_eligible & ~warmup
    out["fault_excluded"] = ~training_eligible
    source_fault_hours = int(labels["fault_hour"].eq(1).sum())
    if int(out["any_fault_target"].sum()) != source_fault_hours:
        raise ValueError("fault labels were lost while joining the continuous grid")
    return out.sort_values(["station_id", "hour_utc"], kind="mergesort").reset_index(
        drop=True
    )


def _fault_station_targets(
    group: pd.DataFrame,
    horizons: tuple[int, ...],
) -> pd.DataFrame:
    out = group.sort_values("hour_utc", kind="mergesort").copy()
    return _add_future_targets(
        out,
        target_column="fault_target",
        scoreable_column="fault_scoreable",
        horizons=horizons,
        excluded_column="fault_excluded",
    )


def build_fault_risk_dataset(
    hourly_labels: pd.DataFrame | None = None,
    availability: pd.DataFrame | None = None,
    horizons: tuple[int, ...] = HORIZONS,
) -> FaultRiskDataset:
    labels = (
        build_live_hourly_fault_labels()
        if hourly_labels is None
        else hourly_labels
    )
    availability_frame = (
        load_operational_availability() if availability is None else availability
    )
    prepared = prepare_fault_risk_states(labels, availability_frame)
    station_frames = [
        _fault_station_targets(group, horizons)
        for _, group in prepared.groupby("station_id", sort=False)
    ]
    if not station_frames:
        return FaultRiskDataset(prepared, horizons)
    frame = pd.concat(station_frames, ignore_index=True)
    return FaultRiskDataset(
        frame.sort_values(["station_id", "hour_utc"], kind="mergesort").reset_index(
            drop=True
        ),
        horizons,
    )


def _network_intervals_by_station(
    network_windows: pd.DataFrame | None,
) -> dict[str, list[tuple[pd.Timestamp, pd.Timestamp]]]:
    if network_windows is None or network_windows.empty:
        return {}
    _require_columns(
        network_windows,
        ["backfill_start_utc", "backfill_end_utc", "station_ids"],
    )
    source = network_windows.copy(deep=True)
    source["backfill_start_utc"] = pd.to_datetime(
        source["backfill_start_utc"], utc=True, errors="coerce"
    )
    source["backfill_end_utc"] = pd.to_datetime(
        source["backfill_end_utc"], utc=True, errors="coerce"
    )
    intervals: dict[str, list[tuple[pd.Timestamp, pd.Timestamp]]] = {}
    for row in source.itertuples(index=False):
        start = getattr(row, "backfill_start_utc")
        end = getattr(row, "backfill_end_utc")
        if pd.isna(start) or pd.isna(end) or end < start:
            continue
        for station_id in str(getattr(row, "station_ids", "")).split(";"):
            station_id = station_id.strip()
            if station_id:
                intervals.setdefault(station_id, []).append((start, end))
    return intervals


def _incident_station_targets(
    group: pd.DataFrame,
    *,
    target: str,
    state_column: str,
    at_risk_state_column: str | None,
    scoreable_column: str,
    censor_column: str | None,
    minimum_duration_hours: int,
    recovery_exclusion_hours: int,
    horizons: tuple[int, ...],
    network_intervals: list[tuple[pd.Timestamp, pd.Timestamp]] | None = None,
) -> pd.DataFrame:
    out = group.sort_values("hour_utc", kind="mergesort").copy()
    state = out[state_column].astype("boolean").fillna(False).to_numpy(dtype=bool)
    at_risk_state = (
        state
        if at_risk_state_column is None
        else out[at_risk_state_column].astype("boolean").fillna(False).to_numpy(dtype=bool)
    )
    scoreable = (
        out[scoreable_column].astype("boolean").fillna(False).to_numpy(dtype=bool)
    )
    if censor_column is None:
        censored = np.zeros(len(out), dtype=bool)
    else:
        censored = (
            out[censor_column].astype("boolean").fillna(False).to_numpy(dtype=bool)
        )

    starts = state & np.concatenate(([True], ~state[:-1]))
    prior_known_clean = (
        pd.Series(scoreable & ~at_risk_state, index=out.index)
        .shift(1, fill_value=False)
        .to_numpy(dtype=bool)
    )
    observed_starts = starts & prior_known_clean
    run_id = np.cumsum(starts).astype(float)
    run_id[~state] = np.nan
    run_duration = (
        pd.Series(run_id, index=out.index)
        .groupby(pd.Series(run_id, index=out.index), dropna=True)
        .transform("size")
        .fillna(0)
        .to_numpy(dtype=int)
    )
    event_id = np.full(len(out), "", dtype=object)
    for index in np.flatnonzero(starts):
        event_id[index] = f"{out['station_id'].iloc[index]}::{out['hour_utc'].iloc[index].isoformat()}"

    network_associated = np.zeros(len(out), dtype=bool)
    if target == "outage" and network_intervals:
        for start_index in np.flatnonzero(starts):
            duration = int(run_duration[start_index])
            event_start = out["hour_utc"].iloc[start_index]
            event_end = event_start + pd.Timedelta(hours=max(0, duration - 1))
            if any(
                interval_start <= event_end and interval_end >= event_start
                for interval_start, interval_end in network_intervals
            ):
                network_associated[start_index] = True

    qualifying_start = observed_starts & (run_duration >= int(minimum_duration_hours))
    competing_start = np.zeros(len(out), dtype=bool)
    if target == "outage":
        competing_start = starts & network_associated
        qualifying_start &= ~network_associated

    out["incident_start"] = starts
    out["incident_observed_start"] = observed_starts
    out["incident_unobserved_start"] = starts & ~observed_starts
    out["incident_duration_hours"] = run_duration
    out["incident_event_id"] = event_id
    out["incident_network_associated"] = network_associated
    out["incident_qualifying_start"] = qualifying_start
    prior_active = pd.Series(at_risk_state, index=out.index).shift(
        1,
        fill_value=False,
    )
    if recovery_exclusion_hours:
        recovery_excluded = (
            prior_active.astype(float)
            .rolling(int(recovery_exclusion_hours), min_periods=1)
            .max()
            .astype(bool)
            .to_numpy(dtype=bool)
        )
    else:
        recovery_excluded = np.zeros(len(out), dtype=bool)

    out["incident_base_scoreable"] = scoreable & ~at_risk_state
    out["incident_post_event_recovery_excluded"] = recovery_excluded
    out["incident_scoreable"] = (
        out["incident_base_scoreable"].astype(bool) & ~recovery_excluded
    )

    for horizon in horizons:
        horizon = int(horizon)
        label_span = horizon + int(minimum_duration_hours) - 1
        eligible = np.zeros(len(out), dtype=bool)
        labels = np.zeros(len(out), dtype=int)
        future_event_id = np.full(len(out), "", dtype=object)
        future_censored = np.zeros(len(out), dtype=bool)
        for index in range(len(out)):
            end = index + label_span
            if end >= len(out) or not out["incident_scoreable"].iloc[index]:
                continue
            censor_window = censored[index + 1 : end + 1]
            competing_window = competing_start[index + 1 : end + 1]
            if censor_window.any() or competing_window.any():
                future_censored[index] = True
                continue
            candidate = np.flatnonzero(qualifying_start[index + 1 : index + horizon + 1])
            eligible[index] = True
            if len(candidate):
                event_index = index + 1 + int(candidate[0])
                labels[index] = 1
                future_event_id[index] = event_id[event_index]
        out[f"incident_eligible_{horizon}h"] = eligible
        out[f"incident_y_{horizon}h"] = labels
        out[f"incident_label_end_{horizon}h"] = out["hour_utc"] + pd.Timedelta(
            hours=label_span
        )
        out[f"incident_future_event_id_{horizon}h"] = future_event_id
        out[f"incident_future_censored_{horizon}h"] = future_censored
    return out


def build_incident_hazard_dataset(
    target: str,
    *,
    hourly_labels: pd.DataFrame | None = None,
    availability: pd.DataFrame | None = None,
    network_windows: pd.DataFrame | None = None,
    horizons: tuple[int, ...] = HORIZONS,
    minimum_duration_hours: int | None = None,
    recovery_exclusion_hours: int = 0,
) -> IncidentHazardDataset:
    target = str(target)
    if target not in INCIDENT_TARGETS:
        raise ValueError(f"incident target must be one of {INCIDENT_TARGETS}")
    if minimum_duration_hours is None:
        minimum_duration_hours = (
            FAULT_INCIDENT_MIN_DURATION_HOURS
            if target in {"fault", *INCIDENT_FAULT_MECHANISM_BY_TARGET}
            else OUTAGE_INCIDENT_MIN_DURATION_HOURS
        )
    minimum_duration_hours = int(minimum_duration_hours)
    if minimum_duration_hours <= 0:
        raise ValueError("minimum incident duration must be positive")
    recovery_exclusion_hours = int(recovery_exclusion_hours)
    if recovery_exclusion_hours < 0:
        raise ValueError("recovery exclusion hours cannot be negative")

    if target in {"fault", *INCIDENT_FAULT_MECHANISM_BY_TARGET}:
        mechanism = INCIDENT_FAULT_MECHANISM_BY_TARGET.get(target)
        labels = (
            build_live_hourly_fault_labels()
            if hourly_labels is None
            else hourly_labels
        )
        availability_frame = (
            load_operational_availability() if availability is None else availability
        )
        source = prepare_fault_risk_states(
            labels,
            availability_frame,
            mechanism=mechanism,
        )
        state_column = "fault_target"
        at_risk_state_column: str | None = "any_fault_target"
        scoreable_column = "fault_scoreable"
        censor_column: str | None = "fault_excluded"
        interval_map: dict[str, list[tuple[pd.Timestamp, pd.Timestamp]]] = {}
    else:
        availability_frame = (
            load_operational_availability() if availability is None else availability
        )
        source = prepare_states(availability_frame)
        state_column = "is_outage"
        at_risk_state_column = None
        scoreable_column = "is_scoreable"
        censor_column = None
        windows = (
            pd.read_csv(NETWORK_OUTAGE_WINDOWS_PATH)
            if network_windows is None and NETWORK_OUTAGE_WINDOWS_PATH.exists()
            else network_windows
        )
        interval_map = _network_intervals_by_station(windows)

    station_frames = [
        _incident_station_targets(
            group,
            target=target,
            state_column=state_column,
            at_risk_state_column=at_risk_state_column,
            scoreable_column=scoreable_column,
            censor_column=censor_column,
            minimum_duration_hours=minimum_duration_hours,
            recovery_exclusion_hours=recovery_exclusion_hours,
            horizons=horizons,
            network_intervals=interval_map.get(str(station_id)),
        )
        for station_id, group in source.groupby("station_id", sort=False)
    ]
    frame = (
        pd.concat(station_frames, ignore_index=True)
        if station_frames
        else source.copy()
    )
    return IncidentHazardDataset(
        frame=frame.sort_values(["station_id", "hour_utc"], kind="mergesort").reset_index(
            drop=True
        ),
        target=target,
        minimum_duration_hours=minimum_duration_hours,
        recovery_exclusion_hours=recovery_exclusion_hours,
        horizons=tuple(int(value) for value in horizons),
    )


def summarize_incident_hazard_construction(
    dataset: IncidentHazardDataset,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    frame = dataset.frame
    for horizon in dataset.horizons:
        horizon = int(horizon)
        eligible = frame[f"incident_eligible_{horizon}h"].astype(bool)
        labels = frame.loc[eligible, f"incident_y_{horizon}h"].astype(int)
        rows.append(
            {
                "target": dataset.target,
                "horizon_h": horizon,
                "minimum_duration_hours": int(dataset.minimum_duration_hours),
                "recovery_exclusion_hours": int(dataset.recovery_exclusion_hours),
                "label_span_hours": horizon + int(dataset.minimum_duration_hours) - 1,
                "independent_incidents": int(frame["incident_start"].astype(bool).sum()),
                "observed_incident_starts": int(
                    frame["incident_observed_start"].astype(bool).sum()
                ),
                "unobserved_incident_starts_excluded": int(
                    frame["incident_unobserved_start"].astype(bool).sum()
                ),
                "qualifying_incidents": int(
                    frame["incident_qualifying_start"].astype(bool).sum()
                ),
                "scoreable_fault_free_or_online_rows": int(
                    frame["incident_scoreable"].astype(bool).sum()
                ),
                "base_scoreable_fault_free_or_online_rows": int(
                    frame["incident_base_scoreable"].astype(bool).sum()
                ),
                "post_incident_recovery_excluded_rows": int(
                    (
                        frame["incident_post_event_recovery_excluded"].astype(bool)
                        & frame["incident_base_scoreable"].astype(bool)
                    ).sum()
                ),
                "eligible_rows": int(eligible.sum()),
                "positive_rows": int(labels.sum()),
                "positive_rate": float(labels.mean()) if len(labels) else np.nan,
                "future_censored_rows": int(
                    frame[f"incident_future_censored_{horizon}h"].astype(bool).sum()
                ),
                "network_associated_incident_starts": int(
                    frame["incident_network_associated"].astype(bool).sum()
                ),
            }
        )
    return pd.DataFrame(rows)


def build_confirmed_incident_recurrence_history(
    dataset: IncidentHazardDataset,
    *,
    window_hours: int = 24 * 7,
) -> pd.DataFrame:
    window_hours = int(window_hours)
    if window_hours <= 0:
        raise ValueError("incident recurrence window must be positive")
    frame = dataset.frame
    _require_columns(
        frame,
        ["station_id", "hour_utc", "incident_qualifying_start"],
    )
    base = _normalise_forecast_keys(frame, "incident recurrence history")
    result = base.loc[:, ["station_id", "hour_utc"]].copy()
    column = f"confirmed_incident_start_count_trailing_{window_hours}h"
    for _, station in base.groupby("station_id", sort=False):
        station_index = station.index
        qualifying = station["incident_qualifying_start"].astype(bool).to_numpy()
        confirmed = np.zeros(len(station), dtype=bool)
        for start_index in np.flatnonzero(qualifying):
            confirmation_index = int(start_index) + int(dataset.minimum_duration_hours) - 1
            if confirmation_index < len(station):
                confirmed[confirmation_index] = True
        result.loc[station_index, column] = (
            pd.Series(confirmed, index=station_index)
            .shift(1, fill_value=False)
            .astype(float)
            .rolling(window_hours, min_periods=1)
            .sum()
            .to_numpy(dtype=float)
        )
    return result.sort_values(["station_id", "hour_utc"], kind="mergesort").reset_index(
        drop=True
    )


def summarize_fault_risk_construction(dataset: FaultRiskDataset) -> pd.DataFrame:
    rows = []
    for horizon in dataset.horizons:
        eligible = dataset.frame[f"eligible_{horizon}h"]
        positives = dataset.frame.loc[eligible, f"y_{horizon}h"]
        rows.append(
            {
                "horizon_h": int(horizon),
                "continuous_grid_rows": int(len(dataset.frame)),
                "current_scoreable_rows": int(dataset.frame["fault_scoreable"].sum()),
                "direct_fault_hours": int(dataset.frame["fault_target"].sum()),
                "label_excluded_grid_hours": int(
                    dataset.frame["fault_excluded"].sum()
                ),
                "warmup_rows": int(
                    dataset.frame["row_state"].astype("string").eq(ROW_STATE_WARMUP).sum()
                ),
                "warmup_fault_hours": int(
                    (
                        dataset.frame["fault_target"]
                        & dataset.frame["row_state"].astype("string").eq(ROW_STATE_WARMUP)
                    ).sum()
                ),
                "materialized_structural_gap_hours": int(
                    dataset.frame["is_materialized_gap"].sum()
                ),
                "eligible_rows": int(eligible.sum()),
                "positive_rows": int(positives.sum()),
                "positive_rate": (
                    float(positives.mean()) if len(positives) else np.nan
                ),
                "eligible_windows_with_excluded_hours": int(
                    dataset.frame.loc[eligible, f"future_excluded_{horizon}h"].sum()
                ),
            }
        )
    return pd.DataFrame(rows)


def summarize_fault_station_support(
    dataset: FaultRiskDataset,
    near_zero_threshold: int = 5,
) -> pd.DataFrame:
    summary = (
        dataset.frame.groupby("station_id", as_index=False)
        .agg(
            direct_fault_hours=("fault_target", "sum"),
            scoreable_rows=("fault_scoreable", "sum"),
        )
        .sort_values("station_id", kind="mergesort")
        .reset_index(drop=True)
    )
    summary["direct_fault_hours"] = summary["direct_fault_hours"].astype(int)
    summary["scoreable_rows"] = summary["scoreable_rows"].astype(int)
    summary["near_zero_direct_fault_hours_le"] = int(near_zero_threshold)
    summary["near_zero_direct_fault_support"] = summary["direct_fault_hours"].le(
        near_zero_threshold
    )
    return summary


def build_risk_dataset(
    hourly_row_states: pd.DataFrame | None = None,
    horizons: tuple[int, ...] = HORIZONS,
) -> RiskDataset:
    source = (
        load_operational_availability()
        if hourly_row_states is None
        else hourly_row_states
    )
    return RiskDataset(add_risk_features(source, horizons), horizons)


def _normalise_forecast_keys(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    _require_columns(frame, ["station_id", "hour_utc"])
    out = frame.copy(deep=True)
    out["station_id"] = out["station_id"].astype(str)
    out["hour_utc"] = pd.to_datetime(out["hour_utc"], utc=True, errors="coerce")
    if out[["station_id", "hour_utc"]].isna().any().any():
        raise ValueError(f"{name} contains invalid station-hour keys")
    if out.duplicated(["station_id", "hour_utc"]).any():
        raise ValueError(f"{name} contains duplicate station-hour keys")
    return out.sort_values(["station_id", "hour_utc"], kind="mergesort").reset_index(
        drop=True
    )


def _default_forecast_observations() -> pd.DataFrame:
    columns = ["station_id", "hour_utc", "n_raw_records", *MEASUREMENT_COLUMNS]
    return pd.read_csv(MERGED_DATASET_PATH, usecols=columns)


def _default_forecast_registry() -> pd.DataFrame:
    return pd.read_csv(
        STATION_REGISTRY_PATH,
        usecols=[
            "station_id",
            "latitude",
            "longitude",
            "elevation",
            "install_date",
        ],
    )


def load_exact_hour_reference(reference_dir: Path | None = None) -> pd.DataFrame:
    reference_dir = (
        Path(reference_dir)
        if reference_dir is not None
        else Path(__file__).resolve().parents[2] / EXTERNAL_CACHE_DIR
    )
    frames = []
    reference_columns = list(
        dict.fromkeys(spec[1] for spec in FORECAST_EXTERNAL_CHANNELS.values())
    )
    for path in sorted(reference_dir.glob("*.parquet")):
        frame = pd.read_parquet(path)
        if "time_utc" in frame.columns:
            source = frame.copy()
        else:
            source = frame.reset_index()
            if "time_utc" not in source.columns:
                source = source.rename(columns={source.columns[0]: "time_utc"})
        source["station_id"] = path.stem
        for column in reference_columns:
            if column not in source.columns:
                source[column] = np.nan
        frames.append(source.loc[:, ["station_id", "time_utc", *reference_columns]])
    if not frames:
        return pd.DataFrame(columns=["station_id", "hour_utc", *reference_columns])
    result = pd.concat(frames, ignore_index=True).rename(columns={"time_utc": "hour_utc"})
    return _normalise_forecast_keys(result, "hourly external reference")


def _default_forecast_reference() -> pd.DataFrame:
    return load_exact_hour_reference()


def load_causal_forecast_sources() -> dict[str, pd.DataFrame]:
    return {
        "availability": load_operational_availability(),
        "observations": _default_forecast_observations(),
        "registry": _default_forecast_registry(),
        "reference": _default_forecast_reference(),
    }


def _reference_lookup(frame: pd.DataFrame) -> pd.DataFrame:
    source = frame.copy(deep=True)
    if "hour_utc" not in source.columns and "time_utc" in source.columns:
        source = source.rename(columns={"time_utc": "hour_utc"})
    source = _normalise_forecast_keys(source, "hourly external reference")
    result = source.loc[:, ["station_id", "hour_utc"]].copy()
    for _, reference_column, _ in FORECAST_EXTERNAL_CHANNELS.values():
        result[reference_column] = (
            pd.to_numeric(source[reference_column], errors="coerce")
            if reference_column in source.columns
            else np.nan
        )
    return result


def _prior_window_summary(
    values: pd.Series,
    *,
    prefix: str,
    suffix: str = "",
    windows: tuple[int, ...] = FORECAST_HISTORY_WINDOWS,
) -> pd.DataFrame:
    prior = pd.to_numeric(values, errors="coerce").shift(1)
    result = pd.DataFrame(index=values.index)
    for window in windows:
        rolling = prior.rolling(int(window), min_periods=1)
        result[f"{prefix}_prior_mean_{window}h{suffix}"] = rolling.mean()
        result[f"{prefix}_prior_std_{window}h{suffix}"] = rolling.std(ddof=0)
        result[f"{prefix}_prior_min_{window}h{suffix}"] = rolling.min()
        result[f"{prefix}_prior_max_{window}h{suffix}"] = rolling.max()
    return result


def _causal_raw_station_features(
    frame: pd.DataFrame,
    *,
    feature_columns: tuple[str, ...] | list[str] | None = None,
) -> pd.DataFrame:
    requested = (
        set(RAW_CAUSAL_FORECAST_FEATURE_COLUMNS)
        if feature_columns is None
        else set(feature_columns)
    )
    invalid = sorted(requested.difference(RAW_CAUSAL_FORECAST_FEATURE_COLUMNS))
    if invalid:
        raise KeyError(f"unknown causal raw features: {invalid}")
    result: dict[str, pd.Series | np.ndarray] = {}
    if "raw_now_n_raw_records" in requested:
        result["raw_now_n_raw_records"] = pd.to_numeric(
            frame["n_raw_records"], errors="coerce"
        )
    for channel in MEASUREMENT_COLUMNS:
        channel_requested = any(
            column.endswith(f"_{channel}") or column == f"raw_now_{channel}"
            for column in requested
        )
        if not channel_requested:
            continue
        values = pd.to_numeric(frame[channel], errors="coerce")
        now_column = f"raw_now_{channel}"
        if now_column in requested:
            result[now_column] = values
        for lag in FORECAST_HISTORY_LAGS:
            column = f"raw_lag_{lag}h_{channel}"
            if column in requested:
                result[column] = values.shift(int(lag))
        if any(
            f"raw_prior_{statistic}_{window}h_{channel}" in requested
            for window in FORECAST_HISTORY_WINDOWS
            for statistic in FORECAST_HISTORY_STATISTICS
        ):
            prior = values.shift(1)
            for window in FORECAST_HISTORY_WINDOWS:
                rolling = prior.rolling(int(window), min_periods=1)
                summaries = {
                    "mean": rolling.mean,
                    "std": lambda rolling=rolling: rolling.std(ddof=0),
                    "min": rolling.min,
                    "max": rolling.max,
                }
                for statistic, calculate in summaries.items():
                    column = f"raw_prior_{statistic}_{window}h_{channel}"
                    if column in requested:
                        result[column] = calculate()
        for lag in FORECAST_RATE_LAGS:
            column = f"raw_delta_{lag}h_{channel}"
            if column in requested:
                result[column] = values - values.shift(int(lag))
    if {"raw_now_winddir_sin", "raw_now_winddir_cos"}.intersection(requested):
        wind_direction = np.deg2rad(
            pd.to_numeric(frame["winddir_avg_deg"], errors="coerce")
        )
        if "raw_now_winddir_sin" in requested:
            result["raw_now_winddir_sin"] = np.sin(wind_direction)
        if "raw_now_winddir_cos" in requested:
            result["raw_now_winddir_cos"] = np.cos(wind_direction)
    return pd.DataFrame(result, index=frame.index)


def _causal_external_station_features(frame: pd.DataFrame) -> pd.DataFrame:
    result = pd.DataFrame(index=frame.index)
    residuals: dict[str, pd.Series] = {}
    for short, (station_column, reference_column, conversion) in (
        FORECAST_EXTERNAL_CHANNELS.items()
    ):
        station = pd.to_numeric(frame[station_column], errors="coerce") * float(
            conversion
        )
        reference = pd.to_numeric(frame[reference_column], errors="coerce")
        residual = station - reference
        result[f"era5_now_{short}"] = reference
        result[f"external_residual_now_{short}"] = residual
        residuals[short] = residual
        summary = _prior_window_summary(
            residual,
            prefix="external_residual",
            suffix=f"_{short}",
        )
        result = pd.concat([result, summary], axis=1)
        for lag in FORECAST_RATE_LAGS:
            result[f"external_residual_delta_{lag}h_{short}"] = residual - residual.shift(
                int(lag)
            )
    solar_reference = result["era5_now_solar"]
    result["external_clear_sky_ratio_now"] = np.where(
        solar_reference.ge(EXTERNAL_CLEARSKY_REF_MIN),
        pd.to_numeric(frame["solar_radiation_high_wm2"], errors="coerce")
        / solar_reference,
        np.nan,
    )
    return result


def _registry_with_install_dates(registry: pd.DataFrame) -> pd.DataFrame:
    _require_columns(registry, ["station_id", "install_date"])
    result = registry.loc[:, [
        column
        for column in ["station_id", "latitude", "longitude", "elevation", "install_date"]
        if column in registry.columns
    ]].copy()
    result["station_id"] = result["station_id"].astype(str)
    if result.duplicated("station_id").any():
        raise ValueError("station registry contains duplicate station IDs")
    for column in ["latitude", "longitude"]:
        if column not in result.columns:
            result[column] = np.nan
        result[column] = pd.to_numeric(result[column], errors="coerce")
    result["install_date"] = pd.to_datetime(
        result["install_date"], utc=True, errors="coerce"
    ).dt.normalize()
    return result


def _as_of_neighbor_pairs(registry: pd.DataFrame) -> pd.DataFrame:
    context = _registry_with_install_dates(registry)
    rows: list[dict[str, object]] = []
    for origin in context.itertuples(index=False):
        if not np.isfinite(origin.latitude) or not np.isfinite(origin.longitude):
            continue
        for neighbor in context.itertuples(index=False):
            if origin.station_id == neighbor.station_id:
                continue
            if not np.isfinite(neighbor.latitude) or not np.isfinite(neighbor.longitude):
                continue
            latitude_1 = np.radians(float(origin.latitude))
            latitude_2 = np.radians(float(neighbor.latitude))
            delta_latitude = latitude_2 - latitude_1
            delta_longitude = np.radians(float(neighbor.longitude - origin.longitude))
            a = (
                np.sin(delta_latitude / 2.0) ** 2
                + np.cos(latitude_1)
                * np.cos(latitude_2)
                * np.sin(delta_longitude / 2.0) ** 2
            )
            distance_km = 6371.0 * 2.0 * np.arcsin(np.sqrt(a))
            if distance_km <= float(SPATIAL_NEIGHBOR_RADIUS_KM):
                rows.append(
                    {
                        "station_id": str(origin.station_id),
                        "neighbor_id": str(neighbor.station_id),
                        "neighbor_install_date": neighbor.install_date,
                    }
                )
    return pd.DataFrame(
        rows,
        columns=["station_id", "neighbor_id", "neighbor_install_date"],
    )


def _causal_spatial_now_features(
    frame: pd.DataFrame,
    registry: pd.DataFrame,
) -> pd.DataFrame:
    keys = frame.loc[:, ["station_id", "hour_utc"]].copy()
    pairs = _as_of_neighbor_pairs(registry)
    result = keys.copy()
    if pairs.empty:
        for short in FORECAST_SPATIAL_CHANNELS:
            result[f"spatial_residual_now_{short}"] = np.nan
            result[f"spatial_neighbor_count_now_{short}"] = 0.0
        return result
    targets = keys.merge(pairs, on="station_id", how="left")
    targets = targets.loc[
        targets["neighbor_id"].notna()
        & targets["neighbor_install_date"].notna()
        & targets["hour_utc"].ge(targets["neighbor_install_date"])
    ].copy()
    neighbor_columns = list(FORECAST_SPATIAL_CHANNELS.values())
    neighbor_values = frame.loc[:, ["station_id", "hour_utc", *neighbor_columns]].rename(
        columns={
            "station_id": "neighbor_id",
            **{column: f"neighbor_{column}" for column in neighbor_columns},
        }
    )
    targets = targets.merge(
        neighbor_values,
        on=["neighbor_id", "hour_utc"],
        how="left",
        validate="many_to_one",
    )
    aggregates = targets.groupby(["station_id", "hour_utc"], as_index=False).agg(
        **{
            f"neighbor_median_{short}": (f"neighbor_{column}", "median")
            for short, column in FORECAST_SPATIAL_CHANNELS.items()
        },
        **{
            f"spatial_neighbor_count_now_{short}": (f"neighbor_{column}", "count")
            for short, column in FORECAST_SPATIAL_CHANNELS.items()
        },
    )
    result = result.merge(
        aggregates,
        on=["station_id", "hour_utc"],
        how="left",
        validate="one_to_one",
    )
    for short, column in FORECAST_SPATIAL_CHANNELS.items():
        count_column = f"spatial_neighbor_count_now_{short}"
        result[count_column] = pd.to_numeric(
            result[count_column], errors="coerce"
        ).fillna(0.0)
        own = pd.to_numeric(frame[column], errors="coerce")
        median = pd.to_numeric(result[f"neighbor_median_{short}"], errors="coerce")
        result[f"spatial_residual_now_{short}"] = (own - median).where(
            result[count_column].ge(int(SPATIAL_MIN_NEIGHBORS_PRESENT))
        )
    return result.drop(
        columns=[f"neighbor_median_{short}" for short in FORECAST_SPATIAL_CHANNELS]
    )


def _causal_spatial_features(
    frame: pd.DataFrame,
    registry: pd.DataFrame,
) -> pd.DataFrame:
    now = _causal_spatial_now_features(frame, registry)
    now.index = frame.index
    result = now.copy()
    for _, station in result.groupby(frame["station_id"], sort=False):
        for short in FORECAST_SPATIAL_CHANNELS:
            residual_column = f"spatial_residual_now_{short}"
            residual = pd.to_numeric(station[residual_column], errors="coerce")
            summary = _prior_window_summary(
                residual,
                prefix="spatial_residual",
                suffix=f"_{short}",
            )
            result.loc[station.index, summary.columns] = summary
            for lag in FORECAST_RATE_LAGS:
                result.loc[station.index, f"spatial_residual_delta_{lag}h_{short}"] = (
                    residual - residual.shift(int(lag))
                )
    return result


def _causal_detector_station_features(
    frame: pd.DataFrame,
    *,
    feature_columns: tuple[str, ...] | list[str] | None = None,
) -> pd.DataFrame:
    requested = (
        set(DETECTOR_CAUSAL_FORECAST_FEATURE_COLUMNS)
        if feature_columns is None
        else set(feature_columns)
    )
    invalid = sorted(requested.difference(DETECTOR_CAUSAL_FORECAST_FEATURE_COLUMNS))
    if invalid:
        raise KeyError(f"unknown causal detector features: {invalid}")
    result = pd.DataFrame(index=frame.index)
    flags = {
        kind: {
            group: pd.Series(False, index=frame.index, dtype=bool)
            for group in SENSOR_GROUP_ORDER
        }
        for kind in FORECAST_DETECTOR_KINDS
    }
    for channel in MEASUREMENT_COLUMNS:
        group = sensor_group_for_channel(channel)
        if group not in SENSOR_GROUP_ORDER:
            continue
        values = pd.to_numeric(frame[channel], errors="coerce")
        flags["physical"][group] = (
            flags["physical"][group] | physical_limit_flags(values, channel)
        )
        variance = values.rolling(
            FORECAST_STUCK_WINDOW_HOURS,
            min_periods=FORECAST_STUCK_WINDOW_HOURS,
        ).var(ddof=0)
        observed = values.rolling(
            FORECAST_STUCK_WINDOW_HOURS,
            min_periods=FORECAST_STUCK_WINDOW_HOURS,
        ).count()
        nonzero = values.rolling(
            FORECAST_STUCK_WINDOW_HOURS,
            min_periods=FORECAST_STUCK_WINDOW_HOURS,
        ).max().abs().gt(FORECAST_STUCK_VARIANCE_THRESHOLD)
        stuck = (
            variance.le(FORECAST_STUCK_VARIANCE_THRESHOLD)
            & observed.ge(FORECAST_STUCK_WINDOW_HOURS)
            & nonzero
        ).fillna(False)
        flags["stuck"][group] = flags["stuck"][group] | stuck
        prior = values.shift(1)
        prior_mean = prior.rolling(
            FORECAST_DEVIATION_WINDOW_HOURS,
            min_periods=FORECAST_DEVIATION_MIN_PERIODS,
        ).mean()
        prior_std = prior.rolling(
            FORECAST_DEVIATION_WINDOW_HOURS,
            min_periods=FORECAST_DEVIATION_MIN_PERIODS,
        ).std(ddof=0)
        deviation = (
            (values - prior_mean).abs().ge(
                FORECAST_DEVIATION_Z_THRESHOLD * prior_std.where(prior_std.gt(0.0))
            )
            & prior_std.notna()
        ).fillna(False)
        flags["deviation"][group] = flags["deviation"][group] | deviation
    for kind in FORECAST_DETECTOR_KINDS:
        kind_flags = []
        for group in SENSOR_GROUP_ORDER:
            flag = flags[kind][group].astype(float)
            now_column = f"causal_detector_{kind}_now_{group}"
            if now_column in requested:
                result[now_column] = flag
            kind_flags.append(flag)
            for window in FORECAST_DETECTOR_WINDOWS:
                column = f"causal_detector_{kind}_count_{window}h_{group}"
                if column in requested:
                    result[column] = flag.rolling(int(window), min_periods=1).sum()
        any_now_column = f"causal_detector_{kind}_any_now"
        any_count_columns = [
            f"causal_detector_{kind}_any_count_{window}h"
            for window in FORECAST_DETECTOR_WINDOWS
        ]
        if any_now_column in requested or set(any_count_columns).intersection(requested):
            any_flag = pd.concat(kind_flags, axis=1).max(axis=1)
            if any_now_column in requested:
                result[any_now_column] = any_flag
            for window, column in zip(
                FORECAST_DETECTOR_WINDOWS,
                any_count_columns,
                strict=True,
            ):
                if column in requested:
                    result[column] = any_flag.rolling(int(window), min_periods=1).sum()
    return result


def build_causal_detector_evidence(frame: pd.DataFrame) -> pd.DataFrame:
    return _causal_detector_station_features(frame)


def build_causal_external_residuals(frame: pd.DataFrame) -> pd.DataFrame:
    return _causal_external_station_features(frame)


def _station_context_from_registry(
    registry: pd.DataFrame,
    station_hours: pd.DataFrame,
) -> pd.DataFrame:
    _require_columns(registry, ["station_id", "install_date"])
    _require_columns(station_hours, ["station_id", "hour_utc"])
    context = registry.loc[:, [
        column
        for column in [
            "station_id",
            "latitude",
            "longitude",
            "elevation",
            "install_date",
        ]
        if column in registry.columns
    ]].copy()
    context["station_id"] = context["station_id"].astype(str)
    if context.duplicated("station_id").any():
        raise ValueError("station registry contains duplicate station IDs")
    context = context.reset_index(drop=True)
    for column in ["latitude", "longitude", "elevation"]:
        if column not in context.columns:
            context[column] = np.nan
        context[column] = pd.to_numeric(context[column], errors="coerce")
    context["install_date"] = pd.to_datetime(
        context["install_date"], utc=True, errors="coerce"
    ).dt.normalize()

    latitude = np.deg2rad(context["latitude"].to_numpy(dtype=float))
    longitude = np.deg2rad(context["longitude"].to_numpy(dtype=float))
    valid = np.isfinite(latitude) & np.isfinite(longitude)
    context["_neighbor_index"] = list(range(len(context)))
    base = _normalise_forecast_keys(station_hours, "forecast station-hour keys")
    result = base.merge(
        context.loc[:, ["station_id", "elevation", "install_date", "_neighbor_index"]],
        on="station_id",
        how="left",
        validate="many_to_one",
    )
    result["ctx_elevation"] = pd.to_numeric(result["elevation"], errors="coerce")
    result["ctx_n_neighbors"] = np.nan
    install_dates = context["install_date"].to_numpy(dtype="datetime64[ns]")
    for station_id, station_rows in result.groupby("station_id", sort=False):
        station_index = context.index[context["station_id"].eq(station_id)]
        if len(station_index) != 1:
            continue
        origin = int(station_index[0])
        if not valid[origin]:
            continue
        delta_latitude = latitude - latitude[origin]
        delta_longitude = longitude - longitude[origin]
        a = (
            np.sin(delta_latitude / 2.0) ** 2
            + np.cos(latitude[origin])
            * np.cos(latitude)
            * np.sin(delta_longitude / 2.0) ** 2
        )
        distance_km = 6371.0 * 2.0 * np.arcsin(np.sqrt(a))
        neighbor_installs = install_dates[
            (distance_km <= 200.0)
            & (distance_km > 0.0)
            & valid
            & ~pd.isna(install_dates)
        ]
        hour_days = station_rows["hour_utc"].dt.normalize().to_numpy(
            dtype="datetime64[ns]"
        )
        if len(neighbor_installs):
            result.loc[station_rows.index, "ctx_n_neighbors"] = (
                hour_days[:, None] >= neighbor_installs[None, :]
            ).sum(axis=1)
        else:
            result.loc[station_rows.index, "ctx_n_neighbors"] = 0
    return result.loc[:, ["station_id", "hour_utc", "ctx_elevation", "ctx_n_neighbors"]]


def _forecast_feature_reason(feature: str) -> tuple[str, str, str]:
    if feature in BASE_CAUSAL_FORECAST_FEATURE_COLUMNS:
        if feature.startswith("past_sensor_group_absent"):
            return (
                "availability classification",
                "strictly_prior",
                "Prior sensor-group absence is shifted one clock hour before rolling.",
            )
        if feature.startswith("past_physical_limit"):
            return (
                "canonical raw measurements plus fixed hard limits",
                "strictly_prior",
                "Hard physical-limit flags are recomputed from raw rows and shifted before rolling.",
            )
        if feature == "ctx_elevation":
            return (
                "fixed station geography",
                "static",
                "Station elevation is static station metadata available when that station is deployed.",
            )
        if feature == "ctx_n_neighbors":
            return (
                "station geography and installation dates",
                "as_of_scored_hour",
                "The 200 km neighbor count includes only stations installed by the scored hour.",
            )
        if feature.startswith("hour_of_day") or feature.startswith("day_of_year"):
            return (
                "UTC calendar",
                "at_scored_hour",
                "Calendar encodings are known at the scored hour without reading observations.",
            )
        if "n_raw_records" in feature:
            return (
                "canonical raw ingestion counts",
                "strictly_prior",
                "Raw record density is shifted one clock hour before rolling.",
            )
        return (
            "operational availability classification",
            "strictly_prior",
            "Availability history is shifted one clock hour before any recency or rolling calculation.",
        )
    if feature in RAW_CAUSAL_FORECAST_FEATURE_COLUMNS:
        if feature.startswith("raw_now"):
            return (
                "canonical raw station-hour measurements",
                "at_scored_hour",
                "The measurement has timestamp t and is available when the forecast is scored.",
            )
        if feature.startswith("raw_lag"):
            return (
                "canonical raw station-hour measurements",
                "strictly_prior",
                "The lagged value comes from a clock hour before the scored hour.",
            )
        if feature.startswith("raw_prior"):
            return (
                "canonical raw station-hour measurements",
                "strictly_prior_baseline",
                "The rolling baseline is shifted one clock hour before its window is calculated.",
            )
        return (
            "canonical raw station-hour measurements",
            "at_or_before_scored_hour",
            "The rate compares the current measurement with a strictly earlier measurement.",
        )
    if feature in EXTERNAL_CAUSAL_FORECAST_FEATURE_COLUMNS:
        if feature.startswith("era5_now"):
            return (
                "exact-hour ERA5/Open-Meteo reference cache",
                "at_scored_hour",
                "The reference value is joined at the same UTC hour as the scored observation.",
            )
        if feature.startswith("external_residual_now") or feature == "external_clear_sky_ratio_now":
            return (
                "canonical raw measurement plus exact-hour ERA5/Open-Meteo reference",
                "at_scored_hour",
                "The residual is rebuilt from same-hour hourly source data without a five-minute forward lookup.",
            )
        if feature.startswith("external_residual_prior"):
            return (
                "canonical raw measurement plus exact-hour ERA5/Open-Meteo reference",
                "strictly_prior_baseline",
                "The residual baseline is shifted one clock hour before the rolling calculation.",
            )
        return (
            "canonical raw measurement plus exact-hour ERA5/Open-Meteo reference",
            "at_or_before_scored_hour",
            "The residual rate compares the current residual with an earlier residual.",
        )
    if feature in SPATIAL_CAUSAL_FORECAST_FEATURE_COLUMNS:
        if feature.startswith("spatial_residual_now") or feature.startswith(
            "spatial_neighbor_count_now"
        ):
            return (
                "same-hour raw neighbour measurements and as-of-time topology",
                "at_scored_hour",
                "Only geographically eligible neighbours installed by the scored hour contribute at that hour.",
            )
        if feature.startswith("spatial_residual_prior"):
            return (
                "same-hour raw neighbour measurements and as-of-time topology",
                "strictly_prior_baseline",
                "The spatial residual baseline is shifted one clock hour before its rolling calculation.",
            )
        return (
            "same-hour raw neighbour measurements and as-of-time topology",
            "at_or_before_scored_hour",
            "The spatial residual rate compares the current residual with an earlier residual.",
        )
    if feature in DETECTOR_CAUSAL_FORECAST_FEATURE_COLUMNS:
        if "_now_" in feature or feature.endswith("_any_now"):
            return (
                "causally reconstructed detector evidence",
                "at_or_before_scored_hour",
                "The detector is computed from the current completed value and observations no later than the scored hour.",
            )
        return (
            "causally reconstructed detector evidence",
            "at_or_before_scored_hour",
            "The count aggregates only detector states computed at or before the scored hour.",
        )
    if feature.startswith("past_sensor_group_absent"):
        return (
            "availability classification",
            "strictly_prior",
            "Prior sensor-group absence is shifted one clock hour before rolling.",
        )
    if feature.startswith("past_physical_limit"):
        return (
            "canonical raw measurements plus fixed hard limits",
            "strictly_prior",
            "Hard physical-limit flags are recomputed from raw rows and shifted before rolling.",
        )
    if feature == "ctx_elevation":
        return (
            "fixed station geography",
            "static",
            "Station elevation is static station metadata available when that station is deployed.",
        )
    if feature == "ctx_n_neighbors":
        return (
            "station geography and installation dates",
            "as_of_scored_hour",
            "The 200 km neighbor count includes only stations installed by the scored hour.",
        )
    if feature.startswith("hour_of_day") or feature.startswith("day_of_year"):
        return (
            "UTC calendar",
            "at_scored_hour",
            "Calendar encodings are known at the scored hour without reading observations.",
        )
    if "n_raw_records" in feature:
        return (
            "canonical raw ingestion counts",
            "strictly_prior",
            "Raw record density is shifted one clock hour before rolling.",
        )
    return (
        "operational availability classification",
        "strictly_prior",
        "Availability history is shifted one clock hour before any recency or rolling calculation.",
    )


def _matrix_exclusion_reason(feature: str) -> str:
    if feature == "rel_ratio_solar":
        return "Whole-day station and fleet medians expose later daylight observations."
    if feature.startswith(("stat_", "z_", "offset_level_")):
        return "Stored statistical evidence uses full-series baselines, global thresholds, or retroactively assigned detector windows."
    if feature.startswith(("r_", "ext_", "clear_sky_")):
        return "The stored external artifact can use a forward five-minute source or an unshifted baseline; the corrected route rebuilds direct residuals from exact-hour inputs."
    if feature.startswith(("spatial_", "n_neighbors_present_")):
        return "The stored spatial artifact combines an unshifted baseline or post-hoc neighbour policy; the corrected route recomputes same-hour peer evidence with as-of-time topology."
    if feature.startswith("ctx_"):
        return "The fixed context is rebuilt directly from the registry instead of inheriting a mixed historical feature matrix."
    return "This stored detector feature is not independently certified as strictly as-of-time."


def build_forecast_feature_causality_audit(
    feature_matrix_columns: list[str] | tuple[str, ...],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for feature in CAUSAL_FORECAST_FEATURE_COLUMNS:
        source, time_contract, reason = _forecast_feature_reason(feature)
        rows.append(
            {
                "feature": feature,
                "source": source,
                "causality": "causal",
                "included": True,
                "time_contract": time_contract,
                "scope_change": (
                    "retained_from_33_feature_scope"
                    if feature in BASE_CAUSAL_FORECAST_FEATURE_COLUMNS
                    else "reinstated_with_as_of_time_reconstruction"
                ),
                "reason": reason,
            }
        )

    for feature in FEATURE_COLUMNS:
        rows.append(
            {
                "feature": feature,
                "source": "legacy outage-risk feature path",
                "causality": "excluded_noncausal",
                "included": False,
                "time_contract": "includes_scored_hour_or_network_snapshot",
                "scope_change": "excluded",
                "reason": "The causal route rebuilds this availability concept with an explicit one-hour lag.",
            }
        )
    for feature in sorted(
        {
            str(column)
            for column in feature_matrix_columns
            if str(column) not in {"station_id", "hour", "hour_utc", "time_utc"}
        }
    ):
        rows.append(
            {
                "feature": f"historical_feature_matrix::{feature}",
                "source": "historical detection feature matrix",
                "causality": "excluded_noncausal",
                "included": False,
                "time_contract": "not_as_of_time_certified",
                "scope_change": "excluded",
                "reason": _matrix_exclusion_reason(feature),
            }
        )
    for feature in [
        "current_availability_class",
        "current_absent_sensor_groups",
        "current_is_outage",
        "current_is_transmitting",
        "current_data_present",
        "current_qc_status",
        "current_row_state",
        "current_source_kind",
        "current_epoch",
        "current_timestamp_utc_dt",
        "current_timestamp_utc",
        "current_timestamp_local",
        "current_latitude",
        "current_longitude",
    ]:
        rows.append(
            {
                "feature": feature,
                "source": "current operational source row",
                "causality": "excluded_noncausal",
                "included": False,
                "time_contract": "scored_hour_not_used",
                "scope_change": "excluded",
                "reason": "This current operational state is a retrospective availability artifact rather than a raw timestamped observation.",
            }
        )
    for feature in [
        "fault_hour",
        "fault_target",
        "fault_scoreable",
        "fault_excluded",
        "display_state",
        "mechanisms",
        "components",
        "source_episode_ids",
        "detectors_fired",
        "training_eligible",
        "episode_id",
        "label_state",
        "binary_fault",
        "availability_events",
        "partial_outage_events",
        "network_outage_windows",
        "station_reliability_summary",
        "eligible_6h",
        "eligible_12h",
        "eligible_24h",
        "y_6h",
        "y_12h",
        "y_24h",
        "label_end_6h",
        "label_end_12h",
        "label_end_24h",
        "future_excluded_6h",
        "future_excluded_12h",
        "future_excluded_24h",
    ]:
        rows.append(
            {
                "feature": feature,
                "source": "label or completed-event artifact",
                "causality": "excluded_noncausal",
                "included": False,
                "time_contract": "target_or_future_derived",
                "scope_change": "excluded",
                "reason": "Ground-truth labels, completed events, and whole-span summaries are not model inputs.",
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["included", "feature"], ascending=[False, True], kind="mergesort"
    ).reset_index(drop=True)


def _availability_lookup(frame: pd.DataFrame) -> pd.DataFrame:
    source = _normalise_forecast_keys(frame, "availability classification")
    result = source.loc[:, ["station_id", "hour_utc"]].copy()
    result["availability_class"] = (
        source["availability_class"].astype("string")
        if "availability_class" in source.columns
        else pd.Series(pd.NA, index=source.index, dtype="string")
    )
    result["absent_sensor_groups"] = (
        source["absent_sensor_groups"].fillna("").astype(str)
        if "absent_sensor_groups" in source.columns
        else ""
    )
    return result


def _observation_lookup(frame: pd.DataFrame) -> pd.DataFrame:
    source = _normalise_forecast_keys(frame, "canonical observations")
    result = source.loc[:, ["station_id", "hour_utc"]].copy()
    for column in ["n_raw_records", *MEASUREMENT_COLUMNS]:
        result[column] = (
            pd.to_numeric(source[column], errors="coerce")
            if column in source.columns
            else np.nan
        )
    return result


def _strictly_prior_station_features(frame: pd.DataFrame) -> pd.DataFrame:
    result = pd.DataFrame(index=frame.index)
    full_outage = frame["is_outage"].astype(bool)
    partial_outage = frame["_partial_outage"].astype(bool)
    transmitting = frame["is_transmitting"].astype(bool)
    outage_start = full_outage & ~full_outage.shift(1, fill_value=False)

    prior_full_outage = full_outage.astype(float).shift(1)
    for window in [6, 24, 72]:
        result[f"past_full_outage_frac_{window}h"] = prior_full_outage.rolling(
            window, min_periods=1
        ).mean()
    result["past_full_outage_start_count_72h"] = outage_start.astype(float).shift(1).rolling(
        72, min_periods=1
    ).sum()
    previous_outage_hour = frame["hour_utc"].where(full_outage).shift(1).ffill()
    hours_since_outage = (
        (frame["hour_utc"] - previous_outage_hour) / pd.Timedelta(hours=1)
    )
    result["hours_since_last_full_outage_log"] = np.log1p(
        hours_since_outage.fillna(720.0).clip(upper=720.0).astype(float)
    )
    transmission_run_id = (~transmitting).cumsum()
    transmission_run_hours = (
        transmitting.astype(int).groupby(transmission_run_id).cumsum().where(transmitting, 0)
    )
    result["past_transmitting_run_hours_log"] = np.log1p(
        transmission_run_hours.shift(1).fillna(0.0).astype(float)
    )
    prior_partial_outage = partial_outage.astype(float).shift(1)
    for window in [6, 24]:
        result[f"past_partial_outage_frac_{window}h"] = prior_partial_outage.rolling(
            window, min_periods=1
        ).mean()
    prior_raw_records = pd.to_numeric(frame["n_raw_records"], errors="coerce").shift(1)
    for window in [6, 24]:
        result[f"past_n_raw_records_mean_{window}h"] = prior_raw_records.rolling(
            window, min_periods=1
        ).mean()
    for group in SENSOR_GROUP_ORDER:
        prior_absent = frame[f"_sensor_group_absent_{group}"].astype(float).shift(1)
        result[f"past_sensor_group_absent_frac_24h_{group}"] = prior_absent.rolling(
            24, min_periods=1
        ).mean()
    prior_physical_any = frame["_physical_limit_any"].astype(float).shift(1)
    for window in [6, 24]:
        result[f"past_physical_limit_any_frac_{window}h"] = prior_physical_any.rolling(
            window, min_periods=1
        ).mean()
    for group in SENSOR_GROUP_ORDER:
        prior_physical = frame[f"_physical_limit_{group}"].astype(float).shift(1)
        result[f"past_physical_limit_frac_24h_{group}"] = prior_physical.rolling(
            24, min_periods=1
        ).mean()
    return result


def _strictly_prior_network_features(frame: pd.DataFrame) -> pd.DataFrame:
    network = (
        frame.groupby("hour_utc", as_index=False)
        .agg(
            _network_full_outage=("is_outage", "mean"),
            _network_full_outage_start=("_full_outage_start", "mean"),
            _network_partial_outage=("_partial_outage", "mean"),
        )
        .sort_values("hour_utc", kind="mergesort")
    )
    network["past_network_full_outage_frac_1h"] = network[
        "_network_full_outage"
    ].shift(1)
    network["past_network_full_outage_start_frac_24h"] = network[
        "_network_full_outage_start"
    ].shift(1).rolling(24, min_periods=1).mean()
    network["past_network_partial_outage_frac_1h"] = network[
        "_network_partial_outage"
    ].shift(1)
    return network.loc[:, [
        "hour_utc",
        "past_network_full_outage_frac_1h",
        "past_network_full_outage_start_frac_24h",
        "past_network_partial_outage_frac_1h",
    ]]


def build_causal_forecast_features(
    grid: pd.DataFrame,
    *,
    availability: pd.DataFrame | None = None,
    observations: pd.DataFrame | None = None,
    registry: pd.DataFrame | None = None,
    reference: pd.DataFrame | None = None,
    feature_matrix_columns: list[str] | tuple[str, ...] = (),
    feature_columns: tuple[str, ...] | list[str] | None = None,
) -> CausalForecastFeatureBundle:
    selected_columns = (
        tuple(CAUSAL_FORECAST_FEATURE_COLUMNS)
        if feature_columns is None
        else tuple(dict.fromkeys(str(column) for column in feature_columns))
    )
    invalid = sorted(set(selected_columns).difference(CAUSAL_FORECAST_FEATURE_COLUMNS))
    if invalid:
        raise KeyError(f"unknown causal forecast features: {invalid}")
    if not selected_columns:
        raise ValueError("causal forecast feature selection must not be empty")
    selected = set(selected_columns)
    selected_base = selected.intersection(BASE_CAUSAL_FORECAST_FEATURE_COLUMNS)
    selected_raw = selected.intersection(RAW_CAUSAL_FORECAST_FEATURE_COLUMNS)
    selected_external = selected.intersection(EXTERNAL_CAUSAL_FORECAST_FEATURE_COLUMNS)
    selected_spatial = selected.intersection(SPATIAL_CAUSAL_FORECAST_FEATURE_COLUMNS)
    selected_detector = selected.intersection(DETECTOR_CAUSAL_FORECAST_FEATURE_COLUMNS)
    _require_columns(grid, ["station_id", "hour_utc", "is_outage", "is_transmitting"])
    base = _normalise_forecast_keys(grid, "continuous risk grid")
    availability_frame = (
        load_operational_availability() if availability is None else availability
    )
    observation_frame = (
        _default_forecast_observations() if observations is None else observations
    )
    registry_frame = _default_forecast_registry() if registry is None else registry
    reference_frame = (
        (_default_forecast_reference() if reference is None else reference)
        if selected_external
        else None
    )
    availability_lookup = _availability_lookup(availability_frame)
    observation_lookup = _observation_lookup(observation_frame)
    reference_lookup = (
        _reference_lookup(reference_frame)
        if reference_frame is not None
        else base.loc[:, ["station_id", "hour_utc"]].copy()
    )
    base = base.merge(
        availability_lookup,
        on=["station_id", "hour_utc"],
        how="left",
        validate="one_to_one",
    ).merge(
        observation_lookup,
        on=["station_id", "hour_utc"],
        how="left",
        validate="one_to_one",
    ).merge(
        reference_lookup,
        on=["station_id", "hour_utc"],
        how="left",
        validate="one_to_one",
    )
    base["is_outage"] = base["is_outage"].astype("boolean").fillna(False).astype(bool)
    base["is_transmitting"] = (
        base["is_transmitting"].astype("boolean").fillna(False).astype(bool)
    )
    base["_partial_outage"] = base["availability_class"].astype("string").eq(
        AVAILABILITY_CLASS_PARTIAL_OUTAGE
    ).fillna(False)
    base["_full_outage_start"] = base["is_outage"] & ~base.groupby("station_id")[
        "is_outage"
    ].shift(1, fill_value=False)
    absent_tokens = base["absent_sensor_groups"].fillna("").astype(str).map(
        lambda value: set(token for token in value.split("|") if token)
    )
    for group in SENSOR_GROUP_ORDER:
        base[f"_sensor_group_absent_{group}"] = (
            base["is_outage"] | absent_tokens.map(lambda values, group=group: group in values)
        )
        base[f"_physical_limit_{group}"] = False
    for channel in MEASUREMENT_COLUMNS:
        if channel not in base.columns:
            continue
        group = sensor_group_for_channel(channel)
        if group not in SENSOR_GROUP_ORDER:
            continue
        base[f"_physical_limit_{group}"] = (
            base[f"_physical_limit_{group}"].astype(bool)
            | physical_limit_flags(base[channel], channel).to_numpy(dtype=bool)
        )
    base["_physical_limit_any"] = base.loc[:, [
        f"_physical_limit_{group}" for group in SENSOR_GROUP_ORDER
    ]].any(axis=1)

    features = base.loc[:, ["station_id", "hour_utc"]].copy()
    station_feature_frames = []
    for _, station_frame in base.groupby("station_id", sort=False):
        frames: list[pd.DataFrame] = []
        if selected_base:
            station_features = _strictly_prior_station_features(station_frame)
            selected_station_features = [
                column for column in station_features.columns if column in selected
            ]
            if selected_station_features:
                frames.append(station_features.loc[:, selected_station_features])
        if selected_raw:
            frames.append(
                _causal_raw_station_features(
                    station_frame,
                    feature_columns=tuple(selected_raw),
                )
            )
        if selected_external:
            external_features = _causal_external_station_features(station_frame)
            frames.append(
                external_features.loc[:, [
                    column for column in external_features.columns if column in selected
                ]]
            )
        if selected_detector:
            frames.append(
                _causal_detector_station_features(
                    station_frame,
                    feature_columns=tuple(selected_detector),
                )
            )
        station_feature_frames.append(
            pd.concat(frames, axis=1) if frames else pd.DataFrame(index=station_frame.index)
        )
    station_features = pd.concat(station_feature_frames).sort_index()
    features = pd.concat([features, station_features], axis=1)
    if selected_spatial:
        spatial_features = _causal_spatial_features(base, registry_frame)
        features = pd.concat(
            [
                features,
                spatial_features.loc[:, [
                    column
                    for column in SPATIAL_CAUSAL_FORECAST_FEATURE_COLUMNS
                    if column in selected
                ]],
            ],
            axis=1,
        )
    network_columns = {
        "past_network_full_outage_frac_1h",
        "past_network_full_outage_start_frac_24h",
        "past_network_partial_outage_frac_1h",
    }
    if selected.intersection(network_columns):
        features = features.merge(
            _strictly_prior_network_features(base).loc[:, [
                "hour_utc",
                *sorted(selected.intersection(network_columns)),
            ]],
            on="hour_utc",
            how="left",
            validate="many_to_one",
        )
    context_columns = {"ctx_elevation", "ctx_n_neighbors"}
    if selected.intersection(context_columns):
        context = _station_context_from_registry(
            registry_frame,
            features.loc[:, ["station_id", "hour_utc"]],
        )
        features = features.merge(
            context.loc[:, [
                "station_id",
                "hour_utc",
                *sorted(selected.intersection(context_columns)),
            ]],
            on=["station_id", "hour_utc"],
            how="left",
            validate="one_to_one",
        )
    if selected.intersection({"hour_of_day_sin", "hour_of_day_cos"}):
        hour = features["hour_utc"].dt.hour.astype(float)
        if "hour_of_day_sin" in selected:
            features["hour_of_day_sin"] = np.sin(2.0 * np.pi * hour / 24.0)
        if "hour_of_day_cos" in selected:
            features["hour_of_day_cos"] = np.cos(2.0 * np.pi * hour / 24.0)
    if selected.intersection({"day_of_year_sin", "day_of_year_cos"}):
        day = features["hour_utc"].dt.dayofyear.astype(float)
        if "day_of_year_sin" in selected:
            features["day_of_year_sin"] = np.sin(2.0 * np.pi * day / 365.25)
        if "day_of_year_cos" in selected:
            features["day_of_year_cos"] = np.cos(2.0 * np.pi * day / 365.25)
    missing = sorted(set(selected_columns).difference(features.columns))
    if missing:
        raise KeyError(f"causal forecast features missing: {missing}")
    features.loc[:, list(selected_columns)] = features.loc[
        :, list(selected_columns)
    ].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    audit = build_forecast_feature_causality_audit(feature_matrix_columns)
    if feature_columns is not None:
        audit = audit.loc[
            audit["included"].astype(bool) & audit["feature"].isin(selected_columns)
        ].copy()
    audited_features = set(audit.loc[audit["included"], "feature"])
    if set(selected_columns) != audited_features:
        raise RuntimeError("causal feature audit does not match the model feature set")
    if not audit.loc[audit["included"], "causality"].eq("causal").all():
        raise RuntimeError("non-causal feature entered the forecast feature set")
    if feature_columns is None and len(selected_columns) != CAUSAL_FORECAST_FEATURE_COUNT:
        raise RuntimeError("unexpected causal forecast feature count")
    return CausalForecastFeatureBundle(
        frame=features.sort_values(["station_id", "hour_utc"], kind="mergesort").reset_index(
            drop=True
        ),
        feature_columns=selected_columns,
        causality_audit=audit,
    )


def incident_hazard_numeric_feature_columns() -> tuple[str, ...]:
    columns: list[str] = [
        *BASE_CAUSAL_FORECAST_FEATURE_COLUMNS,
        "raw_now_n_raw_records",
    ]
    for channel in INCIDENT_HAZARD_RAW_CHANNELS:
        columns.append(f"raw_now_{channel}")
        columns.extend(
            f"raw_lag_{lag}h_{channel}" for lag in INCIDENT_HAZARD_HISTORY_LAGS
        )
        for window in INCIDENT_HAZARD_HISTORY_WINDOWS:
            columns.extend(
                [
                    f"raw_prior_mean_{window}h_{channel}",
                    f"raw_prior_std_{window}h_{channel}",
                    f"incident_z_{window}h_{channel}",
                    f"incident_abs_z_ewma_{window}h_{channel}",
                ]
            )
        columns.extend(
            f"raw_delta_{lag}h_{channel}" for lag in FORECAST_RATE_LAGS
        )
    for short in FORECAST_EXTERNAL_CHANNELS:
        columns.extend(
            [f"era5_now_{short}", f"external_residual_now_{short}"]
        )
        for window in INCIDENT_HAZARD_HISTORY_WINDOWS:
            columns.extend(
                [
                    f"external_residual_prior_mean_{window}h_{short}",
                    f"external_residual_prior_std_{window}h_{short}",
                ]
            )
        columns.extend(
            f"external_residual_delta_{lag}h_{short}" for lag in FORECAST_RATE_LAGS
        )
    columns.append("external_clear_sky_ratio_now")
    for short in FORECAST_SPATIAL_CHANNELS:
        columns.extend(
            [
                f"spatial_residual_now_{short}",
                f"spatial_neighbor_count_now_{short}",
            ]
        )
        for window in INCIDENT_HAZARD_HISTORY_WINDOWS:
            columns.extend(
                [
                    f"spatial_residual_prior_mean_{window}h_{short}",
                    f"spatial_residual_prior_std_{window}h_{short}",
                ]
            )
        columns.extend(
            f"spatial_residual_delta_{lag}h_{short}" for lag in FORECAST_RATE_LAGS
        )
    columns.extend(DETECTOR_CAUSAL_FORECAST_FEATURE_COLUMNS)
    columns.append("incident_station_age_hours")
    return tuple(dict.fromkeys(columns))


def _incident_hazard_feature_audit(
    base_audit: pd.DataFrame,
    numeric_columns: tuple[str, ...],
    categorical_columns: tuple[str, ...],
) -> pd.DataFrame:
    included = set(numeric_columns) | set(categorical_columns)
    audit = base_audit.copy(deep=True)
    was_included = audit["included"].astype(bool)
    no_longer_selected = was_included & ~audit["feature"].isin(included)
    audit.loc[:, "included"] = audit["feature"].isin(included)
    audit.loc[no_longer_selected, "causality"] = "available_causal_not_selected"
    audit.loc[no_longer_selected, "scope_change"] = "compact_incident_schema"
    audit.loc[no_longer_selected, "reason"] = (
        "The feature is causal but was not selected for the compact incident-hazard "
        "schema, which is sized for independent incident support rather than dense "
        "station-hour rows."
    )
    derived_rows = [
        {
            "feature": "station_id",
            "source": "station registry identity",
            "causality": "causal",
            "included": True,
            "time_contract": "known fixed-fleet identifier at scored hour",
            "scope_change": "incident_hazard_addition",
            "reason": "Known-station identity permits a partially pooled fleet model to learn stable station reliability differences from training data only.",
        },
        {
            "feature": "incident_station_age_hours",
            "source": "station registry install_date",
            "causality": "causal",
            "included": True,
            "time_contract": "hour_utc minus install_date at or before scored hour",
            "scope_change": "incident_hazard_addition",
            "reason": "Station age is known at the scored hour and captures commissioning cohort context.",
        },
    ]
    for channel in INCIDENT_HAZARD_RAW_CHANNELS:
        for window in INCIDENT_HAZARD_HISTORY_WINDOWS:
            derived_rows.extend(
                [
                    {
                        "feature": f"incident_z_{window}h_{channel}",
                        "source": f"raw current {channel} plus prior {window}h station baseline",
                        "causality": "causal",
                        "included": True,
                        "time_contract": "current reading t minus strictly prior rolling mean and standard deviation",
                        "scope_change": "incident_hazard_addition",
                        "reason": "A station-normalized deviation exposes a causal precursor without using labels or future observations.",
                    },
                    {
                        "feature": f"incident_abs_z_ewma_{window}h_{channel}",
                        "source": f"causal absolute incident_z_{window}h_{channel}",
                        "causality": "causal",
                        "included": True,
                        "time_contract": "trailing exponentially weighted summary ending at t",
                        "scope_change": "incident_hazard_addition",
                        "reason": "A causal trajectory summary captures persistent escalation before an incident onset.",
                    },
                ]
            )
    audit = pd.concat([audit, pd.DataFrame(derived_rows)], ignore_index=True)
    audit = audit.drop_duplicates("feature", keep="last")
    if set(audit.loc[audit["included"], "feature"]) != included:
        raise RuntimeError("incident-hazard feature audit does not match the model schema")
    if not audit.loc[audit["included"], "causality"].eq("causal").all():
        raise RuntimeError("a non-causal feature entered the incident-hazard schema")
    return audit.sort_values(["included", "feature"], ascending=[False, True], kind="mergesort").reset_index(
        drop=True
    )


def build_incident_hazard_features(
    grid: pd.DataFrame,
    *,
    availability: pd.DataFrame | None = None,
    observations: pd.DataFrame | None = None,
    registry: pd.DataFrame | None = None,
    reference: pd.DataFrame | None = None,
    feature_matrix_columns: list[str] | tuple[str, ...] = (),
) -> IncidentHazardFeatureBundle:
    base = build_causal_forecast_features(
        grid,
        availability=availability,
        observations=observations,
        registry=registry,
        reference=reference,
        feature_matrix_columns=feature_matrix_columns,
    )
    registry_frame = _default_forecast_registry() if registry is None else registry
    context = _registry_with_install_dates(registry_frame).loc[:, ["station_id", "install_date"]]
    features = base.frame.loc[:, ["station_id", "hour_utc"]].copy()
    numeric_columns = incident_hazard_numeric_feature_columns()
    derived_columns = {
        "incident_station_age_hours",
        *(
            f"incident_z_{window}h_{channel}"
            for channel in INCIDENT_HAZARD_RAW_CHANNELS
            for window in INCIDENT_HAZARD_HISTORY_WINDOWS
        ),
        *(
            f"incident_abs_z_ewma_{window}h_{channel}"
            for channel in INCIDENT_HAZARD_RAW_CHANNELS
            for window in INCIDENT_HAZARD_HISTORY_WINDOWS
        ),
    }
    missing = sorted(
        set(numeric_columns).difference(base.frame.columns).difference(derived_columns)
    )
    if missing:
        raise KeyError(f"incident-hazard source features missing: {missing}")
    selected = [column for column in numeric_columns if column in base.frame.columns]
    features = features.join(base.frame.loc[:, selected])
    features = features.merge(context, on="station_id", how="left", validate="many_to_one")
    install = pd.to_datetime(features["install_date"], utc=True, errors="coerce")
    age = (features["hour_utc"] - install) / pd.Timedelta(hours=1)
    features["incident_station_age_hours"] = age.where(age.ge(0.0))
    features = features.drop(columns=["install_date"])
    for station_id, station in features.groupby("station_id", sort=False):
        station_index = station.index
        for channel in INCIDENT_HAZARD_RAW_CHANNELS:
            current = pd.to_numeric(station[f"raw_now_{channel}"], errors="coerce")
            for window in INCIDENT_HAZARD_HISTORY_WINDOWS:
                mean = pd.to_numeric(
                    station[f"raw_prior_mean_{window}h_{channel}"], errors="coerce"
                )
                std = pd.to_numeric(
                    station[f"raw_prior_std_{window}h_{channel}"], errors="coerce"
                )
                z = (current - mean) / std.where(std.gt(0.0))
                z = z.clip(lower=-12.0, upper=12.0)
                z_column = f"incident_z_{window}h_{channel}"
                ewma_column = f"incident_abs_z_ewma_{window}h_{channel}"
                features.loc[station_index, z_column] = z.to_numpy(dtype=float)
                features.loc[station_index, ewma_column] = z.abs().ewm(
                    span=int(window), adjust=False, min_periods=1
                ).mean().to_numpy(dtype=float)
    features.loc[:, numeric_columns] = features.loc[:, numeric_columns].apply(
        pd.to_numeric, errors="coerce"
    ).replace([np.inf, -np.inf], np.nan)
    audit = _incident_hazard_feature_audit(
        base.causality_audit,
        numeric_columns,
        ("station_id",),
    )
    return IncidentHazardFeatureBundle(
        frame=features.sort_values(["station_id", "hour_utc"], kind="mergesort").reset_index(
            drop=True
        ),
        numeric_feature_columns=numeric_columns,
        categorical_feature_columns=("station_id",),
        causality_audit=audit,
    )


def discrete_hazard_numeric_feature_columns(target: str) -> tuple[str, ...]:
    target = str(target)
    if target not in {"fault", "outage"}:
        raise ValueError("discrete hazard target must be fault or outage")
    availability = (
        "past_full_outage_frac_24h",
        "hours_since_last_full_outage_log",
        "past_transmitting_run_hours_log",
        "past_partial_outage_frac_24h",
        "past_n_raw_records_mean_24h",
        *(f"past_sensor_group_absent_frac_24h_{group}" for group in SENSOR_GROUP_ORDER),
    )
    channels = (
        "pressure_max_hpa",
        "pressure_trend_hpa",
        "temp_avg_c",
        "windspeed_avg_kmh",
        "windgust_high_kmh",
    )
    return (
        "hour_of_day_sin",
        "hour_of_day_cos",
        "hazard_day_of_week_sin",
        "hazard_day_of_week_cos",
        "hazard_month_sin",
        "hazard_month_cos",
        "ctx_elevation",
        "ctx_n_neighbors",
        "hazard_station_age_hours",
        *availability,
        *(f"hazard_detector_count_24h_{group}" for group in SENSOR_GROUP_ORDER),
        *(f"raw_now_{channel}" for channel in channels),
        *(f"raw_prior_mean_24h_{channel}" for channel in channels),
        *(f"raw_delta_6h_{channel}" for channel in channels),
        f"history_{target}_hours_since_last_event_end",
        f"hazard_log1p_hours_since_last_{target}_event_end",
        f"history_{target}_distinct_event_count_trailing_168h",
        f"history_{target}_distinct_event_count_trailing_720h",
    )


def discrete_hazard_causal_source_columns(target: str) -> tuple[str, ...]:
    target = str(target)
    numeric = discrete_hazard_numeric_feature_columns(target)
    derived = {
        "hazard_day_of_week_sin",
        "hazard_day_of_week_cos",
        "hazard_month_sin",
        "hazard_month_cos",
        "hazard_station_age_hours",
        f"hazard_log1p_hours_since_last_{target}_event_end",
        *(f"hazard_detector_count_24h_{group}" for group in SENSOR_GROUP_ORDER),
    }
    history = tuple(column for column in numeric if column.startswith("history_"))
    causal = tuple(
        column
        for column in numeric
        if column not in derived and column not in set(history)
    )
    detector_sources = tuple(
        f"causal_detector_{kind}_count_24h_{group}"
        for kind in FORECAST_DETECTOR_KINDS
        for group in SENSOR_GROUP_ORDER
    )
    return tuple(dict.fromkeys([*causal, *detector_sources]))


def _discrete_hazard_source_columns(target: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    numeric = discrete_hazard_numeric_feature_columns(target)
    history = tuple(column for column in numeric if column.startswith("history_"))
    return discrete_hazard_causal_source_columns(target), history


def _discrete_hazard_feature_audit(
    *,
    target: str,
    numeric_columns: tuple[str, ...],
    station_columns: tuple[str, ...],
) -> pd.DataFrame:
    target = str(target)
    rows: list[dict[str, object]] = []
    history_prefix = f"history_{target}_"
    for feature in numeric_columns:
        if feature.startswith(history_prefix):
            source = "strictly prior reviewed fault labels" if target == "fault" else "strictly prior observed outage states"
            deployment_status = (
                "optimistic_ground_truth_history_not_available_live"
                if target == "fault"
                else "operationally_observable_after_event"
            )
            time_contract = "strictly before the scored hour"
            causality = "strictly_prior_history"
        elif feature.startswith("hazard_log1p_hours_since_last_"):
            source = f"history_{target}_hours_since_last_event_end"
            deployment_status = (
                "optimistic_ground_truth_history_not_available_live"
                if target == "fault"
                else "operationally_observable_after_event"
            )
            time_contract = "deterministic transform of strictly prior history"
            causality = "strictly_prior_history"
        elif feature.startswith("hazard_detector_count_24h_"):
            group = feature.removeprefix("hazard_detector_count_24h_")
            source = "; ".join(
                f"causal_detector_{kind}_count_24h_{group}"
                for kind in FORECAST_DETECTOR_KINDS
            )
            deployment_status = "causal_measurement_or_static_context"
            time_contract = "sum of causal detector counts ending at the scored hour"
            causality = "causal"
        elif feature in {"hazard_day_of_week_sin", "hazard_day_of_week_cos", "hazard_month_sin", "hazard_month_cos"}:
            source = "hour_utc"
            deployment_status = "causal_measurement_or_static_context"
            time_contract = "calendar value known at the scored hour"
            causality = "causal"
        elif feature == "hazard_station_age_hours":
            source = "station registry install_date"
            deployment_status = "causal_measurement_or_static_context"
            time_contract = "hour_utc minus fixed install_date"
            causality = "causal"
        else:
            source = feature
            deployment_status = "causal_measurement_or_static_context"
            time_contract = "causal source value at or before the scored hour"
            causality = "causal"
        rows.append(
            {
                "feature": feature,
                "source": source,
                "causality": causality,
                "included": True,
                "time_contract": time_contract,
                "scope_change": "discrete_hazard_compact_schema",
                "reason": "Predeclared compact discrete-time incident-hazard feature.",
                "deployment_status": deployment_status,
            }
        )
    for feature in station_columns:
        rows.append(
            {
                "feature": feature,
                "source": "station registry identity",
                "causality": "causal",
                "included": True,
                "time_contract": "known fixed-fleet station identity at the scored hour",
                "scope_change": "discrete_hazard_station_effect",
                "reason": "One-hot station indicator for the regularised fixed-effect approximation.",
                "deployment_status": "causal_measurement_or_static_context",
            }
        )
    return pd.DataFrame(rows).sort_values("feature", kind="mergesort").reset_index(
        drop=True
    )


def build_discrete_hazard_features(
    causal_features: CausalForecastFeatureBundle,
    event_history: pd.DataFrame,
    *,
    target: str,
    registry: pd.DataFrame,
) -> DiscreteHazardFeatureBundle:
    target = str(target)
    numeric_columns = discrete_hazard_numeric_feature_columns(target)
    causal_columns, history_columns = _discrete_hazard_source_columns(target)
    base = _normalise_forecast_keys(causal_features.frame, "discrete-hazard causal features")
    history = _normalise_forecast_keys(event_history, "discrete-hazard event history")
    _require_columns(base, causal_columns)
    _require_columns(history, history_columns)
    base_keys = base.loc[:, ["station_id", "hour_utc"]].reset_index(drop=True)
    history_keys = history.loc[:, ["station_id", "hour_utc"]].reset_index(drop=True)
    if not base_keys.equals(history_keys):
        raise RuntimeError("discrete-hazard causal and history feature keys differ")
    features = base.loc[:, ["station_id", "hour_utc", *causal_columns]].copy()
    features = features.merge(
        history.loc[:, ["station_id", "hour_utc", *history_columns]],
        on=["station_id", "hour_utc"],
        how="left",
        validate="one_to_one",
        indicator="_history_merge",
    )
    if not features["_history_merge"].eq("both").all():
        raise RuntimeError("discrete-hazard rows lack event-history values")
    features = features.drop(columns="_history_merge")
    _require_columns(registry, ["station_id", "install_date"])
    station_context = registry.loc[:, ["station_id", "install_date"]].copy()
    station_context["station_id"] = station_context["station_id"].astype(str)
    station_context["install_date"] = pd.to_datetime(
        station_context["install_date"], utc=True, errors="coerce"
    )
    station_context = station_context.drop_duplicates("station_id", keep="last")
    features = features.merge(
        station_context,
        on="station_id",
        how="left",
        validate="many_to_one",
        indicator="_registry_merge",
    )
    if not features["_registry_merge"].eq("both").all():
        missing = sorted(
            features.loc[
                ~features["_registry_merge"].eq("both"), "station_id"
            ].astype(str).unique()
        )
        raise RuntimeError(f"discrete-hazard rows lack station registry context: {missing}")
    features["hazard_station_age_hours"] = (
        (features["hour_utc"] - features["install_date"]) / pd.Timedelta(hours=1)
    ).where(lambda value: value.ge(0.0))
    features = features.drop(columns=["install_date", "_registry_merge"])
    timestamp = pd.to_datetime(features["hour_utc"], utc=True, errors="coerce")
    day_of_week = timestamp.dt.dayofweek.astype(float)
    month = timestamp.dt.month.astype(float) - 1.0
    features["hazard_day_of_week_sin"] = np.sin(2.0 * np.pi * day_of_week / 7.0)
    features["hazard_day_of_week_cos"] = np.cos(2.0 * np.pi * day_of_week / 7.0)
    features["hazard_month_sin"] = np.sin(2.0 * np.pi * month / 12.0)
    features["hazard_month_cos"] = np.cos(2.0 * np.pi * month / 12.0)
    for group in SENSOR_GROUP_ORDER:
        source_columns = [
            f"causal_detector_{kind}_count_24h_{group}"
            for kind in FORECAST_DETECTOR_KINDS
        ]
        features[f"hazard_detector_count_24h_{group}"] = features.loc[
            :, source_columns
        ].apply(pd.to_numeric, errors="coerce").sum(axis=1, min_count=1)
    since_end = f"history_{target}_hours_since_last_event_end"
    features[f"hazard_log1p_hours_since_last_{target}_event_end"] = np.log1p(
        pd.to_numeric(features[since_end], errors="coerce").clip(lower=0.0)
    )
    stations = tuple(sorted(station_context["station_id"].astype(str).unique()))
    station_columns = tuple(f"hazard_station_indicator_{station}" for station in stations)
    for station, column in zip(stations, station_columns, strict=True):
        features[column] = features["station_id"].eq(station).astype(float)
    missing = sorted(set(numeric_columns).difference(features.columns))
    if missing:
        raise KeyError(f"discrete-hazard features missing: {missing}")
    model_columns = (*numeric_columns, *station_columns)
    features.loc[:, list(model_columns)] = features.loc[:, list(model_columns)].apply(
        pd.to_numeric, errors="coerce"
    ).replace([np.inf, -np.inf], np.nan)
    audit = _discrete_hazard_feature_audit(
        target=target,
        numeric_columns=numeric_columns,
        station_columns=station_columns,
    )
    if set(audit["feature"]) != set(model_columns):
        raise RuntimeError("discrete-hazard feature audit does not match model schema")
    if not audit["causality"].isin({"causal", "strictly_prior_history"}).all():
        raise RuntimeError("discrete-hazard feature audit contains an invalid causality state")
    return DiscreteHazardFeatureBundle(
        frame=features.sort_values(["station_id", "hour_utc"], kind="mergesort").reset_index(
            drop=True
        ),
        numeric_feature_columns=numeric_columns,
        station_indicator_columns=station_columns,
        model_feature_columns=model_columns,
        causality_audit=audit,
    )


def _future_validation_sample_keys(
    grid: pd.DataFrame,
    *,
    sample_size: int,
) -> pd.DataFrame:
    source = _normalise_forecast_keys(grid, "future-isolation validation grid")
    candidates = source.loc[
        source["is_transmitting"].astype("boolean").fillna(False),
        ["station_id", "hour_utc"],
    ].drop_duplicates()
    if candidates.empty:
        candidates = source.loc[:, ["station_id", "hour_utc"]].drop_duplicates()
    candidates = candidates.sort_values(["hour_utc", "station_id"], kind="mergesort")
    count = min(int(sample_size), len(candidates))
    if count <= 0:
        return candidates.iloc[0:0].copy()
    positions = np.linspace(0, len(candidates) - 1, num=count, dtype=int)
    return candidates.iloc[np.unique(positions)].reset_index(drop=True)


def _truncate_as_of(frame: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    source = frame.copy(deep=True)
    time_column = "hour_utc" if "hour_utc" in source.columns else "time_utc"
    if time_column not in source.columns:
        raise KeyError("hour_utc")
    timestamps = pd.to_datetime(source[time_column], utc=True, errors="coerce")
    return source.loc[timestamps.le(cutoff)].copy()


def _registry_as_of(registry: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    source = registry.copy(deep=True)
    if "install_date" not in source.columns:
        raise KeyError("install_date")
    installed = pd.to_datetime(source["install_date"], utc=True, errors="coerce")
    return source.loc[installed.le(cutoff.normalize())].copy()


def validate_delete_future_features(
    grid: pd.DataFrame,
    *,
    availability: pd.DataFrame,
    observations: pd.DataFrame,
    registry: pd.DataFrame,
    reference: pd.DataFrame,
    feature_matrix_columns: list[str] | tuple[str, ...] = (),
    feature_columns: tuple[str, ...] | list[str] | None = None,
    sample_keys: pd.DataFrame | None = None,
    sample_size: int = 8,
    full_bundle: CausalForecastFeatureBundle | None = None,
) -> pd.DataFrame:
    full = (
        build_causal_forecast_features(
            grid,
            availability=availability,
            observations=observations,
            registry=registry,
            reference=reference,
            feature_matrix_columns=feature_matrix_columns,
            feature_columns=feature_columns,
        )
        if full_bundle is None
        else full_bundle
    )
    keys = (
        _future_validation_sample_keys(grid, sample_size=sample_size)
        if sample_keys is None
        else _normalise_forecast_keys(sample_keys, "future-isolation validation keys")
    )
    rows: list[dict[str, object]] = []
    for key in keys.itertuples(index=False):
        cutoff = pd.Timestamp(key.hour_utc)
        truncated_grid = _truncate_as_of(grid, cutoff)
        truncated = build_causal_forecast_features(
            truncated_grid,
            availability=_truncate_as_of(availability, cutoff),
            observations=_truncate_as_of(observations, cutoff),
            registry=_registry_as_of(registry, cutoff),
            reference=_truncate_as_of(reference, cutoff),
            feature_matrix_columns=feature_matrix_columns,
            feature_columns=feature_columns,
        )
        full_row = full.frame.loc[
            full.frame["station_id"].eq(str(key.station_id))
            & full.frame["hour_utc"].eq(cutoff),
            list(full.feature_columns),
        ]
        truncated_row = truncated.frame.loc[
            truncated.frame["station_id"].eq(str(key.station_id))
            & truncated.frame["hour_utc"].eq(cutoff),
            list(full.feature_columns),
        ]
        if len(full_row) != 1 or len(truncated_row) != 1:
            raise RuntimeError("future-isolation check could not recover its scored row")
        left = full_row.iloc[0].to_numpy(dtype=float)
        right = truncated_row.iloc[0].to_numpy(dtype=float)
        passed = np.isclose(left, right, rtol=1e-10, atol=1e-12, equal_nan=True)
        for feature, original_value, truncated_value, is_equal in zip(
            full.feature_columns,
            left,
            right,
            passed,
            strict=True,
        ):
            rows.append(
                {
                    "station_id": str(key.station_id),
                    "hour_utc": cutoff,
                    "feature": feature,
                    "full_value": float(original_value)
                    if np.isfinite(original_value)
                    else np.nan,
                    "as_of_value": float(truncated_value)
                    if np.isfinite(truncated_value)
                    else np.nan,
                    "passed": bool(is_equal),
                }
            )
    return pd.DataFrame(rows)


def summarize_delete_future_validation(detail: pd.DataFrame) -> pd.DataFrame:
    required = {"station_id", "hour_utc", "feature", "passed"}
    missing = sorted(required.difference(detail.columns))
    if missing:
        raise KeyError(missing)
    if detail.empty:
        return pd.DataFrame(
            [
                {
                    "sample_rows_validated": 0,
                    "features_validated": 0,
                    "feature_value_comparisons": 0,
                    "failed_comparisons": 0,
                    "all_passed": False,
                }
            ]
        )
    return pd.DataFrame(
        [
            {
                "sample_rows_validated": int(
                    detail.loc[:, ["station_id", "hour_utc"]].drop_duplicates().shape[0]
                ),
                "features_validated": int(detail["feature"].nunique()),
                "feature_value_comparisons": int(len(detail)),
                "failed_comparisons": int((~detail["passed"].astype(bool)).sum()),
                "all_passed": bool(detail["passed"].astype(bool).all()),
            }
        ]
    )


def event_history_feature_columns(horizon: int) -> tuple[str, ...]:
    if int(horizon) <= 0:
        raise ValueError("event-history horizon must be positive")
    trailing_windows = tuple(
        dict.fromkeys(
            [
                int(horizon),
                int(horizon) * 2,
                *EVENT_HISTORY_FIXED_WINDOWS,
            ]
        )
    )
    columns: list[str] = []
    for source in EVENT_HISTORY_SOURCES:
        for window in trailing_windows:
            columns.extend(
                [
                    f"history_{source}_any_event_trailing_{window}h",
                    f"history_{source}_event_hour_count_trailing_{window}h",
                ]
            )
            if source == "fault":
                columns.extend(
                    [
                        f"history_fault_known_fraction_trailing_{window}h",
                        f"history_fault_unknown_present_trailing_{window}h",
                    ]
                )
        columns.extend(
            [
                f"history_{source}_hours_since_last_event_end",
                f"history_{source}_hours_since_last_event_begin",
                f"history_{source}_last_event_end_never_seen",
                f"history_{source}_last_event_begin_never_seen",
            ]
        )
        for window in EVENT_HISTORY_LONG_WINDOWS:
            columns.extend(
                [
                    f"history_{source}_distinct_event_count_trailing_{window}h",
                    f"history_{source}_event_hour_rate_trailing_{window}h",
                ]
            )
    return tuple(columns)


def _event_history_all_feature_columns(
    horizons: tuple[int, ...],
) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            column
            for horizon in horizons
            for column in event_history_feature_columns(int(horizon))
        )
    )


def build_backward_event_history_features(
    grid: pd.DataFrame,
    *,
    horizons: tuple[int, ...] = HORIZONS,
) -> pd.DataFrame:
    required = [
        "station_id",
        "hour_utc",
        *EVENT_HISTORY_SOURCES.values(),
        *(
            column
            for column in EVENT_HISTORY_KNOWN_COLUMNS.values()
            if column is not None
        ),
    ]
    _require_columns(grid, required)
    if not horizons:
        raise ValueError("event-history horizons must be non-empty")
    base = _normalise_forecast_keys(grid, "event-history grid")
    result = base.loc[:, ["station_id", "hour_utc"]].copy()
    for source, event_column in EVENT_HISTORY_SOURCES.items():
        base[event_column] = (
            base[event_column].astype("boolean").fillna(False).astype(bool)
        )
    all_columns = _event_history_all_feature_columns(horizons)
    for station_id, station in base.groupby("station_id", sort=False):
        station_index = station.index
        hours = station["hour_utc"]
        for source, event_column in EVENT_HISTORY_SOURCES.items():
            known_column = EVENT_HISTORY_KNOWN_COLUMNS[source]
            known = (
                pd.Series(True, index=station.index, dtype=bool)
                if known_column is None
                else station[known_column].astype("boolean").fillna(False).astype(bool)
            )
            event = station[event_column].astype(bool) & known
            prior_event = event.shift(1, fill_value=False)
            prior_known = known.shift(1, fill_value=False)
            observed_start = (
                known
                & event
                & prior_known
                & ~event.shift(1, fill_value=False)
            )
            prior_observed_start = observed_start.shift(1, fill_value=False)
            prior_confirmed_end = (
                prior_known
                & ~prior_event
                & known.shift(2, fill_value=False)
                & event.shift(2, fill_value=False)
            )
            last_confirmed_end = hours.shift(2).where(prior_confirmed_end).ffill()
            last_known_begin = hours.shift(1).where(prior_observed_start).ffill()
            since_end = (
                (hours - last_confirmed_end) / pd.Timedelta(hours=1)
            ).to_numpy(dtype=float)
            since_begin = (
                (hours - last_known_begin) / pd.Timedelta(hours=1)
            ).to_numpy(dtype=float)
            end_never_seen = ~last_confirmed_end.notna().to_numpy(dtype=bool)
            begin_never_seen = ~last_known_begin.notna().to_numpy(dtype=bool)
            result.loc[station_index, f"history_{source}_hours_since_last_event_end"] = (
                np.where(end_never_seen, 720.0, since_end).clip(max=720.0)
            )
            result.loc[station_index, f"history_{source}_hours_since_last_event_begin"] = (
                np.where(begin_never_seen, 720.0, since_begin).clip(max=720.0)
            )
            result.loc[station_index, f"history_{source}_last_event_end_never_seen"] = (
                end_never_seen.astype(float)
            )
            result.loc[station_index, f"history_{source}_last_event_begin_never_seen"] = (
                begin_never_seen.astype(float)
            )
            for window in sorted(
                {
                    int(value)
                    for horizon in horizons
                    for value in (int(horizon), int(horizon) * 2)
                }
                | set(EVENT_HISTORY_FIXED_WINDOWS)
            ):
                trailing_count = prior_event.astype(float).rolling(
                    int(window), min_periods=1
                ).sum()
                result.loc[
                    station_index,
                    f"history_{source}_any_event_trailing_{window}h",
                ] = trailing_count.gt(0.0).to_numpy(dtype=float)
                result.loc[
                    station_index,
                    f"history_{source}_event_hour_count_trailing_{window}h",
                ] = trailing_count.to_numpy(dtype=float)
                if source == "fault":
                    known_fraction = prior_known.astype(float).rolling(
                        int(window), min_periods=1
                    ).mean()
                    result.loc[
                        station_index,
                        f"history_fault_known_fraction_trailing_{window}h",
                    ] = known_fraction.to_numpy(dtype=float)
                    result.loc[
                        station_index,
                        f"history_fault_unknown_present_trailing_{window}h",
                    ] = known_fraction.lt(1.0).to_numpy(dtype=float)
            for window in EVENT_HISTORY_LONG_WINDOWS:
                prior_start_count = prior_observed_start.astype(float).rolling(
                    int(window), min_periods=1
                ).sum()
                prior_observed_hours = prior_known.astype(float).rolling(
                    int(window), min_periods=1
                ).sum()
                prior_event_hours = prior_event.astype(float).rolling(
                    int(window), min_periods=1
                ).sum()
                result.loc[
                    station_index,
                    f"history_{source}_distinct_event_count_trailing_{window}h",
                ] = prior_start_count.to_numpy(dtype=float)
                result.loc[
                    station_index,
                    f"history_{source}_event_hour_rate_trailing_{window}h",
                ] = (
                    prior_event_hours / prior_observed_hours.where(prior_observed_hours.gt(0.0))
                ).to_numpy(dtype=float)
    missing = sorted(set(all_columns).difference(result.columns))
    if missing:
        raise RuntimeError(f"event-history features missing: {missing}")
    result.loc[:, all_columns] = result.loc[:, all_columns].apply(
        pd.to_numeric, errors="coerce"
    ).replace([np.inf, -np.inf], np.nan)
    return result.sort_values(["station_id", "hour_utc"], kind="mergesort").reset_index(
        drop=True
    )


def build_event_history_feature_audit(horizon: int) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for feature in event_history_feature_columns(int(horizon)):
        source = "fault labels" if "history_fault_" in feature else "outage observations"
        deployment_status = (
            "optimistic_ground_truth_history_not_available_live"
            if source == "fault labels"
            else "operationally_observable_after_event"
        )
        rows.append(
            {
                "feature": feature,
                "source": source,
                "causality": "strictly_prior_history",
                "included": True,
                "time_contract": "strictly before the scored hour",
                "scope_change": "event_history_experiment",
                "reason": (
                    "The feature summarizes only prior event states on the continuous "
                    "station-hour grid. Fault history uses confirmed ground-truth labels "
                    "for this optimistic experiment; excluded fault-label hours remain "
                    "unknown rather than fault-free, and it is not a live deployment input."
                    if source == "fault labels"
                    else "The feature summarizes only prior observed full-outage states on the continuous station-hour grid."
                ),
                "deployment_status": deployment_status,
            }
        )
    return pd.DataFrame(rows).sort_values("feature", kind="mergesort").reset_index(
        drop=True
    )


def validate_delete_future_event_history_features(
    grid: pd.DataFrame,
    *,
    horizons: tuple[int, ...] = HORIZONS,
    sample_keys: pd.DataFrame | None = None,
    sample_size: int = 8,
    full_history: pd.DataFrame | None = None,
) -> pd.DataFrame:
    base = _normalise_forecast_keys(grid, "event-history future-isolation grid")
    full = (
        build_backward_event_history_features(base, horizons=horizons)
        if full_history is None
        else _normalise_forecast_keys(full_history, "event-history feature frame")
    )
    feature_columns = _event_history_all_feature_columns(horizons)
    _require_columns(full, feature_columns)
    if sample_keys is None:
        candidates = base.loc[:, ["station_id", "hour_utc"]].sort_values(
            ["hour_utc", "station_id"], kind="mergesort"
        )
        count = min(int(sample_size), len(candidates))
        positions = np.linspace(0, len(candidates) - 1, num=count, dtype=int)
        keys = candidates.iloc[np.unique(positions)].reset_index(drop=True)
    else:
        keys = _normalise_forecast_keys(sample_keys, "event-history validation keys")
    rows: list[dict[str, object]] = []
    for key in keys.itertuples(index=False):
        cutoff = pd.Timestamp(key.hour_utc)
        truncated = build_backward_event_history_features(
            _truncate_as_of(base, cutoff),
            horizons=horizons,
        )
        full_row = full.loc[
            full["station_id"].eq(str(key.station_id))
            & full["hour_utc"].eq(cutoff),
            feature_columns,
        ]
        truncated_row = truncated.loc[
            truncated["station_id"].eq(str(key.station_id))
            & truncated["hour_utc"].eq(cutoff),
            feature_columns,
        ]
        if len(full_row) != 1 or len(truncated_row) != 1:
            raise RuntimeError("event-history future-isolation check could not recover its scored row")
        left = full_row.iloc[0].to_numpy(dtype=float)
        right = truncated_row.iloc[0].to_numpy(dtype=float)
        passed = np.isclose(left, right, rtol=1e-10, atol=1e-12, equal_nan=True)
        for feature, original_value, truncated_value, is_equal in zip(
            feature_columns, left, right, passed, strict=True
        ):
            rows.append(
                {
                    "station_id": str(key.station_id),
                    "hour_utc": cutoff,
                    "feature": feature,
                    "full_value": float(original_value)
                    if np.isfinite(original_value)
                    else np.nan,
                    "as_of_value": float(truncated_value)
                    if np.isfinite(truncated_value)
                    else np.nan,
                    "passed": bool(is_equal),
                }
            )
    return pd.DataFrame(rows)


def build_retrospective_persistence_history(
    grid: pd.DataFrame,
    *,
    event_column: str,
    horizons: tuple[int, ...] = HORIZONS,
) -> pd.DataFrame:
    _require_columns(grid, ["station_id", "hour_utc", event_column])
    base = _normalise_forecast_keys(grid, "persistence history grid")
    base[event_column] = base[event_column].astype("boolean").fillna(False).astype(bool)
    result = base.loc[:, ["station_id", "hour_utc"]].copy()
    for _, station_frame in base.groupby("station_id", sort=False):
        prior_event = station_frame[event_column].astype(float).shift(1)
        for horizon in horizons:
            result.loc[station_frame.index, f"persistence_event_count_{int(horizon)}h"] = (
                prior_event.rolling(int(horizon), min_periods=1).sum().to_numpy()
            )
    return result.sort_values(["station_id", "hour_utc"], kind="mergesort").reset_index(
        drop=True
    )
