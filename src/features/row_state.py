from __future__ import annotations

import pandas as pd

from src.config.paths import MEASUREMENT_COLUMNS

WARMUP_DAYS = 7

ROW_STATE_INVALID = "invalid_missing_id_or_time"
ROW_STATE_NEVER_ACTIVE = "station_never_active"
ROW_STATE_BEFORE_FIRST = "before_first_present"
ROW_STATE_TERMINAL_PADDED = "terminal_padded_absence"
ROW_STATE_WARMUP = "warmup"
ROW_STATE_TRUE_OUTAGE = "true_outage_candidate"
ROW_STATE_PARTIAL = "online_partial_missing"
ROW_STATE_COMPLETE = "online_complete"


def _require_columns(
    frame: pd.DataFrame,
    required_columns: list[str],
    frame_name: str,
) -> None:
    missing_columns = [
        column for column in required_columns
        if column not in frame.columns
    ]
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(f"{frame_name} is missing required columns: {missing}")


def classify_row_states(
    df: pd.DataFrame,
    registry: pd.DataFrame,
) -> pd.DataFrame:
    _require_columns(df, ["station_id", "hour_utc", "data_present"], "df")
    _require_columns(df, MEASUREMENT_COLUMNS, "df")
    _require_columns(registry, ["station_id"], "registry")

    out = df.copy(deep=True)

    out["hour_utc"] = pd.to_datetime(
        out["hour_utc"],
        utc=True,
        errors="coerce",
    )
    data_present_numeric = pd.to_numeric(
        out["data_present"],
        errors="coerce",
    )
    data_present = data_present_numeric.eq(1)

    valid_station_id = out["station_id"].notna()
    valid_timestamp = out["hour_utc"].notna()
    valid_station_time = valid_station_id & valid_timestamp

    present_rows = valid_station_time & data_present
    present_bounds = (
        out.loc[present_rows]
        .groupby("station_id", dropna=True)["hour_utc"]
        .agg(["min", "max"])
    )

    first_present = out["station_id"].map(present_bounds["min"])
    last_present = out["station_id"].map(present_bounds["max"])
    out["first_present_timestamp"] = first_present
    out["last_present_timestamp"] = last_present

    data_absent = data_present_numeric.eq(0)
    measurement_has_null = out[MEASUREMENT_COLUMNS].isna().any(axis=1)

    out["flag_missing_station_id"] = ~valid_station_id
    out["flag_missing_timestamp"] = ~valid_timestamp
    out["flag_station_never_active"] = (
        valid_station_id
        & ~out["station_id"].isin(present_bounds.index)
    )
    out["flag_data_absent"] = data_absent
    out["flag_before_first_present"] = (
        valid_station_time
        & first_present.notna()
        & out["hour_utc"].lt(first_present)
    )
    out["flag_after_last_present"] = (
        valid_station_time
        & last_present.notna()
        & out["hour_utc"].gt(last_present)
    )
    warmup_end = first_present + pd.Timedelta(days=WARMUP_DAYS)
    out["flag_warmup"] = (
        valid_station_time
        & data_present
        & first_present.notna()
        & out["hour_utc"].ge(first_present)
        & out["hour_utc"].lt(warmup_end)
    )
    out["flag_online_partial_missing"] = (
        data_present
        & measurement_has_null
    )
    out["flag_online_complete"] = (
        data_present
        & ~measurement_has_null
    )
    out["flag_true_outage_candidate"] = (
        valid_station_time
        & data_absent
        & first_present.notna()
        & last_present.notna()
        & out["hour_utc"].ge(first_present)
        & out["hour_utc"].le(last_present)
    )

    row_state = pd.Series(pd.NA, index=out.index, dtype="object")

    def assign_state(mask: pd.Series, state: str) -> None:
        pending = row_state.isna()
        row_state.loc[pending & mask] = state

    assign_state(
        out["flag_missing_station_id"] | out["flag_missing_timestamp"],
        ROW_STATE_INVALID,
    )
    assign_state(out["flag_station_never_active"], ROW_STATE_NEVER_ACTIVE)
    assign_state(out["flag_before_first_present"], ROW_STATE_BEFORE_FIRST)
    assign_state(
        out["flag_after_last_present"] & out["flag_data_absent"],
        ROW_STATE_TERMINAL_PADDED,
    )
    assign_state(out["flag_warmup"], ROW_STATE_WARMUP)
    assign_state(out["flag_true_outage_candidate"], ROW_STATE_TRUE_OUTAGE)
    assign_state(out["flag_online_partial_missing"], ROW_STATE_PARTIAL)
    assign_state(out["flag_online_complete"], ROW_STATE_COMPLETE)

    out["row_state"] = row_state
    return out
