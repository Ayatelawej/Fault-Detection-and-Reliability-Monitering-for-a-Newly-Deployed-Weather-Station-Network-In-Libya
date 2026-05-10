from __future__ import annotations

import pandas as pd

from src.availability.build_network_outage_windows import assign_outage_class
from src.config.paths import (
    AVAILABILITY_EVENTS_PATH,
    HOURLY_ROW_STATES_PATH,
    NETWORK_OUTAGE_WINDOWS_PATH,
    ensure_directories,
)
from src.features.row_state import ROW_STATE_TRUE_OUTAGE

EVENT_COLUMNS = [
    "event_id",
    "station_id",
    "start_utc",
    "end_utc",
    "duration_hours",
    "outage_class",
]

DURATION_BUCKETS = ["short", "medium", "long", "very_long"]


def _empty_events_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "event_id": pd.Series(dtype="object"),
            "station_id": pd.Series(dtype="object"),
            "start_utc": pd.Series(dtype="datetime64[ns, UTC]"),
            "end_utc": pd.Series(dtype="datetime64[ns, UTC]"),
            "duration_hours": pd.Series(dtype="int64"),
            "outage_class": pd.Series(dtype="object"),
        }
    )[EVENT_COLUMNS]


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


def build_availability_events(hourly_row_states: pd.DataFrame) -> pd.DataFrame:
    _require_columns(
        hourly_row_states,
        ["station_id", "hour_utc", "row_state"],
        "hourly_row_states",
    )

    row_state = hourly_row_states["row_state"].astype("string")
    outages = hourly_row_states.loc[
        row_state.eq(ROW_STATE_TRUE_OUTAGE).fillna(False),
        ["station_id", "hour_utc"],
    ].copy()

    outages["station_id"] = outages["station_id"].astype("string")
    outages["hour_utc"] = pd.to_datetime(
        outages["hour_utc"],
        utc=True,
        errors="coerce",
    )
    outages = outages.loc[
        outages["station_id"].notna() & outages["hour_utc"].notna()
    ].copy()

    if outages.empty:
        return _empty_events_frame()

    outages = outages.sort_values(
        ["station_id", "hour_utc"],
        kind="mergesort",
    )
    hour_gap = outages.groupby("station_id")["hour_utc"].diff()
    new_segment = hour_gap.ne(pd.Timedelta(hours=1))
    outages["segment_id"] = new_segment.groupby(outages["station_id"]).cumsum()

    events = (
        outages.groupby(["station_id", "segment_id"], sort=False)
        .agg(
            start_utc=("hour_utc", "first"),
            end_utc=("hour_utc", "last"),
            duration_hours=("hour_utc", "size"),
        )
        .reset_index()
    )
    events["station_id"] = events["station_id"].astype(str)
    events["duration_hours"] = (
        pd.to_numeric(events["duration_hours"], errors="coerce")
        .fillna(0)
        .astype("int64")
    )
    events["outage_class"] = "unknown"
    events["event_id"] = [
        f"{row.station_id}__{row.start_utc:%Y%m%dT%H}__{row.duration_hours:04d}h"
        for row in events.itertuples(index=False)
    ]

    events = events.sort_values(
        ["start_utc", "station_id"],
        kind="mergesort",
    ).reset_index(drop=True)
    return events[EVENT_COLUMNS]


def classify_availability_events(
    events: pd.DataFrame,
    network_outage_windows: pd.DataFrame,
) -> pd.DataFrame:
    return assign_outage_class(events, network_outage_windows)[EVENT_COLUMNS]


def _duration_bucket_counts(events: pd.DataFrame) -> pd.Series:
    duration_hours = pd.to_numeric(
        events["duration_hours"],
        errors="coerce",
    )
    buckets = pd.cut(
        duration_hours,
        bins=[0, 3, 24, 168, float("inf")],
        labels=DURATION_BUCKETS,
        right=True,
    )
    counts = buckets.value_counts().reindex(DURATION_BUCKETS, fill_value=0)
    counts.index.name = None
    return counts


def _print_summary(events: pd.DataFrame) -> None:
    duration_hours = pd.to_numeric(
        events["duration_hours"],
        errors="coerce",
    ).fillna(0)

    print("Phase 2 availability events complete.")
    print(f"Total events: {len(events):,}")
    print(f"Total absent hours: {int(duration_hours.sum()):,}")
    print("Per-duration-bucket counts:")
    print(_duration_bucket_counts(events).to_string())
    print("10 longest events:")
    longest_events = events.sort_values(
        ["duration_hours", "start_utc", "station_id"],
        ascending=[False, True, True],
        kind="mergesort",
    ).head(10)
    if longest_events.empty:
        print("(none)")
    else:
        print(
            longest_events[
                ["event_id", "station_id", "start_utc", "duration_hours"]
            ].to_string(index=False)
        )
    print("Per-station event count:")
    station_counts = (
        events.groupby("station_id")
        .size()
        .rename("event_count")
        .reset_index()
        .sort_values(
            ["event_count", "station_id"],
            ascending=[False, True],
            kind="mergesort",
        )
    )
    if station_counts.empty:
        print("(none)")
    else:
        print(station_counts.to_string(index=False))
    print("Outage class counts:")
    print(events["outage_class"].value_counts().to_string())


def main() -> None:
    hourly_row_states = pd.read_parquet(HOURLY_ROW_STATES_PATH)
    events = build_availability_events(hourly_row_states)
    if NETWORK_OUTAGE_WINDOWS_PATH.exists():
        network_outage_windows = pd.read_csv(NETWORK_OUTAGE_WINDOWS_PATH)
        events = classify_availability_events(events, network_outage_windows)

    ensure_directories()
    events.to_parquet(AVAILABILITY_EVENTS_PATH, index=False)
    _print_summary(events)


if __name__ == "__main__":
    main()
