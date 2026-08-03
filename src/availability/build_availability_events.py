from __future__ import annotations

import pandas as pd

from src.availability.build_network_outage_windows import assign_outage_class
from src.config.paths import (
    AVAILABILITY_CLASSIFICATION_PATH,
    AVAILABILITY_EVENTS_PATH,
    AVAILABILITY_REPORT_PATH,
    HOURLY_ROW_STATES_PATH,
    MEASUREMENT_COLUMNS,
    NETWORK_OUTAGE_WINDOWS_PATH,
    PARTIAL_OUTAGE_EVENTS_PATH,
    STRUCTURAL_AVAILABILITY_GAPS_PATH,
    ensure_directories,
)
from src.features.row_state import ROW_STATE_TERMINAL_PADDED, ROW_STATE_TRUE_OUTAGE
from src.rules.channel_handlers import sensor_group_for_channel
from src.rules.config import SENSOR_GROUP_PREFIXES

EVENT_COLUMNS = [
    "event_id",
    "station_id",
    "start_utc",
    "end_utc",
    "duration_hours",
    "outage_class",
]

PARTIAL_EVENT_COLUMNS = [
    "event_id",
    "station_id",
    "start_utc",
    "end_utc",
    "duration_hours",
    "availability_class",
    "absent_sensor_groups",
]

STRUCTURAL_GAP_COLUMNS = [
    "gap_id",
    "station_id",
    "preceding_hour_utc",
    "following_hour_utc",
    "gap_duration_hours",
    "first_missing_hour_utc",
    "last_missing_hour_utc",
    "omitted_hour_count",
    "preceding_row_state",
    "following_row_state",
]

AVAILABILITY_CLASS_FULL_OUTAGE = "full_outage"
AVAILABILITY_CLASS_PARTIAL_OUTAGE = "partial_outage"
AVAILABILITY_CLASS_ONLINE = "online"
AVAILABILITY_CLASS_EXCLUDED = "excluded"

AVAILABILITY_SCOPE_ACTIVE = "active"
AVAILABILITY_SCOPE_STRUCTURAL_GAP = "structural_gap"
AVAILABILITY_SCOPE_TERMINAL_PADDED = "terminal_padded"
AVAILABILITY_SCOPE_OTHER_EXCLUDED = "other_excluded"

SOURCE_KIND_OBSERVED = "observed_row"
SOURCE_KIND_STRUCTURAL_GAP = "materialized_structural_gap"

SENSOR_GROUP_ORDER = tuple(sorted(set(SENSOR_GROUP_PREFIXES.values())))
SENSOR_GROUP_CHANNELS = {
    group: tuple(
        column
        for column in MEASUREMENT_COLUMNS
        if sensor_group_for_channel(column) == group
    )
    for group in SENSOR_GROUP_ORDER
}

CLASSIFICATION_COLUMNS = [
    "station_id",
    "hour_utc",
    "availability_scope",
    "availability_class",
    "absent_sensor_groups",
    "is_transmitting",
    "row_state",
    "source_kind",
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


def _empty_partial_events_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "event_id": pd.Series(dtype="object"),
            "station_id": pd.Series(dtype="object"),
            "start_utc": pd.Series(dtype="datetime64[ns, UTC]"),
            "end_utc": pd.Series(dtype="datetime64[ns, UTC]"),
            "duration_hours": pd.Series(dtype="int64"),
            "availability_class": pd.Series(dtype="object"),
            "absent_sensor_groups": pd.Series(dtype="object"),
        }
    )[PARTIAL_EVENT_COLUMNS]


def _empty_structural_gaps_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "gap_id": pd.Series(dtype="object"),
            "station_id": pd.Series(dtype="object"),
            "preceding_hour_utc": pd.Series(dtype="datetime64[ns, UTC]"),
            "following_hour_utc": pd.Series(dtype="datetime64[ns, UTC]"),
            "gap_duration_hours": pd.Series(dtype="int64"),
            "first_missing_hour_utc": pd.Series(dtype="datetime64[ns, UTC]"),
            "last_missing_hour_utc": pd.Series(dtype="datetime64[ns, UTC]"),
            "omitted_hour_count": pd.Series(dtype="int64"),
            "preceding_row_state": pd.Series(dtype="object"),
            "following_row_state": pd.Series(dtype="object"),
        }
    )[STRUCTURAL_GAP_COLUMNS]


def _empty_classification_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "station_id": pd.Series(dtype="object"),
            "hour_utc": pd.Series(dtype="datetime64[ns, UTC]"),
            "availability_scope": pd.Series(dtype="object"),
            "availability_class": pd.Series(dtype="object"),
            "absent_sensor_groups": pd.Series(dtype="object"),
            "is_transmitting": pd.Series(dtype="bool"),
            "row_state": pd.Series(dtype="object"),
            "source_kind": pd.Series(dtype="object"),
        }
    )[CLASSIFICATION_COLUMNS]


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


def sensor_group_channels() -> dict[str, tuple[str, ...]]:
    return dict(SENSOR_GROUP_CHANNELS)


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


def find_structural_availability_gaps(
    hourly_row_states: pd.DataFrame,
) -> pd.DataFrame:
    _require_columns(
        hourly_row_states,
        ["station_id", "hour_utc", "row_state"],
        "hourly_row_states",
    )

    rows = hourly_row_states.loc[
        :, ["station_id", "hour_utc", "row_state"]
    ].copy()
    rows["station_id"] = rows["station_id"].astype("string")
    rows["hour_utc"] = pd.to_datetime(
        rows["hour_utc"],
        utc=True,
        errors="coerce",
    )
    rows = rows.loc[
        rows["station_id"].notna() & rows["hour_utc"].notna()
    ].copy()
    if rows.empty:
        return _empty_structural_gaps_frame()

    rows = rows.sort_values(["station_id", "hour_utc"], kind="mergesort")
    rows["preceding_hour_utc"] = rows.groupby("station_id")["hour_utc"].shift()
    rows["preceding_row_state"] = rows.groupby("station_id")["row_state"].shift()
    rows["gap_duration_hours"] = (
        (rows["hour_utc"] - rows["preceding_hour_utc"])
        .dt.total_seconds()
        .div(3600)
    )
    gaps = rows.loc[rows["gap_duration_hours"].gt(1)].copy()
    if gaps.empty:
        return _empty_structural_gaps_frame()

    gaps["gap_duration_hours"] = (
        pd.to_numeric(gaps["gap_duration_hours"], errors="coerce")
        .fillna(0)
        .round()
        .astype("int64")
    )
    gaps["following_hour_utc"] = gaps["hour_utc"]
    gaps["following_row_state"] = gaps["row_state"]
    gaps["first_missing_hour_utc"] = (
        gaps["preceding_hour_utc"] + pd.Timedelta(hours=1)
    )
    gaps["last_missing_hour_utc"] = (
        gaps["following_hour_utc"] - pd.Timedelta(hours=1)
    )
    gaps["omitted_hour_count"] = gaps["gap_duration_hours"] - 1
    gaps["gap_id"] = [
        f"{row.station_id}__{row.preceding_hour_utc:%Y%m%dT%H}__"
        f"{row.omitted_hour_count:04d}h"
        for row in gaps.itertuples(index=False)
    ]
    gaps["station_id"] = gaps["station_id"].astype(str)
    gaps = gaps.sort_values(
        ["preceding_hour_utc", "station_id"],
        kind="mergesort",
    ).reset_index(drop=True)
    return gaps[STRUCTURAL_GAP_COLUMNS]


def _materialize_structural_gap_rows(gaps: pd.DataFrame) -> pd.DataFrame:
    if gaps.empty:
        return _empty_classification_frame()

    rows: list[dict[str, object]] = []
    for gap in gaps.itertuples(index=False):
        hours = pd.date_range(
            gap.first_missing_hour_utc,
            gap.last_missing_hour_utc,
            freq="h",
        )
        for hour_utc in hours:
            rows.append(
                {
                    "station_id": str(gap.station_id),
                    "hour_utc": hour_utc,
                    "availability_scope": AVAILABILITY_SCOPE_STRUCTURAL_GAP,
                    "availability_class": AVAILABILITY_CLASS_FULL_OUTAGE,
                    "absent_sensor_groups": "",
                    "is_transmitting": False,
                    "row_state": pd.NA,
                    "source_kind": SOURCE_KIND_STRUCTURAL_GAP,
                }
            )
    return pd.DataFrame(rows, columns=CLASSIFICATION_COLUMNS)


def build_hourly_availability_classification(
    hourly_row_states: pd.DataFrame,
    *,
    include_structural_gaps: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    _require_columns(
        hourly_row_states,
        ["station_id", "hour_utc", "data_present", "row_state", *MEASUREMENT_COLUMNS],
        "hourly_row_states",
    )

    rows = hourly_row_states.copy(deep=True)
    rows["station_id"] = rows["station_id"].astype("string")
    rows["hour_utc"] = pd.to_datetime(rows["hour_utc"], utc=True, errors="coerce")
    valid_station_hour = rows["station_id"].notna() & rows["hour_utc"].notna()
    transmitting = (
        pd.to_numeric(rows["data_present"], errors="coerce")
        .eq(1)
        .fillna(False)
        & valid_station_hour
    )
    full_outage = (
        rows["row_state"].astype("string").eq(ROW_STATE_TRUE_OUTAGE).fillna(False)
        & valid_station_hour
    )

    group_absence = pd.DataFrame(
        {
            group: rows.loc[:, list(channels)].isna().all(axis=1)
            for group, channels in SENSOR_GROUP_CHANNELS.items()
        },
        index=rows.index,
    )
    partial_outage = transmitting & group_absence.any(axis=1)

    absent_sensor_groups = pd.Series("", index=rows.index, dtype="object")
    group_flags = group_absence.loc[:, SENSOR_GROUP_ORDER]
    absent_sensor_groups.loc[partial_outage] = [
        "|".join(
            group
            for group, absent in zip(SENSOR_GROUP_ORDER, flags)
            if bool(absent)
        )
        for flags in group_flags.loc[partial_outage].itertuples(
            index=False,
            name=None,
        )
    ]

    availability_class = pd.Series(
        AVAILABILITY_CLASS_EXCLUDED,
        index=rows.index,
        dtype="object",
    )
    availability_class.loc[full_outage] = AVAILABILITY_CLASS_FULL_OUTAGE
    availability_class.loc[transmitting & ~partial_outage] = (
        AVAILABILITY_CLASS_ONLINE
    )
    availability_class.loc[partial_outage] = AVAILABILITY_CLASS_PARTIAL_OUTAGE

    availability_scope = pd.Series(
        AVAILABILITY_SCOPE_OTHER_EXCLUDED,
        index=rows.index,
        dtype="object",
    )
    availability_scope.loc[full_outage | transmitting] = AVAILABILITY_SCOPE_ACTIVE
    terminal_padded = (
        rows["row_state"].astype("string").eq(ROW_STATE_TERMINAL_PADDED).fillna(False)
    )
    availability_scope.loc[terminal_padded] = AVAILABILITY_SCOPE_TERMINAL_PADDED

    classification = pd.DataFrame(
        {
            "station_id": rows["station_id"].astype("object"),
            "hour_utc": rows["hour_utc"],
            "availability_scope": availability_scope,
            "availability_class": availability_class,
            "absent_sensor_groups": absent_sensor_groups,
            "is_transmitting": transmitting.astype(bool),
            "row_state": rows["row_state"].astype("object"),
            "source_kind": SOURCE_KIND_OBSERVED,
        }
    )

    gaps = find_structural_availability_gaps(rows)
    if include_structural_gaps:
        structural_gap_rows = _materialize_structural_gap_rows(gaps)
        classification = pd.concat(
            [classification, structural_gap_rows],
            ignore_index=True,
        )

    classification = classification.loc[
        classification["station_id"].notna() & classification["hour_utc"].notna()
    ].copy()
    classification["station_id"] = classification["station_id"].astype(str)
    classification["hour_utc"] = pd.to_datetime(
        classification["hour_utc"],
        utc=True,
        errors="coerce",
    )
    if classification.duplicated(["station_id", "hour_utc"]).any():
        raise ValueError("availability classification contains duplicate station-hour keys")
    classification = classification.sort_values(
        ["station_id", "hour_utc"],
        kind="mergesort",
    ).reset_index(drop=True)
    return classification[CLASSIFICATION_COLUMNS], gaps


def _union_sensor_groups(values: pd.Series) -> str:
    observed_groups: set[str] = set()
    for value in values.dropna().astype(str):
        observed_groups.update(
            group
            for group in value.split("|")
            if group in SENSOR_GROUP_ORDER
        )
    return "|".join(
        group for group in SENSOR_GROUP_ORDER if group in observed_groups
    )


def build_partial_outage_events(
    availability_classification: pd.DataFrame,
) -> pd.DataFrame:
    _require_columns(
        availability_classification,
        [
            "station_id",
            "hour_utc",
            "availability_class",
            "absent_sensor_groups",
        ],
        "availability_classification",
    )

    partial = availability_classification.loc[
        availability_classification["availability_class"].eq(
            AVAILABILITY_CLASS_PARTIAL_OUTAGE
        ),
        ["station_id", "hour_utc", "absent_sensor_groups"],
    ].copy()
    partial["station_id"] = partial["station_id"].astype("string")
    partial["hour_utc"] = pd.to_datetime(
        partial["hour_utc"],
        utc=True,
        errors="coerce",
    )
    partial = partial.loc[
        partial["station_id"].notna() & partial["hour_utc"].notna()
    ].copy()
    if partial.empty:
        return _empty_partial_events_frame()

    partial = partial.sort_values(
        ["station_id", "hour_utc"],
        kind="mergesort",
    )
    hour_gap = partial.groupby("station_id")["hour_utc"].diff()
    new_segment = hour_gap.ne(pd.Timedelta(hours=1))
    partial["segment_id"] = new_segment.groupby(partial["station_id"]).cumsum()

    events = (
        partial.groupby(["station_id", "segment_id"], sort=False)
        .agg(
            start_utc=("hour_utc", "first"),
            end_utc=("hour_utc", "last"),
            duration_hours=("hour_utc", "size"),
            absent_sensor_groups=("absent_sensor_groups", _union_sensor_groups),
        )
        .reset_index()
    )
    events["station_id"] = events["station_id"].astype(str)
    events["duration_hours"] = (
        pd.to_numeric(events["duration_hours"], errors="coerce")
        .fillna(0)
        .astype("int64")
    )
    events["availability_class"] = AVAILABILITY_CLASS_PARTIAL_OUTAGE
    events["event_id"] = [
        f"partial__{row.station_id}__{row.start_utc:%Y%m%dT%H}__"
        f"{row.duration_hours:04d}h"
        for row in events.itertuples(index=False)
    ]
    events = events.sort_values(
        ["start_utc", "station_id"],
        kind="mergesort",
    ).reset_index(drop=True)
    return events[PARTIAL_EVENT_COLUMNS]


def sensor_group_absence_mask(
    availability_classification: pd.DataFrame,
    sensor_group: str,
) -> pd.Series:
    if sensor_group not in SENSOR_GROUP_ORDER:
        raise ValueError(f"unknown sensor group: {sensor_group}")
    _require_columns(
        availability_classification,
        ["absent_sensor_groups"],
        "availability_classification",
    )
    return availability_classification["absent_sensor_groups"].fillna("").map(
        lambda groups: sensor_group in str(groups).split("|")
    )


def build_sensor_group_availability(
    availability_classification: pd.DataFrame,
) -> pd.DataFrame:
    _require_columns(
        availability_classification,
        ["is_transmitting", "absent_sensor_groups"],
        "availability_classification",
    )
    transmitting = availability_classification["is_transmitting"].fillna(False).astype(bool)
    transmitting_hours = int(transmitting.sum())
    rows = []
    for sensor_group in SENSOR_GROUP_ORDER:
        absent = transmitting & sensor_group_absence_mask(
            availability_classification,
            sensor_group,
        )
        absent_hours = int(absent.sum())
        present_hours = transmitting_hours - absent_hours
        availability_pct = (
            100.0 * present_hours / transmitting_hours
            if transmitting_hours > 0
            else float("nan")
        )
        rows.append(
            {
                "sensor_group": sensor_group,
                "transmitting_hours": transmitting_hours,
                "present_hours": present_hours,
                "absent_hours": absent_hours,
                "availability_pct": availability_pct,
            }
        )
    return pd.DataFrame(rows)


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


def _duration_summary(events: pd.DataFrame) -> dict[str, float | int]:
    duration_hours = pd.to_numeric(
        events["duration_hours"],
        errors="coerce",
    ).dropna()
    if duration_hours.empty:
        return {
            "event_count": 0,
            "total_hours": 0,
            "mean_hours": float("nan"),
            "median_hours": float("nan"),
            "max_hours": float("nan"),
        }
    return {
        "event_count": int(len(events)),
        "total_hours": int(duration_hours.sum()),
        "mean_hours": float(duration_hours.mean()),
        "median_hours": float(duration_hours.median()),
        "max_hours": float(duration_hours.max()),
    }


def write_operational_availability_outputs(
    hourly_row_states: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    classification, structural_gaps = build_hourly_availability_classification(
        hourly_row_states,
    )
    partial_events = build_partial_outage_events(classification)
    ensure_directories()
    classification.to_parquet(AVAILABILITY_CLASSIFICATION_PATH, index=False)
    partial_events.to_parquet(PARTIAL_OUTAGE_EVENTS_PATH, index=False)
    structural_gaps.to_csv(STRUCTURAL_AVAILABILITY_GAPS_PATH, index=False)
    return classification, partial_events, structural_gaps


def write_availability_report(
    full_events: pd.DataFrame,
    availability_classification: pd.DataFrame,
    partial_events: pd.DataFrame,
    structural_gaps: pd.DataFrame,
) -> None:
    _require_columns(
        availability_classification,
        ["availability_class", "availability_scope", "source_kind"],
        "availability_classification",
    )
    full_summary = _duration_summary(full_events)
    partial_summary = _duration_summary(partial_events)
    class_counts = (
        availability_classification["availability_class"]
        .value_counts()
        .reindex(
            [
                AVAILABILITY_CLASS_FULL_OUTAGE,
                AVAILABILITY_CLASS_PARTIAL_OUTAGE,
                AVAILABILITY_CLASS_ONLINE,
                AVAILABILITY_CLASS_EXCLUDED,
            ],
            fill_value=0,
        )
    )
    observed_rows = int(
        availability_classification["source_kind"].eq(SOURCE_KIND_OBSERVED).sum()
    )
    materialized_gap_hours = int(
        availability_classification["source_kind"].eq(
            SOURCE_KIND_STRUCTURAL_GAP
        ).sum()
    )
    in_scope = availability_classification["availability_class"].ne(
        AVAILABILITY_CLASS_EXCLUDED
    )
    in_scope_counts = (
        availability_classification.loc[in_scope, "availability_class"]
        .value_counts()
        .reindex(
            [
                AVAILABILITY_CLASS_FULL_OUTAGE,
                AVAILABILITY_CLASS_PARTIAL_OUTAGE,
                AVAILABILITY_CLASS_ONLINE,
            ],
            fill_value=0,
        )
    )
    group_availability = build_sensor_group_availability(availability_classification)
    mapping_lines = [
        f"{group}: {', '.join(SENSOR_GROUP_CHANNELS[group])}"
        for group in SENSOR_GROUP_ORDER
    ]

    lines = [
        "Availability classification report",
        "",
        "This is a rule-derived availability monitor. Partial outages have no independent ground truth, so no precision, recall, F1, or model-validation metrics are reported.",
        "",
        "Frozen legacy full-outage series",
        f"events: {full_summary['event_count']:,}",
        f"hours: {full_summary['total_hours']:,}",
        f"mean_duration_hours: {full_summary['mean_hours']:.2f}",
        f"median_duration_hours: {full_summary['median_hours']:.2f}",
        f"max_duration_hours: {full_summary['max_hours']:.0f}",
        "",
        "Sensor-group channel mapping",
        *mapping_lines,
        "",
        "Classification accounting",
        f"observed_station_hour_rows: {observed_rows:,}",
        f"materialized_structural_gap_hours: {materialized_gap_hours:,}",
        f"terminal_or_other_excluded_rows: {int(class_counts.loc[AVAILABILITY_CLASS_EXCLUDED]):,}",
        f"in_scope_rows: {int(in_scope.sum()):,}",
        f"in_scope_full_outage: {int(in_scope_counts.loc[AVAILABILITY_CLASS_FULL_OUTAGE]):,}",
        f"in_scope_partial_outage: {int(in_scope_counts.loc[AVAILABILITY_CLASS_PARTIAL_OUTAGE]):,}",
        f"in_scope_online: {int(in_scope_counts.loc[AVAILABILITY_CLASS_ONLINE]):,}",
        "",
        "Structural station-time gaps",
        f"gap_count_over_one_hour: {len(structural_gaps):,}",
        f"omitted_station_hours: {int(structural_gaps['omitted_hour_count'].sum()) if not structural_gaps.empty else 0:,}",
        "The omitted hours are materialized as full_outage only in the operational classification table. The frozen legacy full-outage event table and its duration statistics are intentionally unchanged.",
        "",
        "Partial-outage event duration statistics",
        f"events: {partial_summary['event_count']:,}",
        f"hours: {partial_summary['total_hours']:,}",
        f"mean_duration_hours: {partial_summary['mean_hours']:.2f}",
        f"median_duration_hours: {partial_summary['median_hours']:.2f}",
        f"max_duration_hours: {partial_summary['max_hours']:.0f}",
        "duration_buckets:",
        _duration_bucket_counts(partial_events).to_string(),
        "",
        "Sensor-group availability among transmitting hours",
        group_availability.to_string(index=False),
    ]
    ensure_directories()
    AVAILABILITY_REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _print_summary(events: pd.DataFrame) -> None:
    duration_hours = pd.to_numeric(
        events["duration_hours"],
        errors="coerce",
    ).fillna(0)

    print("Availability events complete.")
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
    classification, partial_events, structural_gaps = (
        write_operational_availability_outputs(hourly_row_states)
    )

    from src.availability.build_station_reliability_summary import (
        write_station_reliability_summary,
    )

    write_station_reliability_summary(
        availability_classification=classification,
        full_events=events,
        partial_events=partial_events,
    )
    write_availability_report(
        events,
        classification,
        partial_events,
        structural_gaps,
    )
    _print_summary(events)
    print(f"Wrote {AVAILABILITY_CLASSIFICATION_PATH}")
    print(f"Wrote {PARTIAL_OUTAGE_EVENTS_PATH}")
    print(f"Wrote {STRUCTURAL_AVAILABILITY_GAPS_PATH}")
    print(f"Wrote {AVAILABILITY_REPORT_PATH}")


if __name__ == "__main__":
    main()
