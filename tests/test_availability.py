from __future__ import annotations

import pandas as pd
import pytest

from src.availability.build_network_outage_windows import (
    NETWORK_OUTAGE_MIN_STATIONS,
)
from src.config.paths import (
    AVAILABILITY_EVENTS_PATH,
    HOURLY_ROW_STATES_PATH,
    NETWORK_OUTAGE_WINDOWS_PATH,
)
from src.features.row_state import ROW_STATE_TRUE_OUTAGE

EXPECTED_AVAILABILITY_EVENT_COUNT = 1_670
EXPECTED_OUTAGE_CLASSES = {
    "local",
    "network_midnight",
    "network_other",
    "unknown",
}
EXPECTED_FINAL_OUTAGE_CLASSES = {
    "local",
    "network_midnight",
    "network_other",
}
EXPECTED_NETWORK_OUTAGE_CLASSES = {
    "network_midnight",
    "network_other",
}
EXPECTED_OUTAGE_CLASS_COUNTS = {
    "local": 1_353,
    "network_midnight": 267,
    "network_other": 50,
}


@pytest.fixture(scope="module")
def availability_events_df() -> pd.DataFrame:
    return pd.read_parquet(AVAILABILITY_EVENTS_PATH)


@pytest.fixture(scope="module")
def hourly_row_states_df() -> pd.DataFrame:
    return pd.read_parquet(HOURLY_ROW_STATES_PATH)


@pytest.fixture(scope="module")
def network_outage_windows_df() -> pd.DataFrame:
    return pd.read_csv(NETWORK_OUTAGE_WINDOWS_PATH)


def _event_start_in_window_mask(
    events: pd.DataFrame,
    windows: pd.DataFrame,
    outage_classes: set[str] | None = None,
) -> pd.Series:
    if outage_classes is not None:
        windows = windows.loc[windows["outage_class"].isin(outage_classes)]

    event_starts = pd.to_datetime(
        events["start_utc"],
        utc=True,
        errors="coerce",
    )
    backfill_starts = pd.to_datetime(
        windows["backfill_start_utc"],
        utc=True,
        errors="coerce",
    )
    backfill_ends = pd.to_datetime(
        windows["backfill_end_utc"],
        utc=True,
        errors="coerce",
    )

    in_window = pd.Series(False, index=events.index)
    for backfill_start, backfill_end in zip(backfill_starts, backfill_ends):
        if pd.isna(backfill_start) or pd.isna(backfill_end):
            continue
        in_window = in_window | (
            event_starts.ge(backfill_start)
            & event_starts.le(backfill_end)
        )
    return in_window.fillna(False)


def test_availability_events_output_exists() -> None:
    assert AVAILABILITY_EVENTS_PATH.exists(), (
        f"Missing availability output: {AVAILABILITY_EVENTS_PATH}"
    )


def test_availability_event_count_expected(
    availability_events_df: pd.DataFrame,
) -> None:
    assert len(availability_events_df) == EXPECTED_AVAILABILITY_EVENT_COUNT


def test_availability_event_durations_are_positive(
    availability_events_df: pd.DataFrame,
) -> None:
    duration_hours = pd.to_numeric(
        availability_events_df["duration_hours"],
        errors="coerce",
    )
    assert duration_hours.gt(0).all()


def test_availability_event_start_before_end(
    availability_events_df: pd.DataFrame,
) -> None:
    start_utc = pd.to_datetime(
        availability_events_df["start_utc"],
        utc=True,
        errors="coerce",
    )
    end_utc = pd.to_datetime(
        availability_events_df["end_utc"],
        utc=True,
        errors="coerce",
    )
    assert start_utc.notna().all()
    assert end_utc.notna().all()
    assert start_utc.le(end_utc).all()


def test_availability_event_duration_matches_bounds(
    availability_events_df: pd.DataFrame,
) -> None:
    start_utc = pd.to_datetime(
        availability_events_df["start_utc"],
        utc=True,
        errors="coerce",
    )
    end_utc = pd.to_datetime(
        availability_events_df["end_utc"],
        utc=True,
        errors="coerce",
    )
    duration_hours = pd.to_numeric(
        availability_events_df["duration_hours"],
        errors="coerce",
    )
    computed_duration = ((end_utc - start_utc) / pd.Timedelta(hours=1)) + 1
    assert computed_duration.eq(duration_hours).all()


def test_availability_event_outage_classes_are_final(
    availability_events_df: pd.DataFrame,
) -> None:
    assert set(availability_events_df["outage_class"].unique()) <= (
        EXPECTED_FINAL_OUTAGE_CLASSES
    )
    assert not availability_events_df["outage_class"].eq("unknown").any()


def test_availability_event_hours_match_true_outage_rows(
    availability_events_df: pd.DataFrame,
    hourly_row_states_df: pd.DataFrame,
) -> None:
    duration_hours = pd.to_numeric(
        availability_events_df["duration_hours"],
        errors="coerce",
    )
    true_outage_rows = (
        hourly_row_states_df["row_state"]
        .astype("string")
        .eq(ROW_STATE_TRUE_OUTAGE)
        .fillna(False)
        .sum()
    )
    assert int(duration_hours.sum()) == int(true_outage_rows)


def test_network_outage_windows_output_exists() -> None:
    assert NETWORK_OUTAGE_WINDOWS_PATH.exists(), (
        f"Missing network outage output: {NETWORK_OUTAGE_WINDOWS_PATH}"
    )


def test_network_outage_windows_have_expected_classes(
    network_outage_windows_df: pd.DataFrame,
) -> None:
    assert "outage_class" in network_outage_windows_df.columns
    assert set(network_outage_windows_df["outage_class"].unique()) <= (
        EXPECTED_NETWORK_OUTAGE_CLASSES
    )


def test_network_outage_window_class_matches_start_hour(
    network_outage_windows_df: pd.DataFrame,
) -> None:
    window_start_utc = pd.to_datetime(
        network_outage_windows_df["window_start_utc"],
        utc=True,
        errors="coerce",
    )
    expected_classes = window_start_utc.dt.hour.apply(
        lambda hour: "network_midnight"
        if hour in {22, 23}
        else "network_other"
    )
    assert network_outage_windows_df["outage_class"].eq(expected_classes).all()


def test_network_outage_windows_station_count_threshold(
    network_outage_windows_df: pd.DataFrame,
) -> None:
    station_count = pd.to_numeric(
        network_outage_windows_df["station_count"],
        errors="coerce",
    )
    assert station_count.ge(NETWORK_OUTAGE_MIN_STATIONS).all()


def test_network_outage_window_bounds_are_ordered(
    network_outage_windows_df: pd.DataFrame,
) -> None:
    window_start_utc = pd.to_datetime(
        network_outage_windows_df["window_start_utc"],
        utc=True,
        errors="coerce",
    )
    window_end_utc = pd.to_datetime(
        network_outage_windows_df["window_end_utc"],
        utc=True,
        errors="coerce",
    )
    backfill_start_utc = pd.to_datetime(
        network_outage_windows_df["backfill_start_utc"],
        utc=True,
        errors="coerce",
    )
    backfill_end_utc = pd.to_datetime(
        network_outage_windows_df["backfill_end_utc"],
        utc=True,
        errors="coerce",
    )

    assert window_start_utc.notna().all()
    assert window_end_utc.notna().all()
    assert backfill_start_utc.notna().all()
    assert backfill_end_utc.notna().all()
    assert window_start_utc.le(window_end_utc).all()
    assert backfill_start_utc.le(backfill_end_utc).all()


def test_availability_event_class_count_still_matches_expected(
    availability_events_df: pd.DataFrame,
) -> None:
    class_counts = availability_events_df["outage_class"].value_counts()
    expected_counts = pd.Series(EXPECTED_OUTAGE_CLASS_COUNTS)
    actual_counts = class_counts.reindex(expected_counts.index, fill_value=0)
    assert actual_counts.eq(expected_counts).all()
    assert int(actual_counts.sum()) == EXPECTED_AVAILABILITY_EVENT_COUNT


def test_network_events_fall_in_window_ranges(
    availability_events_df: pd.DataFrame,
    network_outage_windows_df: pd.DataFrame,
) -> None:
    in_window = _event_start_in_window_mask(
        availability_events_df,
        network_outage_windows_df,
    )
    network_event = availability_events_df["outage_class"].isin(
        EXPECTED_NETWORK_OUTAGE_CLASSES,
    )
    assert in_window.loc[network_event].all()


@pytest.mark.parametrize(
    "outage_class",
    ["network_midnight", "network_other"],
)
def test_network_events_fall_in_matching_window_ranges(
    availability_events_df: pd.DataFrame,
    network_outage_windows_df: pd.DataFrame,
    outage_class: str,
) -> None:
    in_matching_window = _event_start_in_window_mask(
        availability_events_df,
        network_outage_windows_df,
        {outage_class},
    )
    matching_events = availability_events_df["outage_class"].eq(outage_class)
    assert in_matching_window.loc[matching_events].all()


def test_local_events_do_not_fall_in_window_ranges(
    availability_events_df: pd.DataFrame,
    network_outage_windows_df: pd.DataFrame,
) -> None:
    in_window = _event_start_in_window_mask(
        availability_events_df,
        network_outage_windows_df,
    )
    local = availability_events_df["outage_class"].eq("local")
    assert (~in_window.loc[local]).all()
