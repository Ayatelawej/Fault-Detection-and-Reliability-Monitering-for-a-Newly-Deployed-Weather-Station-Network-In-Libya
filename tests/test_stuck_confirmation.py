from __future__ import annotations

import pandas as pd

from src.rules.stuck_confirmation import (
    confirm_episode,
    dominant_stuck_channel,
    load_five_min_window,
    select_dominant_stuck_channels,
)


BASE_TIME = pd.Timestamp("2026-01-01 00:00:00", tz="UTC")


def _write_station_file(tmp_path, station_id, values, column="windspeed_avg_kmh"):
    frame = pd.DataFrame(
        {
            "expected_time_utc": pd.date_range(
                BASE_TIME,
                periods=len(values),
                freq="5min",
                tz="UTC",
            ),
            column: values,
        }
    )
    path = tmp_path / f"{station_id}_complete.csv"
    frame.to_csv(path, index=False)
    return path


def _row(station_id="STA", channel="windspeed_avg_kmh", duration_hours=2):
    return {
        "station_id": station_id,
        "channel": channel,
        "start_hour": BASE_TIME,
        "end_hour": BASE_TIME + pd.Timedelta(hours=duration_hours - 1),
        "duration_hours": duration_hours,
    }


def test_constant_five_min_series_confirms_stuck(tmp_path):
    _write_station_file(tmp_path, "STA", [0.0] * 24)

    result = confirm_episode(_row(), five_min_dir=tmp_path)

    assert result["status"] == "confirmed_stuck"
    assert result["confirmed_stuck_5min"] is True
    assert result["modal_fraction"] == 1.0
    assert result["longest_constant_run_hours"] == 2.0


def test_varying_five_min_series_is_not_constant(tmp_path):
    _write_station_file(tmp_path, "STA", list(range(24)))

    result = confirm_episode(_row(), five_min_dir=tmp_path)

    assert result["status"] == "not_constant"
    assert result["confirmed_stuck_5min"] is False


def test_mostly_constant_series_uses_modal_threshold(tmp_path):
    _write_station_file(tmp_path, "OK", [4.0] * 119 + [5.0])
    _write_station_file(tmp_path, "BAD", [4.0] * 118 + [5.0, 6.0])

    confirmed = confirm_episode(
        _row(station_id="OK", duration_hours=10),
        five_min_dir=tmp_path,
    )
    rejected = confirm_episode(
        _row(station_id="BAD", duration_hours=10),
        five_min_dir=tmp_path,
    )

    assert confirmed["status"] == "confirmed_stuck"
    assert confirmed["modal_fraction"] >= 0.99
    assert rejected["status"] == "not_constant"
    assert rejected["modal_fraction"] < 0.99


def test_too_few_present_readings_is_insufficient(tmp_path):
    _write_station_file(tmp_path, "STA", [0.0] * 23)

    result = confirm_episode(_row(), five_min_dir=tmp_path)

    assert result["status"] == "insufficient_5min_data"
    assert result["confirmed_stuck_5min"] is False


def test_unmapped_channel_is_reported(tmp_path):
    _write_station_file(tmp_path, "STA", [0.0] * 24)

    result = confirm_episode(
        _row(channel="unknown_channel"),
        five_min_dir=tmp_path,
    )

    assert result["status"] == "unmapped_channel"
    assert result["confirmed_stuck_5min"] is False


def test_load_five_min_window_excludes_outside_timestamps(tmp_path):
    frame = pd.DataFrame(
        {
            "expected_time_utc": [
                BASE_TIME - pd.Timedelta(minutes=5),
                BASE_TIME,
                BASE_TIME + pd.Timedelta(minutes=5),
                BASE_TIME + pd.Timedelta(minutes=10),
            ],
            "windspeed_avg_kmh": [1.0, 2.0, 3.0, 4.0],
        }
    )
    frame.to_csv(tmp_path / "STA_complete.csv", index=False)

    result = load_five_min_window(
        "STA",
        BASE_TIME,
        BASE_TIME + pd.Timedelta(minutes=5),
        tmp_path,
    )

    assert result["windspeed_avg_kmh"].tolist() == [2.0, 3.0]


def test_select_dominant_stuck_channel_uses_stuck_reason_events():
    episodes = pd.DataFrame(
        [
            {
                "station_id": "STA",
                "start_hour": BASE_TIME,
                "end_hour": BASE_TIME + pd.Timedelta(hours=3),
                "duration_hours": 4,
                "affected_channels": "windspeed_avg_kmh|winddir_sin",
            }
        ]
    )
    events = pd.DataFrame(
        [
            {
                "station_id": "STA",
                "channel": "winddir_sin",
                "start_hour": BASE_TIME,
                "end_hour": BASE_TIME + pd.Timedelta(hours=3),
                "duration_hours": 4,
                "min_rolling_variance": 0.0,
                "reasons": "stuck_variance_zero",
            },
            {
                "station_id": "STA",
                "channel": "windspeed_avg_kmh",
                "start_hour": BASE_TIME,
                "end_hour": BASE_TIME + pd.Timedelta(hours=1),
                "duration_hours": 2,
                "min_rolling_variance": 0.0,
                "reasons": "stuck_variance_zero",
            },
            {
                "station_id": "STA",
                "channel": "temp_avg_c",
                "start_hour": BASE_TIME,
                "end_hour": BASE_TIME + pd.Timedelta(hours=3),
                "duration_hours": 4,
                "min_rolling_variance": 0.0,
                "reasons": "iforest_outlier",
            },
        ]
    )

    result = select_dominant_stuck_channels(episodes, events)

    assert result.iloc[0]["channel"] == "winddir_sin"
    assert result.iloc[0]["channel_resolution"] == "stuck_event"


def test_dominant_stuck_channel_uses_row_channel_when_reason_is_present():
    row = {
        "station_id": "STA",
        "channel": "windspeed_avg_kmh",
        "start_hour": BASE_TIME,
        "end_hour": BASE_TIME + pd.Timedelta(hours=1),
        "duration_hours": 2,
        "dominant_detector": "stuck",
        "reasons": "stuck_variance_zero",
    }

    result = dominant_stuck_channel(row)

    assert result == "windspeed_avg_kmh"
