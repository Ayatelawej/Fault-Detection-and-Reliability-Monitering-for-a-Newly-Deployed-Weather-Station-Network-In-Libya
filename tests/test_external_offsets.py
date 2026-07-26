from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.rules.external_offsets import (
    add_hour_flags,
    build_offset_queue,
    detect_offset_channel,
    detect_ratio,
    fleet_frame,
    offset_episodes_for_station,
    offset_score,
    offset_status,
    ratio_episodes_for_station,
)
from src.rules.review_queue import OUTPUT_COLUMNS


def test_fleet_frame_excludes_timestamps_below_minimum_station_count() -> None:
    frame = pd.DataFrame(
        {
            "station_id": ["A", "B", "A", "B", "C"],
            "time_utc": pd.to_datetime(
                [
                    "2026-01-01 00:00",
                    "2026-01-01 00:00",
                    "2026-01-01 01:00",
                    "2026-01-01 01:00",
                    "2026-01-01 01:00",
                ],
                utc=True,
            ),
            "base_pressure": [1.0, 3.0, 1.0, 3.0, 5.0],
        }
    )

    result = fleet_frame(frame, "pressure", min_stations=3)

    assert pd.isna(result.loc[0, "fleet_median"])
    assert result.loc[1, "fleet_median"] == pytest.approx(3.0)
    assert result.loc[1, "fleet_mad"] == pytest.approx(2.0)


def test_offset_score_math_is_exact() -> None:
    station = pd.DataFrame(
        {
            "station_id": ["A"],
            "time_utc": pd.to_datetime(["2026-01-01 00:00"], utc=True),
            "base_pressure": [12.0],
            "bmad_pressure": [0.2],
        }
    )
    fleet = pd.DataFrame(
        {
            "time_utc": pd.to_datetime(["2026-01-01 00:00"], utc=True),
            "fleet_median": [10.0],
            "fleet_mad": [0.5],
        }
    )

    result = offset_score(station, fleet, "pressure", spread_floor=1.0)

    assert result.loc[0, "offset_score"] == pytest.approx(2.0)
    assert result.loc[0, "offset_abs_score"] == pytest.approx(2.0)


def test_gap_merge_and_min_duration_rules() -> None:
    start = pd.Timestamp("2026-01-01 00:00", tz="UTC")
    hours_a = list(range(101))
    hours_b = list(range(152, 253))
    hours_short = [400, 401]
    all_hours = hours_a + hours_b + hours_short
    frame = pd.DataFrame(
        {
            "time_utc": [start + pd.Timedelta(hours=h) for h in all_hours],
            "offset_flag": [True] * len(all_hours),
            "offset_flag_high": [True] * len(all_hours),
            "offset_level": [10.0] * len(all_hours),
            "offset_score": [3.5] * len(all_hours),
            "offset_bmad": [0.2] * len(all_hours),
        }
    )

    result = offset_episodes_for_station(frame, "STA", "pressure", gap_hours=72, min_days=7)

    assert len(result) == 1
    assert result.loc[0, "duration_hours"] == pytest.approx(253.0)
    assert result.loc[0, "tier"] == "HIGH"


def test_insufficient_status_uses_fleet_valid_hours() -> None:
    scored = pd.DataFrame(
        {
            "base_pressure": [1.0, 1.0, np.nan, 1.0],
            "fleet_median": [0.0, 0.0, 0.0, np.nan],
        }
    )
    episodes = pd.DataFrame()

    result = offset_status(scored, episodes, "pressure", min_hours=3)

    assert result == "insufficient"


def test_ratio_episode_rule_detects_both_directions() -> None:
    over = pd.DataFrame(
        {
            "time_utc": pd.date_range("2026-01-01", periods=8, freq="D", tz="UTC"),
            "ratio_rolling_median": [0.9] * 8,
            "rel_ratio_rolling_median": [1.3] * 8,
        }
    )
    under = pd.DataFrame(
        {
            "time_utc": pd.date_range("2026-02-01", periods=8, freq="D", tz="UTC"),
            "ratio_rolling_median": [0.9] * 8,
            "rel_ratio_rolling_median": [0.7] * 8,
        }
    )

    over_result = ratio_episodes_for_station(over, "STA")
    under_result = ratio_episodes_for_station(under, "STA")

    assert over_result.loc[0, "direction"] == "over"
    assert over_result.loc[0, "rel_p50"] == pytest.approx(1.3)
    assert over_result.loc[0, "ratio_p50"] == pytest.approx(0.9)
    assert under_result.loc[0, "direction"] == "under"
    assert under_result.loc[0, "rel_p50"] == pytest.approx(0.7)
    assert under_result.loc[0, "ratio_p50"] == pytest.approx(0.9)


def test_queue_writer_uses_review_queue_schema() -> None:
    episodes = pd.DataFrame(
        [
            {
                "station_id": "STA",
                "channel": "pressure",
                "start_hour": pd.Timestamp("2026-01-01 00:00", tz="UTC"),
                "end_hour": pd.Timestamp("2026-01-08 00:00", tz="UTC"),
                "duration_hours": 169.0,
                "n_flagged_hours": 10,
                "tier": "HIGH",
                "level_p10": -11.0,
                "level_p50": -10.0,
                "level_p90": -9.0,
                "mean_score": -4.0,
                "peak_abs_score": 5.0,
                "median_bmad": 0.2,
                "episode_type": "offset",
            }
        ]
    )

    result = build_offset_queue(episodes)

    assert result.columns.tolist() == OUTPUT_COLUMNS
    assert result.loc[0, "label"] == "calibration_offset"
    assert result.loc[0, "dominant_detector"] == "external_offset"
    assert "channel=pressure" in result.loc[0, "reasons"]


def test_physical_floor_blocks_flag_when_level_below_floor() -> None:
    scored = pd.DataFrame(
        {
            "time_utc": pd.to_datetime(["2026-01-01 00:00"], utc=True),
            "offset_abs_score": [5.0],
            "offset_score": [5.0],
            "offset_level": [1.5],
            "offset_bmad": [0.1],
        }
    )

    result = add_hour_flags(scored, "pressure")

    assert not result.loc[0, "offset_flag"]


def test_density_rule_drops_sparse_episodes() -> None:
    start = pd.Timestamp("2026-01-01 00:00", tz="UTC")
    frame = pd.DataFrame(
        {
            "time_utc": [start, start + pd.Timedelta(hours=335)],
            "offset_flag": [True, True],
            "offset_flag_high": [True, True],
            "offset_level": [10.0, 10.0],
            "offset_score": [4.0, 4.0],
            "offset_bmad": [0.2, 0.2],
        }
    )

    result = offset_episodes_for_station(
        frame, "STA", "pressure", gap_hours=400, min_days=7
    )

    assert result.empty


def test_insufficient_suppresses_episodes() -> None:
    n = 100
    start = pd.Timestamp("2026-01-01 00:00", tz="UTC")
    times = pd.to_datetime(
        [start + pd.Timedelta(hours=i) for i in range(n)], utc=True
    )
    rows = []
    for sid in [f"S{i:02d}" for i in range(10)]:
        for t in times:
            rows.append(
                {
                    "station_id": sid,
                    "time_utc": t,
                    "base_pressure": 10.0 if sid == "S00" else 0.0,
                    "bmad_pressure": 0.1,
                }
            )
    frame = pd.DataFrame(rows)

    statuses, episodes = detect_offset_channel(frame, "pressure")

    s00 = statuses.loc[statuses["station_id"] == "S00"]
    assert s00["status"].iloc[0] == "insufficient"
    assert episodes.empty


def test_per_channel_gap_uses_channel_specific_value() -> None:
    start = pd.Timestamp("2026-01-01 00:00", tz="UTC")
    hours_a = list(range(169))
    hours_b = list(range(244, 413))
    all_hours = hours_a + hours_b
    frame = pd.DataFrame(
        {
            "time_utc": [start + pd.Timedelta(hours=h) for h in all_hours],
            "offset_flag": [True] * len(all_hours),
            "offset_flag_high": [True] * len(all_hours),
            "offset_level": [10.0] * len(all_hours),
            "offset_score": [3.5] * len(all_hours),
            "offset_bmad": [0.2] * len(all_hours),
        }
    )

    pressure_result = offset_episodes_for_station(
        frame, "STA", "pressure", min_days=7, min_density=0.0
    )
    temp_result = offset_episodes_for_station(
        frame, "STA", "temp", min_days=7, min_density=0.0
    )

    assert len(pressure_result) == 1
    assert len(temp_result) == 2


def test_absolute_ratio_extremes_do_not_flag_without_relative_extreme() -> None:
    ratio = pd.DataFrame(
        {
            "time_utc": pd.date_range("2026-01-01", periods=8, freq="D", tz="UTC"),
            "ratio_rolling_median": [2.0] * 8,
            "rel_ratio_rolling_median": [1.0] * 8,
        }
    )

    result = ratio_episodes_for_station(ratio, "STA")

    assert result.empty


def test_days_without_fleet_support_produce_no_ratio_flags() -> None:
    dates = pd.date_range("2026-01-01", periods=10, freq="D", tz="UTC")
    rows = []
    for date in dates:
        for hour in range(6, 18):
            rows.append(
                {
                    "station_id": "S00",
                    "time_utc": date + pd.Timedelta(hours=hour),
                    "clear_sky_ratio": 2.0,
                }
            )
    frame = pd.DataFrame(rows)
    frame["time_utc"] = pd.to_datetime(frame["time_utc"], utc=True)

    statuses, episodes, daily_ratios = detect_ratio(frame)

    assert episodes.empty
    assert statuses.loc[0, "status"] == "insufficient"
    assert daily_ratios["rel_ratio"].isna().all()


def test_rel_ratio_computed_against_fleet_median() -> None:
    n_days = 10
    dates = pd.date_range("2026-01-01", periods=n_days, freq="D", tz="UTC")
    rows = []
    for date in dates:
        for hour in range(6, 18):
            t = date + pd.Timedelta(hours=hour)
            rows.append(
                {
                    "station_id": "S00",
                    "time_utc": t,
                    "clear_sky_ratio": 0.6,
                    "ref_solar": 400.0,
                }
            )
            for i in range(1, 10):
                rows.append(
                    {
                        "station_id": f"S{i:02d}",
                        "time_utc": t,
                        "clear_sky_ratio": 1.0,
                        "ref_solar": 400.0,
                    }
                )
    frame = pd.DataFrame(rows)
    frame["time_utc"] = pd.to_datetime(frame["time_utc"], utc=True)

    _, _, daily_ratios = detect_ratio(frame)

    s00 = daily_ratios.loc[daily_ratios["station_id"] == "S00"].dropna(subset=["rel_ratio"])
    assert not s00.empty
    assert s00["rel_ratio"].iloc[0] == pytest.approx(0.6, abs=0.02)
