from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.rules.external_residuals import (
    CHANNEL_SPECS,
    compute_external_residuals,
    nearest_snapshot_values,
    trailing_mean_values,
)


def _joined(periods: int, start: str = "2026-01-01") -> pd.DataFrame:
    hours = pd.date_range(start, periods=periods, freq="h", tz="UTC")
    frame = pd.DataFrame(
        {
            "station_id": "STA",
            "time_utc": hours,
        }
    )
    for short in CHANNEL_SPECS:
        frame[f"station_{short}"] = 0.0
        frame[f"ref_{short}"] = 0.0
    return frame


def test_nearest_snapshot_picks_closest_sample_inside_tolerance() -> None:
    five_min = pd.DataFrame(
        {
            "expected_time_utc": pd.to_datetime(
                [
                    "2026-01-01 00:10",
                    "2026-01-01 00:07",
                    "2026-01-01 00:20",
                    "2026-01-01 01:16",
                ],
                utc=True,
            ),
            "temp_avg_c": [10.0, 7.0, 20.0, 16.0],
        }
    )
    targets = pd.to_datetime(
        ["2026-01-01 00:00", "2026-01-01 01:00"],
        utc=True,
    )

    result = nearest_snapshot_values(five_min, targets, "temp_avg_c", 15)

    assert result.iloc[0] == pytest.approx(7.0)
    assert pd.isna(result.iloc[1])


def test_trailing_mean_requires_minimum_slots() -> None:
    times = pd.date_range("2026-01-01 00:05", periods=24, freq="5min", tz="UTC")
    five_min = pd.DataFrame(
        {
            "expected_time_utc": times,
            "solar_radiation_high_wm2": [1.0] * 12 + [2.0] * 7 + [np.nan] * 5,
        }
    )
    targets = pd.to_datetime(
        ["2026-01-01 01:00", "2026-01-01 02:00"],
        utc=True,
    )

    result = trailing_mean_values(
        five_min,
        targets,
        "solar_radiation_high_wm2",
        8,
    )

    assert result.iloc[0] == pytest.approx(1.0)
    assert pd.isna(result.iloc[1])


def test_scalar_residual_baseline_and_z_are_hand_computed() -> None:
    frame = _joined(5)
    frame["station_pressure"] = [10.0, 12.0, 14.0, 16.0, 30.0]

    result = compute_external_residuals(
        frame,
        scalar_window_hours=3,
        scalar_min_hours=3,
        hourbin_window_days=3,
        hourbin_min_days=3,
    )

    assert result.loc[2, "r_pressure"] == pytest.approx(14.0)
    assert result.loc[2, "base_pressure"] == pytest.approx(12.0)
    assert result.loc[2, "bmad_pressure"] == pytest.approx(2.0)
    assert result.loc[2, "z_pressure"] == pytest.approx(1.0)
    assert result.loc[4, "base_pressure"] == pytest.approx(16.0)
    assert result.loc[4, "bmad_pressure"] == pytest.approx(2.0)
    assert result.loc[4, "z_pressure"] == pytest.approx(7.0)


def test_hourbin_baseline_absorbs_hour_of_day_bias() -> None:
    frame = _joined(24 * 20)
    hours = frame["time_utc"].dt.hour
    frame["station_temp"] = hours.astype(float) * 0.2
    frame["ref_temp"] = 0.0

    result = compute_external_residuals(
        frame,
        scalar_window_hours=3,
        scalar_min_hours=3,
        hourbin_window_days=10,
        hourbin_min_days=5,
    )
    mature = result.loc[
        result["time_utc"].ge(result["time_utc"].min() + pd.Timedelta(days=10))
    ]

    assert mature["z_temp"].dropna().abs().max() == pytest.approx(0.0)


def test_missing_station_hours_keep_residual_and_z_nan() -> None:
    frame = _joined(5)
    frame["station_pressure"] = [1.0, 1.0, np.nan, 1.0, 2.0]

    result = compute_external_residuals(
        frame,
        scalar_window_hours=3,
        scalar_min_hours=3,
        hourbin_window_days=3,
        hourbin_min_days=3,
    )

    assert pd.isna(result.loc[2, "r_pressure"])
    assert pd.isna(result.loc[2, "z_pressure"])


def test_solar_daylight_and_clear_sky_masks() -> None:
    frame = _joined(5)
    frame["station_solar"] = [0.0, 5.0, 20.0, 600.0, 800.0]
    frame["ref_solar"] = [0.0, 5.0, 11.0, 300.0, 400.0]

    result = compute_external_residuals(
        frame,
        scalar_window_hours=3,
        scalar_min_hours=3,
        hourbin_window_days=1,
        hourbin_min_days=1,
    )

    assert result["r_solar"].isna().tolist()[:2] == [True, True]
    assert result.loc[2, "r_solar"] == pytest.approx(9.0)
    assert pd.isna(result.loc[2, "clear_sky_ratio"])
    assert result.loc[3, "clear_sky_ratio"] == pytest.approx(2.0)
    assert result.loc[4, "clear_sky_ratio"] == pytest.approx(2.0)


def test_mad_floor_engages_when_mad_is_zero() -> None:
    frame = _joined(4)
    frame["station_pressure"] = [1.0, 1.0, 1.0, 2.0]

    result = compute_external_residuals(
        frame,
        scalar_window_hours=3,
        scalar_min_hours=3,
        hourbin_window_days=3,
        hourbin_min_days=3,
    )

    assert result.loc[3, "base_pressure"] == pytest.approx(1.0)
    assert result.loc[3, "bmad_pressure"] == pytest.approx(0.0)
    assert result.loc[3, "z_pressure"] == pytest.approx(1.25)
